"""The learner's situation, in a form a prompt can read.

The agent used to see two things: the learner's profile, and their message. That
is enough to answer a question and nothing like enough to coach — it could not
say "you owe a revision on Ownership" because nothing had told it. Every
judgement of that kind lived in the client instead, as a table of copy keyed on
`blocked_reason`, which is why a tutor sitting beside a screen full of advice
could only ever answer what it was asked.

This module is the other half: what is running, what is blocked and why, what has
come due, and what is not sticking — assembled from exactly the reads the home
screen already makes, so the tutor and the screen cannot disagree about where the
learner is standing.

Two rules hold here.

**`probe` never leaves.** A misconception carries the instruction for writing a
question that catches it. Everything downstream of this module produces text the
learner reads, so the field is dropped at this boundary rather than trusted to
each caller — the same reason `get_misconception_report` drops it on the way out
of the API.

**Nothing here is authoritative.** It is a snapshot for a prompt to reason over,
never a source of truth to act on: anything that mutates re-reads and re-checks
server-side. A stale context can make the agent say something slightly out of
date; it must never let it do something wrong.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from app.agents.memory_store import get_profile

from .repository import (
    LEARNING_NS,
    due_reviews,
    learning_focus,
    learning_stats,
    list_misconceptions,
)

logger = logging.getLogger(__name__)

# Caps, not pagination: this is material for a prompt, and a prompt that lists
# every due review buries the one that matters. The counts are reported in full
# alongside the samples so the model can still say "and 9 others".
MAX_REVIEWS = 5
MAX_WEAKEST = 3
MAX_MISCONCEPTIONS = 6

# What each blocked_reason means for what the learner can do about it. The model
# gets the explanation rather than the enum: "cap_reached" tells it nothing, and
# a model guessing at an identifier invents remedies that don't exist.
BLOCKED_MEANING = {
    "no_roadmap": "they have no roadmap yet — one needs building before anything else happens",
    "cap_reached": "unread digests have hit the cap on this topic; no more will be sent until they catch up",
    "awaiting_quiz": "a recall check on an earlier digest is unanswered, which is holding the next one",
    "needs_revision": "a checkpoint was failed here; a revision digest must be read before a retry is allowed",
    "needs_review": "the digests have covered this topic — only the checkpoint stands between them and the next one",
    "roadmap_complete": "every topic on this roadmap is done",
    "digests_off": "the daily digest is switched off, though one can still be pulled by hand",
}


def _overdue_days(due_at: Optional[str], now: datetime) -> int:
    """How long a review has been sitting past its date, in whole days."""
    if not due_at:
        return 0
    try:
        due = datetime.fromisoformat(due_at)
        if due.tzinfo is None:
            due = due.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0
    return max(int((now - due).total_seconds() // 86_400), 0)


async def learner_context(user_id: str, *, with_stats: bool = False) -> dict:
    """Everything about where this learner is standing right now.

    `with_stats` adds mastery and the streak, and is off by default because
    `learning_stats` sweeps every roadmap and every graded attempt the learner
    has ever made. That is the right cost for a briefing, which is generated once
    per change of situation, and the wrong cost for a chat turn, which happens
    whenever somebody types. What a chat turn needs — what is running, what is
    blocked, what is due — is in the cheap half.

    Never raises. A context that fails to assemble degrades the agent from
    situated to generic, which is where it started; taking the turn down with it
    would be a worse trade.
    """
    import asyncio

    now = datetime.now(timezone.utc)

    async def safe(coro, default):
        try:
            return await coro
        except Exception as e:
            logger.error("learner_context partial failure user=%s: %s", user_id, e)
            return default

    jobs = [
        safe(learning_focus(user_id), {}),
        safe(due_reviews(user_id, limit=MAX_REVIEWS * 4), []),
        safe(list_misconceptions(user_id), []),
        safe(get_profile(user_id, LEARNING_NS), {}),
    ]
    if with_stats:
        jobs.append(safe(learning_stats(user_id), {}))

    results = await asyncio.gather(*jobs)
    focus, reviews, misconceptions, profile = results[:4]
    stats = results[4] if with_stats else {}

    roadmaps = [
        {
            "roadmapId": item.get("roadmapId"),
            "title": item.get("roadmapTitle"),
            "topic": (item.get("topic") or {}).get("title"),
            "topicId": (item.get("topic") or {}).get("id"),
            "topic_status": (item.get("topic") or {}).get("progress_status"),
            "completed": (item.get("progress") or {}).get("completed_count"),
            "total": (item.get("progress") or {}).get("total"),
            "percent": (item.get("progress") or {}).get("percent"),
            "unread": item.get("unread"),
            "can_generate": item.get("can_generate"),
            "blocked_reason": item.get("blocked_reason"),
            "blocked_meaning": BLOCKED_MEANING.get(item.get("blocked_reason") or ""),
        }
        for item in (focus.get("roadmaps") or [])
    ]

    due = [
        {
            "roadmapId": r.get("roadmapId"),
            "roadmapTitle": r.get("roadmapTitle"),
            "topicId": r.get("topicId"),
            "title": r.get("title"),
            "overdue_days": _overdue_days(r.get("due_at"), now),
        }
        for r in reviews
    ]

    # Labels only, and `probe` stripped — see the module docstring. `detail` is
    # what the misunderstanding actually is, which is what teaching against it
    # requires; `probe` is how it will be caught, which is the learner's to
    # discover rather than to be told.
    patterns = []
    for entry in misconceptions:
        for p in entry.get("patterns") or []:
            patterns.append(
                {
                    "topic": entry.get("topicTitle"),
                    "roadmap": entry.get("roadmapTitle"),
                    "label": p.get("label"),
                    "detail": p.get("detail"),
                }
            )

    mastery = (stats or {}).get("mastery") or {}

    return {
        "roadmaps": roadmaps,
        "unread_digests": focus.get("unread") or 0,
        "digest_cap": focus.get("cap"),
        "next_digest_at": focus.get("next_at"),
        "account_blocked": focus.get("blocked_reason"),
        "reviews_due": due[:MAX_REVIEWS],
        "reviews_due_total": len(due),
        "misconceptions": patterns[:MAX_MISCONCEPTIONS],
        "misconceptions_total": len(patterns),
        "mastery": (
            {
                "score": mastery.get("score"),
                "trend": mastery.get("trend"),
                "topics_scored": mastery.get("topics_scored"),
                "weakest": [
                    {
                        "title": t.get("title"),
                        "roadmapId": t.get("roadmapId"),
                        "topicId": t.get("topicId"),
                        "mastery": t.get("mastery"),
                        "trend": t.get("trend"),
                    }
                    for t in (mastery.get("weakest") or [])[:MAX_WEAKEST]
                ],
            }
            if with_stats
            else None
        ),
        "streak_days": (stats or {}).get("streak_days") if with_stats else None,
        "profile": {
            "skill_level": (profile or {}).get("skill_level"),
            "explanation_style": (profile or {}).get("preferred_explanation_style"),
            "goals": (profile or {}).get("goals") or [],
            "minutes_per_day": ((profile or {}).get("availability") or {}).get(
                "minutes_per_day"
            ),
        },
    }


def situation_text(ctx: Optional[dict]) -> str:
    """The context as a block of prose for a system prompt.

    Prose rather than the raw dict: a model reading JSON tends to quote its keys
    back ("your blocked_reason is needs_revision"), and the whole point of this
    is that the learner hears a person rather than a record.
    """
    if not ctx:
        return "No information about this learner's current position is available."

    lines: list[str] = []

    roadmaps = ctx.get("roadmaps") or []
    if not roadmaps:
        lines.append("They have no roadmap running.")
    for r in roadmaps:
        bits = [f'Roadmap "{r.get("title")}" — {r.get("percent")}% done']
        if r.get("completed") is not None:
            bits.append(f'{r.get("completed")}/{r.get("total")} topics')
        if r.get("topic"):
            bits.append(f'currently on "{r.get("topic")}" ({r.get("topic_status")})')
        if r.get("unread"):
            bits.append(f'{r.get("unread")} unread digest(s)')
        line = "; ".join(bits) + "."
        if r.get("blocked_meaning"):
            line += f' Blocked: {r.get("blocked_meaning")}.'
        lines.append(line)

    if ctx.get("account_blocked") == "digests_off":
        lines.append("Daily digests are switched off for this account.")

    due = ctx.get("reviews_due") or []
    if due:
        named = ", ".join(
            f'"{d.get("title")}"'
            + (f' ({d.get("overdue_days")}d overdue)' if d.get("overdue_days") else "")
            for d in due
        )
        total = ctx.get("reviews_due_total") or len(due)
        more = f" (and {total - len(due)} more)" if total > len(due) else ""
        lines.append(f"Reviews due: {named}{more}.")

    mastery = ctx.get("mastery")
    if mastery and mastery.get("score") is not None:
        lines.append(
            f'Mastery {mastery.get("score")}% across {mastery.get("topics_scored")} '
            f'graded topic(s), trend {mastery.get("trend")}.'
        )
        weak = [
            f'"{w.get("title")}" ({w.get("mastery")}%)'
            for w in (mastery.get("weakest") or [])
            if (w.get("mastery") or 100) < 70
        ]
        if weak:
            lines.append(f'Not holding well: {", ".join(weak)}.')

    if ctx.get("streak_days"):
        lines.append(f'Current streak: {ctx.get("streak_days")} day(s).')

    patterns = ctx.get("misconceptions") or []
    if patterns:
        lines.append("Recurring misunderstandings the teaching should work against:")
        for p in patterns:
            lines.append(f'  - [{p.get("topic")}] {p.get("label")}: {p.get("detail")}')

    profile = ctx.get("profile") or {}
    known = [
        f'level {profile.get("skill_level")}' if profile.get("skill_level") else None,
        (
            f'prefers {profile.get("explanation_style")} explanations'
            if profile.get("explanation_style")
            else None
        ),
        (
            f'{profile.get("minutes_per_day")} min/day available'
            if profile.get("minutes_per_day")
            else None
        ),
    ]
    known = [k for k in known if k]
    if known:
        lines.append("Learner: " + ", ".join(known) + ".")
    if profile.get("goals"):
        lines.append("Goals: " + ", ".join(profile["goals"]) + ".")

    return "\n".join(lines)


def situation_key(ctx: Optional[dict]) -> str:
    """A stable fingerprint of the *decisions* this context implies.

    What a briefing should say changes when the learner's position changes, not
    when the clock moves. So `next_digest_at` is deliberately excluded: it ticks
    forward every day and would expire a briefing that is still accurate. The
    same reasoning is why nothing here is a timestamp — the fields are the ones
    that would change the advice.
    """
    if not ctx:
        return "empty"

    salient = {
        "roadmaps": [
            [
                r.get("roadmapId"),
                r.get("topicId"),
                r.get("topic_status"),
                r.get("blocked_reason"),
                r.get("unread"),
                r.get("can_generate"),
                r.get("percent"),
            ]
            for r in (ctx.get("roadmaps") or [])
        ],
        "reviews": sorted(d.get("topicId") or "" for d in (ctx.get("reviews_due") or [])),
        "unread": ctx.get("unread_digests"),
        "account_blocked": ctx.get("account_blocked"),
        "mastery": (ctx.get("mastery") or {}).get("score"),
        "streak": ctx.get("streak_days"),
        "misconceptions": sorted(
            p.get("label") or "" for p in (ctx.get("misconceptions") or [])
        ),
    }
    blob = json.dumps(salient, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
