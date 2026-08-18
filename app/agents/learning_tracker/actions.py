"""What the tutor is allowed to *do*, rather than only say.

Until this existed the agent could recommend a next step perfectly and the
learner still had to go and find the button. These are the buttons, as tools.

Three rules decide what belongs here, and the first is the important one.

**Nothing may open a gate.** This product's claims rest on a small number of
things a learner cannot talk their way past: a digest is acknowledged only by
passing its recall check, a topic is completed only by passing its checkpoint,
and a failed checkpoint owes revision before a retry. A tool that marked a digest
read, set a topic complete, or answered on the learner's behalf would not be a
convenience — it would make every mastery score in the app a claim about nothing.
So none exists, and `set_topic_progress` is reachable here only as
`in_progress`, exactly as the chat path has always been restricted.

**Everything is reversible.** No approval interrupt guards these, because a
confirmation dialog on every action would rebuild by hand the clicking this set
exists to remove. That trade is only honest while the worst outcome of a
misunderstood instruction is one tap to undo — so pausing a roadmap qualifies and
deleting one never will.

**No tool takes a user_id.** They close over it instead, bound per turn by
`build_action_tools`. A model cannot address another learner's data because there
is no argument through which to try, which is a stronger guarantee than asking it
not to.

Every tool answers in a plain sentence, including when it declines. The model
relays that to the learner, so a refusal has to read like something a person
would say — the server's own reasons, not an error code.
"""

import logging
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .repository import (
    ActiveRoadmapLimit,
    create_note,
    fetch_roadmap,
    resolve_roadmap_id,
    set_digest_schedule,
    set_roadmap_status,
    start_topic,
)
from .triggers import pull_next_digest

logger = logging.getLogger(__name__)


# ── argument schemas ─────────────────────────────────────────────────────────
class RoadmapArg(BaseModel):
    roadmapId: str = Field(description="The roadmap's id, from list_topics or the situation.")


class OptionalRoadmapArg(BaseModel):
    roadmapId: Optional[str] = Field(
        default=None,
        description="The roadmap's id. Omit to use whichever roadmap is current.",
    )


class TopicArg(BaseModel):
    roadmapId: str = Field(description="The roadmap's id.")
    topicId: str = Field(description="The topic's id, exactly as list_topics reported it.")


class NoteArg(BaseModel):
    roadmapId: str = Field(description="The roadmap's id.")
    topicId: str = Field(description="The topic the note belongs to.")
    body: str = Field(description="What to write down, in the learner's own terms.")
    kind: str = Field(
        default="note",
        description="One of: note, snippet, link, question. Use 'question' for something to come back to.",
    )


class ScheduleArg(BaseModel):
    hour: int = Field(description="Hour of day to send the daily digest, 0-23, in their timezone.")
    timezone: Optional[str] = Field(
        default=None,
        description="IANA timezone name, e.g. Europe/London. Omit to keep the current one.",
    )


def build_action_tools(user_id: str) -> list[StructuredTool]:
    """The tools, bound to one learner for one turn."""

    async def _list_topics(roadmapId: str) -> str:
        roadmap = await fetch_roadmap(roadmapId, user_id)
        if not roadmap:
            return "No roadmap with that id."
        rows = sorted(roadmap.get("topics") or [], key=lambda t: t.get("order") or 0)
        if not rows:
            return "That roadmap has no topics."
        lines = [
            f'{t.get("order")}. {t.get("title")} — id={t.get("id")}, '
            f'status={t.get("progress_status") or "not_started"}'
            for t in rows
        ]
        return f'Topics on "{roadmap.get("title")}":\n' + "\n".join(lines)

    async def _start_topic(roadmapId: str, topicId: str) -> str:
        roadmap = await fetch_roadmap(roadmapId, user_id)
        if not roadmap:
            return "No roadmap with that id."
        topic = next(
            (t for t in roadmap.get("topics") or [] if t.get("id") == topicId), None
        )
        if not topic:
            return "No topic with that id on that roadmap."
        # Exactly one topic is underway at a time; the repository demotes whoever
        # held the slot. Saying so matters — the learner is giving something up.
        previous = next(
            (
                t
                for t in roadmap.get("topics") or []
                if t.get("progress_status") == "in_progress" and t.get("id") != topicId
            ),
            None,
        )
        ok = await start_topic(roadmapId, topicId, user_id)
        if not ok:
            return "Could not start that topic just now."
        moved = (
            f' "{previous.get("title")}" is no longer the one underway.'
            if previous
            else ""
        )
        return (
            f'Started "{topic.get("title")}" — daily lessons for it begin now.{moved}'
        )

    async def _pull_next_lesson(roadmapId: Optional[str] = None) -> str:
        outcome = await pull_next_digest(user_id, roadmapId)
        if "digest" in outcome:
            digest = outcome["digest"]
            bullets = digest.get("bullets") or []
            head = f'Wrote the next lesson on "{digest.get("topicTitle")}"'
            # The recall check is the reason a digest can't just be handed over in
            # chat and ticked off — say where it is rather than reproducing it.
            gate = (
                " It carries a recall check, which is on the digest itself."
                if digest.get("quiz")
                else ""
            )
            return f"{head} ({len(bullets)} points). It's waiting on their Today screen.{gate}"
        return outcome["refused"]

    async def _pause_roadmap(roadmapId: str) -> str:
        roadmap = await fetch_roadmap(roadmapId, user_id)
        if not roadmap:
            return "No roadmap with that id."
        ok = await set_roadmap_status(roadmapId, user_id, "paused")
        return (
            f'Paused "{roadmap.get("title")}" — no more daily lessons until it is resumed.'
            if ok
            else "Could not pause that roadmap just now."
        )

    async def _resume_roadmap(roadmapId: str) -> str:
        roadmap = await fetch_roadmap(roadmapId, user_id)
        if not roadmap:
            return "No roadmap with that id."
        try:
            ok = await set_roadmap_status(roadmapId, user_id, "active")
        except ActiveRoadmapLimit as limit:
            # The cap exists so attention isn't split; report it as the choice it
            # is rather than as a failure, and name what holds the slots.
            holders = ", ".join(f'"{h.get("title")}"' for h in (limit.active or []))
            return (
                f"Cannot resume it — {limit.limit} roadmaps already running"
                + (f" ({holders})" if holders else "")
                + ". One of those has to be paused first."
            )
        return (
            f'Resumed "{roadmap.get("title")}" — daily lessons start again.'
            if ok
            else "Could not resume that roadmap just now."
        )

    async def _save_note(
        roadmapId: str, topicId: str, body: str, kind: str = "note"
    ) -> str:
        if kind not in ("note", "snippet", "link", "question"):
            kind = "note"
        note = await create_note(user_id, roadmapId, topicId, kind, body)
        if not note:
            return "Could not save that note."
        return f"Saved it as a {kind} against that topic. It's on their Notes screen."

    async def _set_digest_time(hour: int, timezone: Optional[str] = None) -> str:
        try:
            await set_digest_schedule(
                user_id, hour=hour, timezone_name=timezone, enabled=True
            )
        except ValueError as e:
            return str(e)
        where = f" ({timezone})" if timezone else ""
        return f"Daily lessons will arrive at {hour:02d}:00{where}."

    spec = [
        (
            _list_topics,
            "list_topics",
            "List a roadmap's topics with their ids and progress. Call this first "
            "whenever an action needs a topicId — never guess one.",
            RoadmapArg,
        ),
        (
            _start_topic,
            "start_topic",
            "Pick up a topic and begin its daily lessons. Exactly one topic is "
            "underway at a time, so this replaces whichever was.",
            TopicArg,
        ),
        (
            _pull_next_lesson,
            "pull_next_lesson",
            "Write and deliver the next daily lesson now instead of waiting for "
            "the schedule. Declines when one is already waiting, when a recall "
            "check is outstanding, or when the topic is fully taught.",
            OptionalRoadmapArg,
        ),
        (
            _pause_roadmap,
            "pause_roadmap",
            "Stop a roadmap's daily lessons, keeping all progress. Reversible.",
            RoadmapArg,
        ),
        (
            _resume_roadmap,
            "resume_roadmap",
            "Start a paused roadmap's daily lessons again. May decline if too "
            "many roadmaps are already running.",
            RoadmapArg,
        ),
        (
            _save_note,
            "save_note",
            "Write something down against a topic — a jotting, a snippet, a link, "
            "or a question to come back to.",
            NoteArg,
        ),
        (
            _set_digest_time,
            "set_digest_time",
            "Change what time of day the daily lesson arrives, and switch daily "
            "lessons on.",
            ScheduleArg,
        ),
    ]

    return [
        StructuredTool.from_function(
            coroutine=fn, name=name, description=description, args_schema=schema
        )
        for fn, name, description, schema in spec
    ]
