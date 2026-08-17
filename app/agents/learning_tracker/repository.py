"""MongoDB persistence and pure domain logic for the learning-tracker agent.

Three rules this module enforces:

1. Every read and write is scoped by `user_id`. `roadmapId` comes from the
   client, so a query keyed on `_id` alone lets any caller reach — and on a
   modify, take over — another learner's roadmap.
2. The server owns identity, lifecycle, and progress. `materialize_roadmap` and
   `merge_roadmap` mint topic ids and progress fields; the LLM never supplies
   them (see state.py).
3. An edit never costs the learner their progress — that's `merge_roadmap`.

Progress lives in `progress_status` only. Documents written before that field
existed (a bare `covered: bool`) are not supported and are not read.
"""

import logging
import re
import uuid
from datetime import date, timedelta, datetime, timezone
from math import ceil
from typing import Optional

from bson import ObjectId

from app.core.config import (
    CHECKPOINT_MAX_ATTEMPTS_PER_DAY,
    CHECKPOINT_PASS_SCORE,
    CHECKPOINT_RETRY_COOLDOWN_MINUTES,
    DIGEST_MAX_UNREAD,
    DIGEST_MIN_BEFORE_CHECKPOINT,
    DIGEST_QUIZ_EVERY,
    FEYNMAN_LADDER_BONUS,
    MAX_ACTIVE_ROADMAPS,
    MISCONCEPTION_EVIDENCE_WINDOW,
)
from app.database import get_db
from app.agents.memory_store import extract_and_save
from app.agents.personal_assistant.repository import TODOS
from app.agents.trigger_store import next_run_at, user_zone
from .state import (
    DONE_STATUSES,
    LearnerMemory,
    Resource,
    RoadmapDraft,
    RoadmapOutput,
    Stage,
    TopicNode,
)

logger = logging.getLogger(__name__)

# Sub-document of the shared `memories` doc that the learning tracker owns, so
# its extraction schema can't overwrite the PA's or meal planner's fields.
LEARNING_NS = "learning"


def get_monday(today: Optional[date] = None) -> str:
    # weekday(): Monday=0, Sunday=6
    if today is None:
        today = date.today()
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


# ── progress reading ─────────────────────────────────────────────────────────
def active_topic(roadmap: Optional[dict]) -> Optional[dict]:
    """The next topic the learner is working towards: lowest `order` that is
    neither completed nor skipped."""
    topics = sorted((roadmap or {}).get("topics", []), key=lambda t: t.get("order", 0))
    for t in topics:
        if t.get("progress_status") not in DONE_STATUSES:
            return t
    return None


def roadmap_progress(roadmap: Optional[dict]) -> dict:
    """How far along a roadmap is. Pure arithmetic over `progress_status` and
    `order` — no LLM, so it can't miscount or invent a next topic."""
    topics = (roadmap or {}).get("topics", []) or []
    completed = [t for t in topics if t.get("progress_status") == "completed"]
    nxt = active_topic(roadmap)
    return {
        "next_topic": nxt.get("title") if nxt else None,
        "next_topic_id": nxt.get("id") if nxt else None,
        "completed_count": len(completed),
        "remaining": len(topics) - len(completed),
        "total": len(topics),
        "percent": round(len(completed) / len(topics) * 100) if topics else 0,
    }


# ── personalization ──────────────────────────────────────────────────────────
# The profile fields that actually reach the roadmap prompt. Anything outside
# this list can change without making an existing roadmap out of date — which is
# the point of keeping the list explicit rather than snapshotting the whole
# profile and crying "stale" every time a quiz preference changes.
PERSONALIZATION_FIELDS = (
    "skill_level",
    "goals",
    "preferred_resource_types",
    "availability",
    "known_topics",
)


def profile_snapshot(memory: Optional[dict]) -> dict:
    """The personalization inputs, with blanks dropped so an unset field and a
    field set to [] don't read as different."""
    memory = memory or {}
    return {
        k: memory[k]
        for k in PERSONALIZATION_FIELDS
        if memory.get(k) not in (None, [], "", {})
    }


def profile_drift(snapshot: Optional[dict], memory: Optional[dict]) -> list[str]:
    """Which personalization inputs have changed since the roadmap was built.

    Empty for a roadmap generated before we recorded a snapshot: we don't know
    what it was built from, so claiming it's out of date would be a guess.
    """
    if snapshot is None:
        return []
    current = profile_snapshot(memory)
    return sorted(
        k for k in set(current) | set(snapshot) if current.get(k) != snapshot.get(k)
    )


def completion_forecast(
    roadmap: Optional[dict], memory: Optional[dict]
) -> Optional[dict]:
    """Turn the remaining study time into a date, at the learner's stated pace.

    None when there's nothing left to do or no pace on file — a forecast built on
    a default nobody chose is worse than no forecast.
    """
    availability = (memory or {}).get("availability") or {}
    per_day = availability.get("minutes_per_day") or 0
    if per_day <= 0:
        return None

    remaining = sum(
        t.get("estimated_minutes") or 0
        for t in (roadmap or {}).get("topics") or []
        if t.get("progress_status") not in DONE_STATUSES
    )
    if remaining <= 0:
        return None

    days_per_week = min(max(availability.get("days_per_week") or 7, 1), 7)
    study_days = ceil(remaining / per_day)
    # Studying 3 days a week stretches 6 study days across two calendar weeks.
    calendar_days = ceil(study_days * 7 / days_per_week)
    target = date.today() + timedelta(days=calendar_days)

    forecast = {
        "remaining_minutes": remaining,
        "study_days": study_days,
        "calendar_days": calendar_days,
        "target_date": target.isoformat(),
        "minutes_per_day": per_day,
        "days_per_week": days_per_week,
        "deadline": availability.get("deadline"),
        "on_track": None,
    }
    deadline = availability.get("deadline")
    if deadline:
        try:
            forecast["on_track"] = target <= date.fromisoformat(deadline[:10])
        except ValueError:
            pass
    return forecast


# ── draft → persisted roadmap ────────────────────────────────────────────────
def materialize_roadmap(
    draft: RoadmapDraft,
    status: str = "active",
    personalization: Optional[dict] = None,
) -> RoadmapOutput:
    """Turn an LLM draft into a persistable roadmap. Ids, status, timestamps and
    progress are assigned here, so a generated roadmap can't arrive pre-archived
    or with progress already set."""
    now = datetime.now(timezone.utc).isoformat()

    stages, stage_id_by_order = [], {}
    for s in sorted(draft.stages, key=lambda s: s.order):
        stage = Stage(
            id=uuid.uuid4().hex[:12],
            order=s.order,
            title=s.title,
            description=s.description,
        )
        stages.append(stage)
        stage_id_by_order.setdefault(s.order, stage.id)

    topics = [
        TopicNode(
            id=uuid.uuid4().hex[:12],
            stage_id=stage_id_by_order.get(t.stage_order),
            order=t.order,
            title=t.title,
            description=t.description,
            learning_outcomes=t.learning_outcomes,
            prerequisites=t.prerequisites,
            estimated_minutes=t.estimated_minutes,
            difficulty=t.difficulty,
            resources=[Resource(**r.model_dump()) for r in t.resources],
        )
        for t in sorted(draft.topics, key=lambda t: t.order)
    ]
    # A new roadmap opens with its first topic underway, so the learner has
    # something in flight — and digests arriving — without a separate step.
    enforce_single_in_progress(topics)

    return RoadmapOutput(
        title=draft.title,
        summary=draft.summary,
        status=status,
        learner_goal=draft.learner_goal,
        target_date=draft.target_date,
        total_estimated_hours=draft.total_estimated_hours,
        stages=stages,
        topics=topics,
        personalization=personalization,
        created_at=now,
        updated_at=now,
    )


def merge_roadmap(
    existing: dict, draft: RoadmapDraft, memory: Optional[dict] = None
) -> RoadmapOutput:
    """Apply an edited draft on top of a stored roadmap without losing progress.

    A topic keeps its id — and so its progress and its PA to-do `source_ref` —
    when the model echoes back `existing_id`, or failing that when its title
    matches a stored topic. Anything else is a new topic at `not_started`.

    Only ids on THIS roadmap are honoured and each stored topic is claimed once,
    so a model repeating or inventing an id can't copy one topic's progress onto
    several or reach outside the document.
    """
    stored = {t["id"]: t for t in existing.get("topics") or [] if t.get("id")}
    by_title = {}
    for t in stored.values():
        by_title.setdefault(t.get("title", "").strip().lower(), t)

    merged = materialize_roadmap(
        draft,
        status=existing.get("status") or "active",
        # An edit re-personalizes against the profile as it stands now; without
        # a profile to hand, keep whatever the roadmap was built from.
        personalization=(
            profile_snapshot(memory)
            if memory is not None
            else existing.get("personalization")
        ),
    )

    claimed = set()
    # Same sort as materialize_roadmap, so each materialized topic lines up with
    # the draft topic it was built from.
    for topic, drafted in zip(
        merged.topics, sorted(draft.topics, key=lambda t: t.order)
    ):
        prior = stored.get(drafted.existing_id) if drafted.existing_id else None
        if prior is None:
            prior = by_title.get(drafted.title.strip().lower())
        if prior is None or prior["id"] in claimed:
            continue

        claimed.add(prior["id"])
        topic.id = prior["id"]
        topic.progress_status = prior.get("progress_status") or "not_started"
        topic.mastery_score = prior.get("mastery_score")
        topic.completed_at = prior.get("completed_at")
        topic.next_review_at = prior.get("next_review_at")
        # `review_count` rides along or the edit silently resets the spaced-
        # repetition ladder to day one; the revision debt rides along or an edit
        # becomes a way to skip revising after a failed checkpoint.
        topic.review_count = int(prior.get("review_count") or 0)
        topic.checkpoint_attempts = int(prior.get("checkpoint_attempts") or 0)
        topic.revisions_done = int(prior.get("revisions_done") or 0)
        topic.weak_points = list(prior.get("weak_points") or [])
        topic.feynman_passed = bool(prior.get("feynman_passed"))
        topic.feynman_score = prior.get("feynman_score")
        topic.feynman_at = prior.get("feynman_at")
        topic.feynman_text = prior.get("feynman_text")

    # An edit can drop the topic that was underway, or carry two across; either
    # way the roadmap comes out with exactly one.
    enforce_single_in_progress(merged.topics)
    merged.created_at = existing.get("created_at") or merged.created_at
    return merged


# ── persistence ──────────────────────────────────────────────────────────────
async def insertRoadmapToDb(
    roadmap: RoadmapOutput, user_id: Optional[str] = None
) -> Optional[str]:
    try:
        doc = roadmap.model_dump()
        doc["user_id"] = user_id
        # A new roadmap defaults to active, but not past the cap: approving one
        # while two are already running parks it rather than refusing to save
        # what the learner just built. They pick which two run from the roadmap
        # list, and nothing they asked for is lost.
        if (
            doc.get("status") == "active"
            and user_id
            and len(await active_roadmaps(user_id)) >= MAX_ACTIVE_ROADMAPS
        ):
            doc["status"] = "paused"
            logger.info("new roadmap parked at active cap user=%s", user_id)
        res = await get_db()["roadmaps"].insert_one(doc)
        logger.info("insertRoadmapToDb inserted: %s", res.inserted_id)
        return str(res.inserted_id)
    except Exception as e:
        logger.error("insertRoadmapToDb error: %s", e)
        return None


async def replace_roadmap(roadmapId: str, user_id: str, roadmap: RoadmapOutput) -> bool:
    """Overwrite a roadmap's content, keeping its owner. Scoped by `user_id`, so a
    guessed roadmapId can't be overwritten or reassigned to the caller."""
    try:
        res = await get_db()["roadmaps"].replace_one(
            {"_id": ObjectId(roadmapId), "user_id": user_id},
            {**roadmap.model_dump(), "user_id": user_id},
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error("replace_roadmap error id=%s: %s", roadmapId, e)
        return False


async def fetch_roadmap(roadmapId: Optional[str], user_id: str) -> Optional[dict]:
    if not roadmapId:
        return None
    try:
        doc = await get_db()["roadmaps"].find_one(
            {"_id": ObjectId(roadmapId), "user_id": user_id}
        )
        if doc:
            doc["_id"] = str(doc["_id"])
            return doc
    except Exception as e:
        logger.error("roadmap fetch error: %s", e)
    return None


async def resolve_roadmap_id(
    user_id: str, explicit: Optional[str] = None
) -> Optional[str]:
    """Whatever the caller passed, else the learner's most recently touched
    roadmap that's still in play — so a bare "what should I study next?" works
    without the client tracking an id. Archived and completed roadmaps are behind
    the learner and never resolve."""
    if explicit:
        return explicit
    try:
        doc = await get_db()["roadmaps"].find_one(
            {"user_id": user_id, "status": {"$nin": ["archived", "completed"]}},
            sort=[("updated_at", -1), ("_id", -1)],
            projection={"_id": 1},
        )
        return str(doc["_id"]) if doc else None
    except Exception as e:
        logger.error("resolve_roadmap_id error user=%s: %s", user_id, e)
        return None


async def list_roadmaps(
    user_id: str, status: Optional[str] = None, limit: int = 20, skip: int = 0
) -> list[dict]:
    """A learner's roadmaps, newest first, optionally filtered by status."""
    query = {"user_id": user_id}
    if status:
        query["status"] = status
    cursor = (
        get_db()["roadmaps"]
        .find(query)
        .sort([("updated_at", -1), ("_id", -1)])
        .skip(max(skip, 0))
        .limit(max(limit, 1))
    )
    docs = await cursor.to_list(None)
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return docs


# ── mastery ──────────────────────────────────────────────────────────────────
# A flat average over every attempt a learner has ever made is a weak signal: it
# treats a quiz failed in week one as evidence about today, and it can only move
# a fraction of a point once there's any history, so it stops responding exactly
# when the learner starts improving. Two decays fix that.
#
# The first weights attempts by age, so mastery reflects where they are now
# rather than where they started. The second decays a topic once its scheduled
# review is overdue: a topic aced in March and untouched since is not knowledge
# you still have, and pretending otherwise is how a dashboard talks a learner out
# of revising.
MASTERY_ATTEMPT_HALF_LIFE_DAYS = 14
MASTERY_OVERDUE_HALF_LIFE_DAYS = 21
# Below this, "improving" and "slipping" are noise in a 4-question quiz.
MASTERY_TREND_BAND = 5


def _age_days(stamp: Optional[str], now: datetime) -> Optional[float]:
    try:
        when = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max((now - when).total_seconds() / 86_400, 0.0)


def topic_mastery(
    attempts: list[dict],
    next_review_at: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """How well one topic is actually held, from its graded attempts.

    `score` is an age-weighted mean of the attempts — each one counts half as
    much per MASTERY_ATTEMPT_HALF_LIFE_DAYS, so a learner is measured on where
    they are rather than on their first stumble. `retention` is 1.0 until the
    topic's scheduled review falls due, then decays; `mastery` is the two
    combined, and is the number worth showing.

    `trend` compares the newest attempt against the weighted mean of the ones
    before it. Returns None when there's nothing graded — a topic with no
    attempts has no mastery, which is different from a mastery of zero.
    """
    now = now or datetime.now(timezone.utc)

    dated = []
    for a in attempts:
        age = _age_days(a.get("createdAt"), now)
        score = a.get("score")
        if age is not None and isinstance(score, (int, float)):
            dated.append((age, float(score)))
    if not dated:
        return None

    dated.sort(key=lambda p: p[0])  # newest first

    def weighted(points: list[tuple[float, float]]) -> float:
        total = sum(0.5 ** (age / MASTERY_ATTEMPT_HALF_LIFE_DAYS) for age, _ in points)
        if total <= 0:
            return points[0][1]
        return (
            sum(
                (0.5 ** (age / MASTERY_ATTEMPT_HALF_LIFE_DAYS)) * s
                for age, s in points
            )
            / total
        )

    score = weighted(dated)

    # Overdue decay, measured against the spaced-repetition schedule rather than
    # raw elapsed time: a topic reviewed on time hasn't faded, however long the
    # rung was. Keeps this number and `reviews_due` telling one story.
    retention = 1.0
    overdue = _age_days(next_review_at, now) if next_review_at else None
    if overdue:
        retention = 0.5 ** (overdue / MASTERY_OVERDUE_HALF_LIFE_DAYS)

    latest = dated[0][1]
    delta = latest - weighted(dated[1:]) if len(dated) > 1 else 0.0
    trend = "new"
    if len(dated) > 1:
        trend = (
            "improving"
            if delta >= MASTERY_TREND_BAND
            else "slipping" if delta <= -MASTERY_TREND_BAND else "steady"
        )

    return {
        "score": round(score),
        "retention": round(retention, 2),
        "mastery": round(score * retention),
        "trend": trend,
        "delta": round(delta),
        "attempts": len(dated),
        "overdue_days": round(overdue) if overdue else 0,
    }


def mastery_summary(topics: list[dict]) -> dict:
    """Roll per-topic mastery into the one figure a landing screen can carry.

    Unweighted across topics on purpose: every topic the learner has been graded
    on counts the same, so a topic they quietly stopped practising drags the
    number down instead of being averaged away by whatever they drilled most.
    """
    scored = [t for t in topics if t.get("mastery") is not None]
    if not scored:
        return {
            "score": None,
            "trend": "new",
            "delta": 0,
            "topics_scored": 0,
            "weakest": [],
        }

    overall = sum(t["mastery"] for t in scored) / len(scored)
    delta = sum(t["delta"] for t in scored) / len(scored)
    movers = [t for t in scored if t["trend"] != "new"]
    trend = "new"
    if movers:
        trend = (
            "improving"
            if delta >= MASTERY_TREND_BAND
            else "slipping" if delta <= -MASTERY_TREND_BAND else "steady"
        )

    return {
        "score": round(overall),
        "trend": trend,
        "delta": round(delta),
        "topics_scored": len(scored),
        # What to actually do about it. Sorted by mastery so the weakest lead,
        # and capped — a list of everything is a list nobody acts on.
        "weakest": sorted(scored, key=lambda t: t["mastery"])[:3],
    }


def _local_day(stamp: Optional[str], zone) -> Optional[str]:
    """The calendar day an ISO timestamp falls on, in `zone`. None if unusable.

    Which day something happened on is a question about where the learner is, not
    where the server is — see the note in `learning_stats`. Timestamps written
    before these were stored tz-aware are read as UTC, which is what they were.
    """
    if not stamp:
        return None
    try:
        at = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at.astimezone(zone).date().isoformat()


def consecutive_days(days: set[str], today: date) -> int:
    """How many days in a row, ending today, appear in `days`.

    A quiet *today* doesn't break the run — it starts counting from yesterday
    instead. Otherwise every streak would read 0 each morning until the learner
    got round to that day's digest, which is the reading most likely to convince
    someone they'd lost a run they still have.
    """
    day = today if today.isoformat() in days else today - timedelta(days=1)
    streak = 0
    while day.isoformat() in days:
        streak += 1
        day -= timedelta(days=1)
    return streak


async def learning_stats(user_id: str) -> dict:
    """Progress aggregated across ALL of a learner's roadmaps, for the landing
    screen.

    Deliberately not folded into `list_roadmaps`: that endpoint is paginated, and
    a global aggregate returned alongside one page of results reads as though it
    describes the page. This also spans `quiz_attempts`, which roadmaps don't
    touch. Two parallel requests beat one misleading response.
    """
    # Every "which day did this happen on" below is answered in the learner's own
    # timezone rather than the server's — see the streak note further down.
    zone = await user_zone(user_id, "learning_digest")

    roadmaps = []
    try:
        cursor = get_db()["roadmaps"].find(
            {"user_id": user_id},
            projection={
                "title": 1,
                "status": 1,
                "topics.id": 1,
                "topics.title": 1,
                "topics.progress_status": 1,
                "topics.completed_at": 1,
                "topics.next_review_at": 1,
            },
        )
        roadmaps = await cursor.to_list(None)
    except Exception as e:
        logger.error("learning_stats roadmap error user=%s: %s", user_id, e)

    # `max_active` rides along so the roadmap list can show "1 of 2 running" and
    # disable resume at the cap, instead of finding out via a 409.
    counts = {
        "total": len(roadmaps),
        "active": 0,
        "completed": 0,
        # Parked, so it's separable from archived: a paused roadmap is a
        # candidate for a free slot, an archived one is behind the learner.
        "paused": 0,
        "max_active": MAX_ACTIVE_ROADMAPS,
    }
    total_topics = 0
    reviews_due = 0
    completed_days: list[Optional[str]] = []  # local ISO date per completed topic
    stamp = datetime.now(timezone.utc).isoformat()
    for r in roadmaps:
        if r.get("status") == "active":
            counts["active"] += 1
        elif r.get("status") == "completed":
            counts["completed"] += 1
        elif r.get("status") == "paused":
            counts["paused"] += 1
        for t in r.get("topics") or []:
            total_topics += 1
            if t.get("progress_status") == "completed":
                # None for a topic completed without a timestamp: it still counts
                # towards the total, it just can't be filed under a week.
                completed_days.append(_local_day(t.get("completed_at"), zone))
                due = t.get("next_review_at")
                if due and due <= stamp:
                    reviews_due += 1

    completed_topics = len(completed_days)

    # Days the learner showed up, which in this product means acknowledging a
    # digest. It used to mean completing a topic, and that made the counter
    # unreachable rather than merely wrong: a topic takes DIGEST_MIN_BEFORE_
    # CHECKPOINT instalments plus a passed checkpoint, the drip-feed sends one a
    # day, and at most MAX_ACTIVE_ROADMAPS run at once — so "completed a topic on
    # each of two consecutive days" is something a normal learner essentially
    # never does, and the tile read 🔥1 at best and — for everyone reading
    # daily and finishing nothing that week — 0.
    #
    # Marking a digest is the daily act the whole feature is built around: it is
    # gated behind the recall check, so it can't be tapped past, and it happens
    # once a day by design. Digests closed by a passed checkpoint count too —
    # they carry the same `marked` status, and passing a checkpoint is the
    # strongest evidence of showing up there is.
    zone = await user_zone(user_id, "learning_digest")
    marked_days: set[str] = set()
    try:
        cursor = get_db()[DIGESTS].find(
            {"user_id": user_id, "status": "marked"}, projection={"updatedAt": 1}
        )
        for d in await cursor.to_list(None):
            # `updatedAt` is when it was acknowledged: marking is the only thing
            # that ever updates a digest (see mark_digest).
            local = _local_day(d.get("updatedAt"), zone)
            if local:
                marked_days.add(local)
    except Exception as e:
        # The streak is one tile. Losing it must not cost the caller the roadmap
        # counts and mastery that share this response.
        logger.error("learning_stats digest error user=%s: %s", user_id, e)

    # In the learner's own timezone, not the server's. The digest arrives at an
    # hour they chose in that zone, so their day has to end where they think it
    # does — on UTC boundaries a learner in Asia/Kolkata reading after midnight
    # has it credited to yesterday, which breaks the run they were building.
    today = datetime.now(zone).date()
    streak = consecutive_days(marked_days, today)

    week_start = (today - timedelta(days=6)).isoformat()
    this_week = sum(1 for d in completed_days if d and d >= week_start)

    attempts, average = 0, 0
    rows: list[dict] = []
    try:
        rows = await (
            get_db()["quiz_attempts"]
            .find(
                {"user_id": user_id},
                projection={
                    "score": 1,
                    "createdAt": 1,
                    "roadmapId": 1,
                    "topicId": 1,
                },
            )
            .to_list(None)
        )
        scores = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
        attempts = len(rows)
        average = round(sum(scores) / len(scores)) if scores else 0
    except Exception as e:
        logger.error("learning_stats quiz error user=%s: %s", user_id, e)

    # Per-topic mastery, from the same rows the average came from.
    by_topic: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r.get("roadmapId"), r.get("topicId"))
        if all(key):
            by_topic.setdefault(key, []).append(r)

    scored_topics = []
    for r in roadmaps:
        # Mastery is an extra on top of the counts; a roadmap we can't key
        # attempts against loses its mastery, not the whole stats response.
        if not r.get("_id"):
            continue
        rid = str(r["_id"])
        for t in r.get("topics") or []:
            graded = by_topic.get((rid, t.get("id")))
            if not graded:
                continue
            m = topic_mastery(graded, t.get("next_review_at"))
            if m:
                scored_topics.append(
                    {
                        **m,
                        "roadmapId": rid,
                        "roadmapTitle": r.get("title"),
                        "topicId": t.get("id"),
                        "title": t.get("title"),
                    }
                )

    return {
        "roadmaps": counts,
        "topics": {
            "total": total_topics,
            "completed": completed_topics,
            "percent": (
                round(completed_topics / total_topics * 100) if total_topics else 0
            ),
        },
        "completed_this_week": this_week,
        "streak_days": streak,
        "reviews_due": reviews_due,
        # The lifetime average is kept because shipped clients read it, but it's
        # the weak signal `mastery` exists to replace — see the note above
        # MASTERY_ATTEMPT_HALF_LIFE_DAYS.
        "quizzes": {"attempts": attempts, "average_score": average},
        "mastery": mastery_summary(scored_topics),
    }


async def delete_roadmap(roadmapId: str, user_id: str) -> Optional[dict]:
    """Delete a roadmap and everything the learning tracker stored against it.

    Returns per-collection counts of what went, or None when there was no such
    roadmap for this caller. Distinct from archiving, which keeps the record and
    the history — this is for a roadmap the learner wants gone.

    Everything keyed by `roadmapId` goes with it. Leaving the children behind
    would put digests for a roadmap that no longer exists into the catch-up
    queue, notes under a heading that can't be resolved, and graded attempts into
    the mastery average for a topic the learner can no longer open.

    The roadmap document is deleted FIRST, and the cascade only runs once that
    lands. It is also what enforces ownership: scoped by `user_id`, so a guessed
    id matches nothing and nothing downstream is touched. If a later step fails
    the leftovers are orphans — invisible and re-cleanable — where the reverse
    order would strip the history off a roadmap that still exists.
    """
    try:
        res = await get_db()["roadmaps"].delete_one(
            {"_id": ObjectId(roadmapId), "user_id": user_id}
        )
    except Exception as e:
        logger.error("delete_roadmap error id=%s: %s", roadmapId, e)
        return None
    if res.deleted_count == 0:
        return None

    removed = {"roadmap": res.deleted_count}
    for label, collection in (
        ("digests", DIGESTS),
        ("notes", NOTES),
        ("quizzes", "quizzes"),
        ("attempts", "quiz_attempts"),
        ("misconceptions", MISCONCEPTIONS),
    ):
        try:
            out = await get_db()[collection].delete_many(
                {"user_id": user_id, "roadmapId": roadmapId}
            )
            removed[label] = out.deleted_count
        except Exception as e:
            # The roadmap is already gone; a failed sweep must not report the
            # delete as failed and invite the learner to try again.
            logger.error(
                "delete_roadmap cascade error id=%s collection=%s: %s",
                roadmapId,
                collection,
                e,
            )
            removed[label] = 0

    # Not deleted: to-dos the learner may have rescheduled or annotated live in
    # the personal assistant, and clearing someone's task list as a side effect
    # of deleting a roadmap is a bigger decision than this endpoint should make.
    # Counted so the caller can say they're still there.
    try:
        removed["linked_tasks"] = await get_db()[TODOS].count_documents(
            {
                "user_id": user_id,
                # The string `roadmap_task_specs` stamps on every task it creates.
                "source": "learning_tracker",
                "source_ref": {"$regex": f"^{re.escape(roadmapId)}:"},
            }
        )
    except Exception as e:
        logger.error("delete_roadmap task count error id=%s: %s", roadmapId, e)
        removed["linked_tasks"] = 0

    logger.info("roadmap %s deleted user=%s removed=%s", roadmapId, user_id, removed)
    return removed


class ActiveRoadmapLimit(Exception):
    """Raised when activating a roadmap would exceed MAX_ACTIVE_ROADMAPS.

    A distinct type rather than a False return because the caller has to tell
    "no such roadmap" (404) from "park one first" (409) — they need different
    words in the UI.
    """

    def __init__(self, limit: int, active: list[dict]):
        self.limit = limit
        # What's holding the slots, so the client can name them in the prompt
        # instead of making the learner go and look.
        self.active = active
        super().__init__(f"At most {limit} roadmaps can be active at once.")


async def active_roadmaps(
    user_id: str, exclude: Optional[str] = None
) -> list[dict]:
    """The learner's active roadmaps as `{_id, title}`, oldest slot first.

    Used to decide whether another one can be activated, so it's deliberately a
    list and not a count: refusing is only useful if we can say what to park.
    """
    try:
        cursor = get_db()["roadmaps"].find(
            {"user_id": user_id, "status": "active"},
            projection={"title": 1},
        )
        docs = await cursor.to_list(None)
    except Exception as e:
        logger.error("active_roadmaps error user=%s: %s", user_id, e)
        return []
    return [
        {"_id": str(d["_id"]), "title": d.get("title")}
        for d in docs
        if str(d["_id"]) != exclude
    ]


async def set_roadmap_status(roadmapId: str, user_id: str, status: str) -> bool:
    """Park, resume, or archive a roadmap.

    Raises ActiveRoadmapLimit when resuming would put the learner over
    MAX_ACTIVE_ROADMAPS. Enforced here rather than in the route so every caller
    is covered by the same check — `active` is what the digest sweep runs on, so
    a bypass doesn't just break a rule, it starts another drip-feed.
    """
    if status == "active":
        # Excluding this roadmap makes activating an already-active one a no-op
        # rather than a spurious refusal.
        holders = await active_roadmaps(user_id, exclude=roadmapId)
        if len(holders) >= MAX_ACTIVE_ROADMAPS:
            raise ActiveRoadmapLimit(MAX_ACTIVE_ROADMAPS, holders)

    try:
        res = await get_db()["roadmaps"].update_one(
            {"_id": ObjectId(roadmapId), "user_id": user_id},
            {
                "$set": {
                    "status": status,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error("set_roadmap_status error id=%s: %s", roadmapId, e)
        return False


async def set_topic_progress(
    roadmapId: str,
    topicId: str,
    status: str,
    user_id: str,
    mastery_score: Optional[int] = None,
) -> bool:
    """Move one topic to `status` via a targeted positional update — no LLM, no
    full-document rewrite. Rolls the roadmap itself to `completed` once every
    topic is done (and back to `active` if one is reopened)."""
    # Starting a topic is a swap, not a set: whichever topic held the slot has
    # to give it up in the same write.
    if status == "in_progress":
        started = await start_topic(roadmapId, topicId, user_id)
        if started:
            await _rollup_roadmap_status(roadmapId, user_id)
        return started

    now = datetime.now(timezone.utc).isoformat()
    done = status == "completed"
    updates = {
        "topics.$.progress_status": status,
        "topics.$.completed_at": now if done else None,
        "updated_at": now,
    }
    if mastery_score is not None:
        updates["topics.$.mastery_score"] = mastery_score

    col = get_db()["roadmaps"]
    try:
        res = await col.update_one(
            {"_id": ObjectId(roadmapId), "user_id": user_id, "topics.id": topicId},
            {"$set": updates},
        )
        if res.matched_count == 0:
            return False
    except Exception as e:
        logger.error("set_topic_progress error: %s", e)
        return False

    await _rollup_roadmap_status(roadmapId, user_id)
    return True


async def _rollup_roadmap_status(roadmapId: str, user_id: str) -> None:
    """Move the roadmap itself to `completed` once every topic is done, and back
    to `active` if one is reopened. Best-effort — the topic write that triggered
    this has already succeeded and must not be reported as failed if the rollup
    doesn't land."""
    col = get_db()["roadmaps"]
    try:
        doc = await col.find_one(
            {"_id": ObjectId(roadmapId), "user_id": user_id},
            projection={"topics": 1, "status": 1},
        )
        topics = (doc or {}).get("topics") or []
        rolled = (
            "completed"
            if all(t.get("progress_status") in DONE_STATUSES for t in topics)
            else "active"
        )
        # Reopening a topic on a finished roadmap would otherwise take the
        # learner over the active cap without them ever asking for it. It gets
        # parked instead — still reopened, just not drip-feeding until they free
        # a slot themselves.
        if rolled == "active" and doc.get("status") == "completed":
            if len(await active_roadmaps(user_id, exclude=roadmapId)) >= MAX_ACTIVE_ROADMAPS:
                rolled = "paused"

        # Only ever move between active and completed: never un-archive or
        # un-pause a roadmap the learner parked by hand.
        if (
            topics
            and doc.get("status") in ("active", "completed")
            and doc["status"] != rolled
        ):
            await col.update_one(
                {"_id": ObjectId(roadmapId), "user_id": user_id},
                {
                    "$set": {
                        "status": rolled,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                },
            )
    except Exception as e:
        logger.error("roadmap status rollup error id=%s: %s", roadmapId, e)


# ── checkpoints & spaced repetition ──────────────────────────────────────────
# Days until a topic resurfaces, indexed by how many checkpoints it has passed
# in a row. Expanding intervals: the better you know it, the less often it comes
# back. A failed check resets to the front of the ladder.
REVIEW_LADDER_DAYS = (1, 3, 7, 16, 35)


def next_review_at(review_count: int) -> str:
    """When a topic with `review_count` consecutive passes should resurface."""
    idx = min(max(review_count - 1, 0), len(REVIEW_LADDER_DAYS) - 1)
    due = datetime.now(timezone.utc) + timedelta(days=REVIEW_LADDER_DAYS[idx])
    return due.isoformat()


def find_topic(roadmap: Optional[dict], topicId: str) -> Optional[dict]:
    return next(
        (t for t in (roadmap or {}).get("topics") or [] if t.get("id") == topicId), None
    )


async def apply_checkpoint(
    roadmapId: str,
    topicId: str,
    user_id: str,
    score: int,
    missed: Optional[list[str]] = None,
) -> Optional[dict]:
    """Fold a graded checkpoint into the topic: mastery, completion, next review.

    Passing is what completes a topic — the point of gating completion on active
    recall rather than on a checkbox.

    Failing is deliberately asymmetric:

    * A failed *first* attempt holds the topic at `needs_review` and opens a
      revision debt (see `revision_outstanding`). It does NOT drop back to
      `in_progress`: coverage has already declared the topic fully taught, so
      re-opening the teaching drip-feed would generate one more arbitrary digest
      and then bounce straight back here. What's owed is revision of the specific
      things that were missed, which is a different artefact.
    * A failed *review* does NOT un-complete a topic the learner already
      finished; it drags the next review back to the front of the ladder instead.
      Clawing back progress for an honest attempt would punish the exact
      behaviour this feature exists to encourage.

    `missed` is the text of the questions got wrong, and becomes what the
    revision digest is written against.
    """
    roadmap = await fetch_roadmap(roadmapId, user_id)
    topic = find_topic(roadmap, topicId)
    if topic is None:
        return None

    passed = score >= CHECKPOINT_PASS_SCORE
    was_completed = topic.get("progress_status") == "completed"
    review_count = (int(topic.get("review_count") or 0) + 1) if passed else 0

    # The reward for explaining the topic in your own words, and the only thing
    # that makes an optional exercise worth doing: a passed Feynman explanation
    # starts the topic further up the review ladder, so someone who can say it
    # plainly sees it again later than someone who only recognised the right
    # option. Spent on use, not banked — the next review has to earn its own.
    feynman_bonus = 0
    if passed and topic.get("feynman_passed"):
        feynman_bonus = FEYNMAN_LADDER_BONUS
        review_count += feynman_bonus

    status = "completed" if (passed or was_completed) else "needs_review"

    now = datetime.now(timezone.utc).isoformat()
    updates = {
        "topics.$.progress_status": status,
        "topics.$.mastery_score": score,
        "topics.$.review_count": review_count,
        "updated_at": now,
    }
    if status == "completed":
        # Only a completed topic has a next review to schedule. Stamping one on a
        # topic that has never been passed puts a date on the resurface ladder
        # for something that hasn't been learned yet.
        updates["topics.$.next_review_at"] = next_review_at(review_count)
        if not topic.get("completed_at"):
            updates["topics.$.completed_at"] = now
        if feynman_bonus:
            # Cleared as it's spent, so the bonus applies to the review it was
            # earned for rather than to every review from here on.
            updates["topics.$.feynman_passed"] = False
    else:
        # A failed first attempt owes one round of revision before the next try.
        updates["topics.$.checkpoint_attempts"] = (
            int(topic.get("checkpoint_attempts") or 0) + 1
        )
        updates["topics.$.weak_points"] = missed or []

    try:
        res = await get_db()["roadmaps"].update_one(
            {"_id": ObjectId(roadmapId), "user_id": user_id, "topics.id": topicId},
            {"$set": updates},
        )
        if res.matched_count == 0:
            return None
    except Exception as e:
        logger.error("apply_checkpoint error: %s", e)
        return None

    # Passing the checkpoint on a topic for the first time hands the slot to the
    # next one, so the learner always has exactly one topic in flight and its
    # digests start straight away.
    advanced = None
    closed = 0
    if passed and not was_completed:
        # The teaching for this topic is over, so nothing it sent is still
        # waiting on an answer. Only on the transition: a review of a topic that
        # was already complete has no drip-feed left to close.
        closed = await close_topic_digests(roadmapId, topicId, user_id)
        ordered = sorted(roadmap.get("topics") or [], key=lambda t: t.get("order", 0))
        nxt = next(
            (
                t
                for t in ordered
                if t.get("id") != topicId
                and t.get("progress_status") not in DONE_STATUSES
            ),
            None,
        )
        if nxt and await start_topic(roadmapId, nxt["id"], user_id):
            advanced = {"topicId": nxt["id"], "title": nxt.get("title")}

    await _rollup_roadmap_status(roadmapId, user_id)

    return {
        "passed": passed,
        "progress_status": status,
        "review_count": review_count,
        "next_review_at": updates.get("topics.$.next_review_at"),
        "was_review": was_completed,
        # The topic that just picked up the baton, if any.
        "advanced_to": advanced,
        # Digests that were still waiting on this topic and have been closed with
        # it, so the client can drop them from the catch-up queue rather than
        # leaving them there until something happens to refetch.
        "digests_closed": closed,
        # Set when the failure opened a revision debt, so the client can send the
        # learner to revise rather than straight back to the same questions.
        "needs_revision": not passed and not was_completed,
        "weak_points": [] if passed else (missed or []),
        # Rungs granted for having explained the topic in your own words, so the
        # client can show the reward landing rather than an unexplained date.
        "feynman_bonus": feynman_bonus,
    }


async def close_topic_digests(roadmapId: str, topicId: str, user_id: str) -> int:
    """Close whatever is still waiting on a topic the learner has just finished.

    A digest is a nudge to read up on the topic you're working on. Once the
    checkpoint is passed that nudge has nothing left to ask for — the learner
    proved they know the material by a harder route than the one the tips were
    softening. Left open they sit in the catch-up queue nagging about finished
    work, and marking one late is worse than useless: its recall check is graded
    against a *completed* topic, denting a mastery score that has since been
    earned, and failing it twice sends a re-teach digest for a topic nobody is
    studying any more.

    `closed_by` keeps the record honest. `status: "marked"` normally means the
    learner acknowledged it, and that distinction matters — it is what
    `digest_quiz_gate` reads as "the check was passed". These were closed by the
    checkpoint, not read, and the archive can say so.

    Returns how many were closed. Best-effort: the checkpoint has already been
    applied, and a tidy-up that fails must not report the pass as failed.
    """
    try:
        res = await get_db()[DIGESTS].update_many(
            {
                "user_id": user_id,
                "roadmapId": roadmapId,
                "topicId": topicId,
                "status": {"$ne": "marked"},
            },
            {
                "$set": {
                    "status": "marked",
                    "closed_by": "checkpoint",
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        return res.modified_count
    except Exception as e:
        logger.error(
            "close_topic_digests error id=%s topic=%s: %s", roadmapId, topicId, e
        )
        return 0


async def record_explanation(
    roadmapId: str,
    topicId: str,
    user_id: str,
    judgement: dict,
    explanation: str,
    passed: bool,
) -> bool:
    """Store a Feynman attempt on the topic.

    `feynman_passed` is the ladder credit the next checkpoint spends. The
    explanation itself is kept because it's the only record of how this learner
    actually talks about the topic — worth more on a later read than the score.
    """
    try:
        res = await get_db()["roadmaps"].update_one(
            {"_id": ObjectId(roadmapId), "user_id": user_id, "topics.id": topicId},
            {
                "$set": {
                    # Never cleared by a poor attempt: the exercise is optional and
                    # trying again must not cost the learner credit they've earned.
                    **({"topics.$.feynman_passed": True} if passed else {}),
                    "topics.$.feynman_score": judgement.get("score"),
                    "topics.$.feynman_at": datetime.now(timezone.utc).isoformat(),
                    "topics.$.feynman_text": explanation,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error("record_explanation error id=%s topic=%s: %s", roadmapId, topicId, e)
        return False


def revision_outstanding(topic: Optional[dict]) -> bool:
    """Whether a failed checkpoint still owes this topic a round of revision.

    The gate on retrying: every failed attempt has to be answered by
    acknowledging one revision digest before the next attempt is issued. Without
    it, "retry" is just re-rolling the same questions until they stick by
    accident, which is the opposite of what a recall check is for.
    """
    t = topic or {}
    return int(t.get("checkpoint_attempts") or 0) > int(t.get("revisions_done") or 0)


async def record_revision(roadmapId: str, topicId: str, user_id: str) -> bool:
    """Credit one round of revision — called when a revision digest is marked.

    Never counts past what's owed: acknowledging a second revision digest for a
    single failure must not bank credit against a failure that hasn't happened,
    which would let the learner skip the revision after their next one.
    """
    topic = find_topic(await fetch_roadmap(roadmapId, user_id), topicId)
    if not revision_outstanding(topic):
        return False
    try:
        res = await get_db()["roadmaps"].update_one(
            {"_id": ObjectId(roadmapId), "user_id": user_id, "topics.id": topicId},
            {
                "$inc": {"topics.$.revisions_done": 1},
                "$set": {"updated_at": datetime.now(timezone.utc).isoformat()},
            },
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error("record_revision error id=%s topic=%s: %s", roadmapId, topicId, e)
        return False


async def due_reviews(user_id: str, limit: int = 20) -> list[dict]:
    """Completed topics whose next review has come due, soonest first — what
    keeps earlier material coming back instead of being ticked off once."""
    now = datetime.now(timezone.utc).isoformat()
    try:
        cursor = get_db()["roadmaps"].find(
            {"user_id": user_id, "topics.next_review_at": {"$lte": now}},
            projection={"title": 1, "topics": 1},
        )
        roadmaps = await cursor.to_list(None)
    except Exception as e:
        logger.error("due_reviews error user=%s: %s", user_id, e)
        return []

    due = []
    for r in roadmaps:
        for t in r.get("topics") or []:
            when = t.get("next_review_at")
            # The document matched because *a* topic is due; re-check each one.
            if not when or when > now or t.get("progress_status") != "completed":
                continue
            due.append(
                {
                    "roadmapId": str(r["_id"]),
                    "roadmapTitle": r.get("title"),
                    "topicId": t.get("id"),
                    "title": t.get("title"),
                    "due_at": when,
                    "mastery_score": t.get("mastery_score"),
                    "review_count": t.get("review_count") or 0,
                }
            )

    due.sort(key=lambda d: d["due_at"])
    return due[:limit]


async def set_topic_resources(
    roadmapId: str, topicId: str, resources: list[Resource], user_id: str
) -> bool:
    """Attach curated resources to one topic. A targeted `$set` rather than a
    round-trip through TopicNode, so a legacy-shaped topic is updated in place
    instead of failing validation on its old `resources: list[str]`."""
    try:
        res = await get_db()["roadmaps"].update_one(
            {"_id": ObjectId(roadmapId), "user_id": user_id, "topics.id": topicId},
            {
                "$set": {
                    "topics.$.resources": [r.model_dump() for r in resources],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error("set_topic_resources error: %s", e)
        return False


# ── quizzes ──────────────────────────────────────────────────────────────────
def public_question(question: dict) -> dict:
    """One question as the learner may see it: the wording, the options, and how
    it is answered.

    A whitelist, and the only shape a question should ever leave the server in.
    `answer` and `expected` are the key, and `hint`/`outcome` are the feedback a
    failure earns — none of them belong in a set the learner is about to sit.

    `kind` has to be here: an `open` question has no options, and a client that
    can't tell it apart from a multiple-choice one waits forever for a tap that
    has nowhere to land.
    """
    return {
        "question": question.get("question"),
        "options": question.get("options") or [],
        "kind": question.get("kind") or "choice",
    }


def open_questions(questions: list[dict]) -> list[tuple[int, dict]]:
    """The free-text questions in a set, with their positions."""
    return [(i, q) for i, q in enumerate(questions) if q.get("kind") == "open"]


def grade_quiz(questions: list[dict], selected: dict[int, int]) -> dict:
    """Score a submission against the stored questions. Shared by the chat path
    and POST /submit-quiz so the two can't grade the same answers differently.
    `review` carries only what the learner got wrong, unanswered included.

    Open questions are skipped entirely — they have no options to have chosen
    between, and an index-based grader would score every one of them wrong.
    They're judged by an LLM alongside this, and their totals are kept apart so
    a machine-graded sentence can't silently move a tap-graded pass mark.
    """
    correct = 0
    review = []
    for idx, q in enumerate(questions):
        if q.get("kind") == "open":
            continue
        chosen = selected.get(idx)
        answer = q.get("answer")
        if chosen is not None and chosen == answer:
            correct += 1
            continue
        options = q.get("options") or []
        review.append(
            {
                "question": idx,
                "selected": chosen,
                "correctAnswer": answer,
                "correctOption": (
                    options[answer]
                    if isinstance(answer, int) and 0 <= answer < len(options)
                    else None
                ),
            }
        )

    # Over the closed questions only. Counting an open one in the denominator
    # would score a flawless tap round at 2/3 — and against an all-or-nothing
    # pass mark that makes the digest unmarkable, which jams the whole topic.
    scored = sum(1 for q in questions if q.get("kind") != "open")
    return {
        "total": scored,
        "correct": correct,
        "score": round(correct / scored * 100) if scored else 0,
        "review": review,
    }


async def fetch_quiz(user_id: str, quizId: Optional[str] = None) -> Optional[dict]:
    """A quiz owned by `user_id`: the one named, else their most recent. Scoped so
    a guessed quizId never hands back another learner's answer key."""
    col = get_db()["quizzes"]
    try:
        if quizId:
            return await col.find_one({"_id": ObjectId(quizId), "user_id": user_id})
        return await col.find_one({"user_id": user_id}, sort=[("_id", -1)])
    except Exception as e:
        logger.error("fetch_quiz error user=%s: %s", user_id, e)
        return None


def attempt_misses(questions: list[dict], result: dict) -> list[dict]:
    """What a learner got wrong, in words rather than indices.

    `grade_quiz` reports misses positionally, which is all the caller needs to
    render a result screen and useless the moment the quiz itself is out of
    scope. Resolving them against the questions here is what makes an attempt
    readable months later, and is the whole basis of misconception analysis.
    """
    misses = []
    for r in result.get("review") or []:
        idx = r.get("question")
        if not isinstance(idx, int) or not 0 <= idx < len(questions):
            continue
        options = questions[idx].get("options") or []
        chosen = r.get("selected")
        misses.append(
            {
                "question": questions[idx].get("question", ""),
                # None when they skipped it — a blank is different evidence from
                # a wrong pick, and flattening the two would invent a belief the
                # learner never expressed.
                "chosen": (
                    options[chosen]
                    if isinstance(chosen, int) and 0 <= chosen < len(options)
                    else None
                ),
                "correct": r.get("correctOption"),
            }
        )
    return misses


async def record_quiz_attempt(
    user_id: str,
    quizId: Optional[str],
    roadmapId: Optional[str],
    topicId: Optional[str],
    result: dict,
    questions: Optional[list[dict]] = None,
    kind: Optional[str] = None,
    misses: Optional[list[dict]] = None,
) -> None:
    """Persist one graded attempt, including what was got wrong and why it was
    being asked.

    `misses` and `kind` are what turn this from a score log into evidence: a
    stream of "62%" says a learner is struggling, but not at what — and "wrong on
    a digest recall check" means something different from "wrong on a
    spaced-repetition review of a topic they completed months ago".

    Pass `misses` directly when there are no options to have chosen between — a
    free-text answer or a whole explanation. Same shape either way, so one
    analysis reads all of it.
    """
    try:
        await get_db()["quiz_attempts"].insert_one(
            {
                "user_id": user_id,
                "quizId": quizId,
                "roadmapId": roadmapId,
                "topicId": topicId,
                # digest | checkpoint | review | chat
                "kind": kind,
                "total": result.get("total"),
                "correct": result.get("correct"),
                "score": result.get("score"),
                "misses": (
                    misses
                    if misses is not None
                    else attempt_misses(questions or [], result)
                ),
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as e:
        logger.error("record_quiz_attempt error user=%s: %s", user_id, e)


# ── misconceptions ───────────────────────────────────────────────────────────
# What a learner keeps getting wrong, inferred from their attempt history and
# cached per topic. Every quiz kind feeds one pool: a digest recall check, a
# completion checkpoint and a spaced-repetition review are three vantage points
# on the same understanding, and a mistake visible from more than one of them is
# exactly the durable kind worth re-probing.
MISCONCEPTIONS = "learning_misconceptions"


async def topic_misses(
    user_id: str,
    roadmapId: str,
    topicId: str,
    limit: int = MISCONCEPTION_EVIDENCE_WINDOW,
) -> list[dict]:
    """Every wrong answer recorded for one topic, oldest first, tagged with the
    kind of quiz it came from. The evidence the analysis reads."""
    try:
        cursor = (
            get_db()["quiz_attempts"]
            .find(
                {
                    "user_id": user_id,
                    "roadmapId": roadmapId,
                    "topicId": topicId,
                    "misses": {"$nin": [None, []]},
                }
            )
            .sort([("createdAt", 1), ("_id", 1)])
        )
        attempts = await cursor.to_list(None)
    except Exception as e:
        logger.error("topic_misses error user=%s: %s", user_id, e)
        return []

    misses = [
        {**m, "kind": a.get("kind"), "at": a.get("createdAt")}
        for a in attempts
        for m in a.get("misses") or []
    ]
    # Newest evidence wins once there's more than the analysis should chew on: a
    # belief the learner has since corrected shouldn't keep steering questions.
    return misses[-limit:]


async def get_misconceptions(
    user_id: str, roadmapId: str, topicId: str
) -> Optional[dict]:
    try:
        return await get_db()[MISCONCEPTIONS].find_one(
            {"user_id": user_id, "roadmapId": roadmapId, "topicId": topicId},
            projection={"_id": 0},
        )
    except Exception as e:
        logger.error("get_misconceptions error user=%s: %s", user_id, e)
        return None


async def save_misconceptions(
    user_id: str,
    roadmapId: str,
    topicId: str,
    patterns: list[dict],
    misses_analyzed: int,
) -> None:
    """Upsert the analysis, stamped with how much evidence it was drawn from.

    `misses_analyzed` is the watermark that keeps the analysis from re-running on
    unchanged history — the whole point of caching an LLM call.

    Re-analysis rewrites the patterns wholesale, so the "when was this last
    addressed" stamps are carried across by label. Losing them would reset every
    decay clock on each new piece of evidence, and a learner who keeps making
    mistakes would be the one whose misconceptions got hammered hardest.
    """
    try:
        previous = await get_misconceptions(user_id, roadmapId, topicId) or {}
        stamps = {
            p.get("label"): p
            for p in previous.get("patterns") or []
            if p.get("label")
        }
        carried = []
        for p in patterns:
            was = stamps.get(p.get("label")) or {}
            carried.append(
                {
                    **p,
                    **{
                        k: was[k]
                        for k in ("last_probed_at", "last_taught_at")
                        if was.get(k)
                    },
                }
            )

        await get_db()[MISCONCEPTIONS].update_one(
            {"user_id": user_id, "roadmapId": roadmapId, "topicId": topicId},
            {
                "$set": {
                    "patterns": carried,
                    "misses_analyzed": misses_analyzed,
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                }
            },
            upsert=True,
        )
    except Exception as e:
        logger.error("save_misconceptions error user=%s: %s", user_id, e)


def due_misconceptions(
    patterns: Optional[list[dict]],
    field: str,
    quiet_days: int,
    now: Optional[datetime] = None,
) -> list[dict]:
    """The patterns worth acting on right now.

    A misconception nobody has addressed yet is always due. One that was
    addressed recently is not: re-teaching the same correction in consecutive
    digests is nagging, and re-probing the same belief in consecutive checkpoints
    measures whether they remember last week's question rather than whether they
    now understand.

    Once the quiet period is up it becomes due again — which is the point. A
    single right answer is not a corrected belief, and the only way to tell the
    difference is to come back to it later.
    """
    now = now or datetime.now(timezone.utc)
    due = []
    for p in patterns or []:
        age = _age_days(p.get(field), now)
        if age is None or age >= quiet_days:
            due.append(p)
    return due


async def mark_misconceptions_addressed(
    user_id: str, roadmapId: str, topicId: str, labels: list[str], field: str
) -> None:
    """Start the quiet clock on the patterns something just aimed at.

    Best-effort: the digest or checkpoint has already been generated, and losing
    the stamp costs a little repetition, not correctness.
    """
    if not labels:
        return
    try:
        stored = await get_misconceptions(user_id, roadmapId, topicId)
        if not stored:
            return
        now = datetime.now(timezone.utc).isoformat()
        patterns = [
            {**p, field: now} if p.get("label") in set(labels) else p
            for p in stored.get("patterns") or []
        ]
        await get_db()[MISCONCEPTIONS].update_one(
            {"user_id": user_id, "roadmapId": roadmapId, "topicId": topicId},
            {"$set": {"patterns": patterns}},
        )
    except Exception as e:
        logger.error(
            "mark_misconceptions_addressed error user=%s topic=%s: %s",
            user_id,
            topicId,
            e,
        )


async def list_misconceptions(
    user_id: str, roadmapId: Optional[str] = None
) -> list[dict]:
    """Everything inferred about a learner, for the insight screen. Titles are
    resolved here so the client doesn't need a roadmap in hand to read it."""
    query: dict = {"user_id": user_id}
    if roadmapId:
        query["roadmapId"] = roadmapId
    try:
        cursor = get_db()[MISCONCEPTIONS].find(query, projection={"_id": 0})
        docs = await cursor.to_list(None)
    except Exception as e:
        logger.error("list_misconceptions error user=%s: %s", user_id, e)
        return []

    labels = await _label_map(user_id)
    out = []
    for d in docs:
        if not d.get("patterns"):
            continue
        meta = labels.get(d.get("roadmapId")) or {}
        d["roadmapTitle"] = meta.get("title")
        d["topicTitle"] = (meta.get("topics") or {}).get(d.get("topicId"))
        out.append(d)
    return out


# ── checkpoint attempt policy ────────────────────────────────────────────────
def redact_review(result: dict, reveal: bool, questions: list[dict]) -> dict:
    """The graded result as the learner may see it.

    Answers are revealed only on a pass. On a failure the learner gets the hint
    and the outcome each missed question was testing — enough to know what to go
    back over, not enough to fill in the right box next time. Handing back
    `correctOption` on a failure turns the retry into transcription, which is
    exactly what a completion gate must not permit.

    Grading is unchanged; this only filters what leaves the server.
    """
    if reveal:
        return result

    redacted = []
    for r in result.get("review") or []:
        idx = r.get("question")
        q = questions[idx] if isinstance(idx, int) and 0 <= idx < len(questions) else {}
        redacted.append(
            {
                "question": idx,
                "selected": r.get("selected"),
                # Deliberately absent rather than nulled: a client that renders
                # `correctOption ?? '—'` shows a dash either way, and a null here
                # would read as "we don't know" instead of "not yet".
                "outcome": q.get("outcome"),
                "hint": q.get("hint"),
            }
        )
    return {**result, "review": redacted, "answers_revealed": False}


async def failed_attempts(
    user_id: str, roadmapId: str, topicId: Optional[str], quizId: str, pass_score: int
) -> int:
    """How many times this learner has come up short on one specific check.

    Counted from the attempt log rather than a counter on the digest, so it
    survives regeneration and stays consistent with everything else the tracker
    reads.
    """
    try:
        rows = await (
            get_db()["quiz_attempts"]
            .find(
                {
                    "user_id": user_id,
                    "roadmapId": roadmapId,
                    "topicId": topicId,
                    "quizId": quizId,
                },
                projection={"score": 1},
            )
            .to_list(None)
        )
    except Exception as e:
        logger.error("failed_attempts error user=%s quiz=%s: %s", user_id, quizId, e)
        # Fail closed on the escalation, not on the learner: an unknown history
        # means no re-teach is triggered, and they simply try again.
        return 0
    return sum(1 for r in rows if (r.get("score") or 0) < pass_score)


async def replace_quiz_questions(quizId: str, user_id: str, questions: list[dict]) -> bool:
    """Swap a stored check's questions, keeping its id.

    The digest points at this quiz by id, so rewriting in place is what lets a
    re-explanation come with questions answerable from it — without orphaning the
    digest or handing the learner a second gate to clear.
    """
    try:
        res = await get_db()["quizzes"].update_one(
            {"_id": ObjectId(quizId), "user_id": user_id},
            {
                "$set": {
                    "questions": questions,
                    "regeneratedAt": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error("replace_quiz_questions error id=%s: %s", quizId, e)
        return False


async def asked_questions(
    user_id: str, roadmapId: str, topicId: str, limit: int = 30
) -> list[str]:
    """Every checkpoint question this learner has already been asked on a topic,
    newest first.

    Regenerating a review is not the same as varying it. The generator sees the
    same description and the same outcomes every time, so left to itself it
    converges on the same handful of obvious questions — and across a 1/3/7/16/35
    day ladder that rewards remembering the answer rather than the material.
    Feeding the previous questions back is what makes each review a new angle.
    """
    try:
        cursor = (
            get_db()["quizzes"]
            .find(
                {
                    "user_id": user_id,
                    "roadmapId": roadmapId,
                    "topicId": topicId,
                    "kind": {"$in": ["checkpoint", "review"]},
                },
                projection={"questions.question": 1},
            )
            .sort([("_id", -1)])
            .limit(10)
        )
        quizzes = await cursor.to_list(None)
    except Exception as e:
        logger.error("asked_questions error user=%s: %s", user_id, e)
        return []

    seen: list[str] = []
    for q in quizzes:
        for item in q.get("questions") or []:
            text = (item.get("question") or "").strip()
            if text and text not in seen:
                seen.append(text)
    return seen[:limit]


async def quiz_graded(user_id: str, quizId: str) -> bool:
    """Whether this quiz has already been through grading.

    A checkpoint set is sat exactly once. The retry limits in `retry_block` are
    enforced where a set is *issued*, so without this a learner who fails can
    re-post the same quizId with the three answers they were just told were
    wrong — no cooldown, no daily cap, and no revision in between. That is the
    completion gate opened from the outside.

    Fails closed: an unreadable history refuses the grading rather than handing
    out a free re-run.
    """
    try:
        return (
            await get_db()["quiz_attempts"].count_documents(
                {"user_id": user_id, "quizId": quizId}, limit=1
            )
            > 0
        )
    except Exception as e:
        logger.error("quiz_graded error user=%s quiz=%s: %s", user_id, quizId, e)
        return True


async def recent_attempts(
    user_id: str, roadmapId: str, topicId: str, kinds: tuple[str, ...] = ()
) -> list[dict]:
    """A topic's graded attempts, newest first."""
    query: dict = {"user_id": user_id, "roadmapId": roadmapId, "topicId": topicId}
    if kinds:
        query["kind"] = {"$in": list(kinds)}
    try:
        cursor = (
            get_db()["quiz_attempts"]
            .find(query, projection={"createdAt": 1, "score": 1, "kind": 1, "quizId": 1})
            .sort([("createdAt", -1), ("_id", -1)])
            .limit(50)
        )
        return await cursor.to_list(None)
    except Exception as e:
        logger.error("recent_attempts error user=%s: %s", user_id, e)
        # Fail closed: an unreadable history must not read as "no attempts yet"
        # and hand out an unlimited retry.
        return [{"createdAt": datetime.now(timezone.utc).isoformat(), "score": 0}] * (
            CHECKPOINT_MAX_ATTEMPTS_PER_DAY
        )


def retry_block(attempts: list[dict], now: Optional[datetime] = None) -> Optional[dict]:
    """Why a new attempt can't be issued yet, or None if it can.

    Two independent limits. The cooldown is about the seconds after a failure,
    when the instinct is to bounce straight back in; the daily cap is about the
    afternoon spent grinding one topic until a regenerated set happens to land on
    what they know. Neither applies to a learner's first go.
    """
    now = now or datetime.now(timezone.utc)

    def when(a: dict) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(a["createdAt"])
        except (KeyError, TypeError, ValueError):
            return None

    stamps = [t for t in (when(a) for a in attempts) if t]
    if not stamps:
        return None

    day_ago = now - timedelta(days=1)
    today = [t for t in stamps if t >= day_ago]
    if len(today) >= CHECKPOINT_MAX_ATTEMPTS_PER_DAY:
        return {
            "reason": "daily_limit",
            "retry_at": (min(today) + timedelta(days=1)).isoformat(),
            "attempts_today": len(today),
            "limit": CHECKPOINT_MAX_ATTEMPTS_PER_DAY,
        }

    ready = max(stamps) + timedelta(minutes=CHECKPOINT_RETRY_COOLDOWN_MINUTES)
    if ready > now:
        return {
            "reason": "cooldown",
            "retry_at": ready.isoformat(),
            "attempts_today": len(today),
            "limit": CHECKPOINT_MAX_ATTEMPTS_PER_DAY,
        }
    return None


# ── digests ──────────────────────────────────────────────────────────────────
# A digest is unread until the learner acknowledges it. That acknowledgement is
# the only evidence we have that one actually landed, so it drives both the
# catch-up screen and the decision not to pile a second digest on an unread one.
#
# `status` missing means unread: digests written before the field existed have
# never been acknowledged either, so `{"status": {"$ne": "marked"}}` is the
# honest query rather than a special case.
DIGESTS = "learning_digests"


def enforce_single_in_progress(topics: list) -> None:
    """A roadmap has exactly one topic underway at a time. Mutates in place.

    The whole drip-feed keys off that single topic — two of them would mean two
    streams of digests competing for the same attention — so the invariant is
    enforced here rather than trusted at each call site.

    A topic sitting at `needs_review` holds the slot: it has been taught in full
    and owes a checkpoint, and starting something new would let the learner
    accumulate unfinished topics instead of closing one out.
    """
    ordered = sorted(topics, key=lambda t: t.order)
    started = [t for t in ordered if t.progress_status == "in_progress"]
    for extra in started[1:]:
        extra.progress_status = "not_started"
    if started or any(t.progress_status == "needs_review" for t in ordered):
        return

    nxt = next((t for t in ordered if t.progress_status not in DONE_STATUSES), None)
    if nxt:
        nxt.progress_status = "in_progress"


async def start_topic(roadmapId: str, topicId: str, user_id: str) -> bool:
    """Make `topicId` the one topic underway, demoting whichever held the slot.

    One update with two array filters, so there is never a moment where the
    roadmap has two topics in progress or none.
    """
    try:
        res = await get_db()["roadmaps"].update_one(
            {"_id": ObjectId(roadmapId), "user_id": user_id, "topics.id": topicId},
            {
                "$set": {
                    "topics.$[other].progress_status": "not_started",
                    "topics.$[target].progress_status": "in_progress",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
            array_filters=[
                {"other.progress_status": "in_progress", "other.id": {"$ne": topicId}},
                {"target.id": topicId},
            ],
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error("start_topic error id=%s topic=%s: %s", roadmapId, topicId, e)
        return False


def in_progress_topic(roadmap: Optional[dict]) -> Optional[dict]:
    """The topic a learner has actually started, in `order`.

    Digests are only generated for these. A topic sitting at `not_started` hasn't
    been picked up yet, and drip-feeding tips about something nobody has opened
    is how an inbox fills with things nobody asked for.
    """
    topics = sorted((roadmap or {}).get("topics", []), key=lambda t: t.get("order", 0))
    return next((t for t in topics if t.get("progress_status") == "in_progress"), None)


async def unread_digest_count(
    user_id: str, roadmapId: str, topicId: Optional[str]
) -> int:
    try:
        return await get_db()[DIGESTS].count_documents(
            {
                "user_id": user_id,
                "roadmapId": roadmapId,
                "topicId": topicId,
                "status": {"$ne": "marked"},
            }
        )
    except Exception as e:
        logger.error("unread_digest_count error user=%s: %s", user_id, e)
        # Fail closed: if we can't tell how many are waiting, don't add another.
        return DIGEST_MAX_UNREAD


async def topic_digests(
    user_id: str, roadmapId: str, topicId: Optional[str]
) -> list[dict]:
    """Every digest sent for a topic, oldest first — the running record of what
    the learner has been told, which both the recall quiz and the coverage check
    are built from."""
    try:
        cursor = (
            get_db()[DIGESTS]
            .find({"user_id": user_id, "roadmapId": roadmapId, "topicId": topicId})
            .sort([("createdAt", 1), ("_id", 1)])
        )
        return await cursor.to_list(None)
    except Exception as e:
        logger.error("topic_digests error user=%s: %s", user_id, e)
        return []


# ── digest cadence ───────────────────────────────────────────────────────────
# A recall check rides every other digest — #2, #4, #6 — rather than every one
# past the first. Back-to-back checks turn a nudge into homework; the point is to
# catch a digest that was swiped away, not to examine.
def digest_carries_quiz(sequence: int) -> bool:
    """Whether the digest at `sequence` (1-based) carries a recall check."""
    return sequence % DIGEST_QUIZ_EVERY == 0


# Digests that are not part of the drip-feed. Both cover ground already sent —
# revision after a failed checkpoint, a re-teach after a failed recall check —
# and both store `sequence: None` to say so.
OFF_SEQUENCE_KINDS: frozenset[str] = frozenset({"revision", "reteach"})


def coverage_may_end(sequence: int) -> bool:
    """Whether a topic that has had `sequence` teaching digests may hand over to
    the checkpoint.

    Coverage answers "has this been taught?", which on a narrow topic is true
    after one digest. That is a correct answer to the wrong question: the recall
    check rides the second digest, so ending the feed at the first means the
    topic never has one, and acknowledging its only digest means "dismissed"
    rather than "read". The floor buys exactly one check per topic.

    It delays the checkpoint; it never skips it. A topic held here is still
    `in_progress`, and the digest it gets is written against what the last one
    didn't reach.
    """
    return sequence >= DIGEST_MIN_BEFORE_CHECKPOINT


def teaching_digests(digests: list[dict]) -> list[dict]:
    """Just the drip-feed, in order, so position == sequence.

    Everything that reasons about the cadence — what number the next digest is,
    which check gates it, what a new check covers — has to count only these. A
    re-teach landing mid-topic would otherwise take the next digest's number with
    it, moving where the checks fall and breaking the positional lookup below.
    """
    return [d for d in digests if d.get("kind") not in OFF_SEQUENCE_KINDS]


def digest_quiz_window(prior: list[dict]) -> list[dict]:
    """The digests a new check should cover: everything since the last one.

    Never includes the digest the check is attached to — the learner hasn't read
    it yet, so quizzing on it would make marking impossible. #2 covers #1, #4
    covers #2 and #3, #6 covers #4 and #5. Re-asking about material an earlier
    check already cleared would make each check longer than the last for no gain.
    """
    teaching = teaching_digests(prior)
    return teaching[max(len(teaching) + 1 - 3, 0) :]


def digest_quiz_gate(prior: list[dict]) -> Optional[int]:
    """The sequence of an outstanding recall check blocking the next digest, or
    None when the way is clear.

    Marking a check-bearing digest requires passing its check, so `marked` is the
    record that it was passed — there is no separate flag to consult.

    Only a check-bearing digest is gated: #4 waits on #2's check, but #3 arrives
    while that check is still outstanding. That's what keeps the DIGEST_MAX_UNREAD
    buffer usable — the learner can read one ahead, just not indefinitely.
    """
    teaching = teaching_digests(prior)
    nxt = len(teaching) + 1
    if not digest_carries_quiz(nxt) or nxt < 4:
        return None
    blocking = nxt - DIGEST_QUIZ_EVERY
    return blocking if teaching[blocking - 1].get("status") != "marked" else None


async def topic_digest_states(
    user_id: str, roadmapId: str, topicId: Optional[str]
) -> list[dict]:
    """Every digest for a topic as `{status, kind}`, oldest first.

    A projection, not `topic_digests`: the questions asked before spending a web
    search and an LLM call — how many are unread, and is a check outstanding —
    need nothing else. `kind` rides along because a revision or re-teach digest
    sits outside the drip-feed, so the cadence has to be counted without them
    (see `teaching_digests`); among the rest, position is the sequence, since
    digests are only ever appended.
    """
    try:
        cursor = (
            get_db()[DIGESTS]
            .find(
                {"user_id": user_id, "roadmapId": roadmapId, "topicId": topicId},
                projection={"status": 1, "kind": 1},
            )
            .sort([("createdAt", 1), ("_id", 1)])
        )
        return await cursor.to_list(None)
    except Exception as e:
        logger.error("topic_digest_states error user=%s: %s", user_id, e)
        # Fail closed, as `unread_digest_count` does: an unreadable history must
        # not read as "nothing sent yet, generate away".
        return [{"status": "unread"}] * DIGEST_MAX_UNREAD


async def list_digests(
    user_id: str,
    status: Optional[str] = None,
    active_only: bool = False,
    limit: int = 20,
    roadmapId: Optional[str] = None,
    topicId: Optional[str] = None,
) -> list[dict]:
    """A learner's digests, newest first.

    `active_only` narrows to roadmaps still in play — the catch-up view exists to
    surface what's outstanding, and a digest for an archived roadmap isn't.

    `roadmapId`/`topicId` narrow to one roadmap and then one of its topics, which
    is how the digest archive is browsed. Filtering here rather than in the client
    keeps `limit` meaning "the newest N of what you asked for" — paginating the
    whole history and then discarding most of it client-side would show a nearly
    empty page for any topic that hasn't been written about recently.
    """
    query: dict = {"user_id": user_id}
    if status == "unread":
        query["status"] = {"$ne": "marked"}
    elif status:
        query["status"] = status
    if topicId:
        query["topicId"] = topicId

    labels = await _label_map(user_id)
    if active_only:
        try:
            cursor = get_db()["roadmaps"].find(
                {"user_id": user_id, "status": "active"}, projection={"_id": 1}
            )
            ids = [str(r["_id"]) for r in await cursor.to_list(None)]
        except Exception as e:
            logger.error("list_digests roadmap filter error user=%s: %s", user_id, e)
            return []
        # The two narrow together rather than one replacing the other: asking for
        # a parked roadmap's digests under active_only must return nothing, not
        # quietly widen back to every active roadmap.
        if roadmapId:
            ids = [i for i in ids if i == roadmapId]
        if not ids:
            return []
        query["roadmapId"] = {"$in": ids}
    elif roadmapId:
        query["roadmapId"] = roadmapId

    try:
        cursor = (
            get_db()[DIGESTS]
            .find(query)
            .sort([("createdAt", -1), ("_id", -1)])
            .limit(max(limit, 1))
        )
        docs = await cursor.to_list(None)
    except Exception as e:
        logger.error("list_digests error user=%s: %s", user_id, e)
        return []

    # Attach each recall check's questions so the card can render it without a
    # second round trip. One query for the whole page, and the answer key is
    # stripped — grading stays server-side.
    quiz_ids = [d["quizId"] for d in docs if d.get("quizId")]
    quizzes: dict[str, list] = {}
    if quiz_ids:
        try:
            found = await (
                get_db()["quizzes"]
                .find({"_id": {"$in": [ObjectId(q) for q in quiz_ids]}})
                .to_list(None)
            )
            quizzes = {
                str(q["_id"]): [public_question(x) for x in q.get("questions") or []]
                for q in found
            }
        except Exception as e:
            logger.error("list_digests quiz fetch error user=%s: %s", user_id, e)

    for d in docs:
        d["_id"] = str(d["_id"])
        d.setdefault("status", "unread")
        # The card maps over both without checking. Every writer sets them, so a
        # digest missing one is a hand-inserted or half-written document — and the
        # cost of that reaching a client is a blank screen for the whole queue,
        # not one odd-looking card.
        d.setdefault("bullets", [])
        d.setdefault("resources", [])
        d["roadmapTitle"] = (labels.get(d.get("roadmapId")) or {}).get("title")
        d["quiz"] = quizzes.get(d.get("quizId")) or []
    return docs


async def learning_focus(user_id: str) -> dict:
    """Every roadmap the learner has in play, and when the next digest is due.

    The home screen's answer to "nothing waiting — so what now?". One entry per
    active roadmap rather than a single "current" one, because the daily sweep
    generates a digest for each of them — reporting only the most recently
    touched hid the other queues the learner is actually accumulating.

    Every entry carries a `blocked_reason` rather than going quiet, because "no
    digest is coming" always has a cause worth showing: the topic is finished,
    the backlog is full, a recall check is outstanding, or digests are off.
    """
    # The schedule is per-user, not per-roadmap — one trigger drives the sweep
    # across all of them — so it's resolved once and reported at the top.
    next_at = None
    digests_off = False
    try:
        trig = await get_db()["triggers"].find_one(
            {"user_id": user_id, "action_type": "learning_digest"}
        )
        # No trigger means never opted in, and a trigger with no next run means
        # switched off. Either way the scheduler won't fire, though digests can
        # still be pulled by hand.
        next_at = next_run_at(trig) if trig else None
        digests_off = next_at is None
    except Exception as e:
        # Can't tell — so don't claim they're off.
        logger.error("learning_focus trigger error user=%s: %s", user_id, e)

    try:
        roadmaps = await list_roadmaps(user_id, status="active", limit=50)
    except Exception as e:
        logger.error("learning_focus roadmap fetch error user=%s: %s", user_id, e)
        roadmaps = []

    items = []
    for roadmap in roadmaps:
        topics = roadmap.get("topics") or []
        topic = in_progress_topic(roadmap)
        awaiting = next(
            (t for t in topics if t.get("progress_status") == "needs_review"), None
        )
        # A topic owed revision is also `needs_review`, but pointing the learner
        # at a checkpoint they'd be refused is worse than saying nothing.
        owed = next((t for t in topics if revision_outstanding(t)), None)

        unread = 0
        # Whether the one thing this roadmap is waiting on is already sitting in
        # the queue. Only meaningful while revision is owed.
        revision_waiting = False
        reason = None
        if owed:
            # A failed checkpoint takes precedence over everything else on this
            # roadmap: it's the only thing the learner can act on here.
            states = await topic_digest_states(
                user_id, str(roadmap["_id"]), owed.get("id")
            )
            unread = sum(1 for d in states if d.get("status") != "marked")
            # Counted over revision digests alone, matching what generating would
            # actually produce. A leftover teaching digest on the same topic is
            # not the revision they're owed, and treating it as one reported
            # "revision tips waiting" for tips that were never written.
            revision_waiting = any(
                d.get("kind") == "revision" and d.get("status") != "marked"
                for d in states
            )
            reason = "needs_revision"
        elif topic:
            # One projection answers both questions the generator asks, so the
            # button on screen agrees with what pressing it would actually do.
            states = await topic_digest_states(
                user_id, str(roadmap["_id"]), topic.get("id")
            )
            unread = sum(1 for d in states if d.get("status") != "marked")
            if unread >= DIGEST_MAX_UNREAD:
                reason = "cap_reached"
            elif digest_quiz_gate(states):
                reason = "awaiting_quiz"
            elif digests_off:
                reason = "digests_off"
        elif awaiting:
            # Fully taught; only the checkpoint stands between here and the next
            # topic.
            reason = "needs_review"
        else:
            reason = "roadmap_complete"

        current = owed or topic or awaiting
        items.append(
            {
                "roadmapId": str(roadmap["_id"]),
                "roadmapTitle": roadmap.get("title"),
                "topic": (
                    {
                        "id": current.get("id"),
                        "title": current.get("title"),
                        "progress_status": current.get("progress_status"),
                        "order": current.get("order"),
                    }
                    if current
                    else None
                ),
                "progress": roadmap_progress(roadmap),
                "unread": unread,
                # Generation is meaningful for a topic still being taught, and
                # for one owed revision — where what it produces is the revision
                # digest. Only when nothing the generator checks would refuse it.
                "can_generate": (
                    # The backlog still caps it, which is also what keeps this
                    # failing closed: an unreadable history comes back as a full
                    # queue of unread digests with no kind on them.
                    not revision_waiting and unread < DIGEST_MAX_UNREAD
                    if owed
                    else bool(topic)
                    and reason not in ("cap_reached", "awaiting_quiz")
                ),
                "blocked_reason": reason,
            }
        )

    return {
        "roadmaps": items,
        "unread": sum(i["unread"] for i in items),
        "cap": DIGEST_MAX_UNREAD,
        "next_at": next_at,
        # Account-level only. Anything that blocks a single roadmap is reported
        # on that roadmap's entry, since the others may still be running.
        "blocked_reason": (
            "no_roadmap" if not items else ("digests_off" if digests_off else None)
        ),
    }


async def fetch_digest(digestId: str, user_id: str) -> Optional[dict]:
    try:
        doc = await get_db()[DIGESTS].find_one(
            {"_id": ObjectId(digestId), "user_id": user_id}
        )
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logger.error("fetch_digest error id=%s: %s", digestId, e)
        return None


async def mark_digest(digestId: str, user_id: str) -> bool:
    """Acknowledge a digest. `updatedAt` doubles as when it was noticed, since
    marking is the only thing that ever updates one."""
    try:
        res = await get_db()[DIGESTS].update_one(
            {"_id": ObjectId(digestId), "user_id": user_id},
            {
                "$set": {
                    "status": "marked",
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error("mark_digest error id=%s: %s", digestId, e)
        return False


# ── notes ────────────────────────────────────────────────────────────────────
# Notes live in their own collection rather than inside the roadmap document.
# Two reasons, and the first is the important one:
#   1. A roadmap edit regenerates its topics from an LLM draft. Anything embedded
#      in a topic has to be carefully carried across that (see merge_roadmap) —
#      and the learner's own writing is the last thing that should ever depend on
#      getting that right. Out here, a regeneration cannot touch them.
#   2. "My notes" is a cross-roadmap query. One indexed read beats scanning every
#      roadmap and flattening.
NOTES = "learning_notes"


async def create_note(
    user_id: str,
    roadmapId: str,
    topicId: str,
    kind: str,
    body: str,
    url: Optional[str] = None,
) -> Optional[dict]:
    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "user_id": user_id,
        "roadmapId": roadmapId,
        "topicId": topicId,
        "kind": kind,
        "body": body,
        "url": url,
        "resolved": False,
        "createdAt": now,
        "updatedAt": now,
    }
    try:
        res = await get_db()[NOTES].insert_one(doc)
        return {**doc, "_id": str(res.inserted_id)}
    except Exception as e:
        logger.error("create_note error user=%s: %s", user_id, e)
        return None


async def _label_map(user_id: str) -> dict:
    """{roadmapId: {"title": …, "topics": {topicId: title}}} for decorating notes.

    Resolved at read time rather than denormalised onto each note: a roadmap edit
    can rename a topic, and a note pointing at a stale label is worse than one
    extra query.
    """
    try:
        cursor = get_db()["roadmaps"].find(
            {"user_id": user_id},
            projection={"title": 1, "topics.id": 1, "topics.title": 1},
        )
        return {
            str(r["_id"]): {
                "title": r.get("title"),
                "topics": {t.get("id"): t.get("title") for t in r.get("topics") or []},
            }
            for r in await cursor.to_list(None)
        }
    except Exception as e:
        logger.error("_label_map error user=%s: %s", user_id, e)
        return {}


async def list_notes(
    user_id: str,
    roadmapId: Optional[str] = None,
    topicId: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 100,
    skip: int = 0,
) -> list[dict]:
    """A learner's notes, newest first, decorated with the roadmap and topic they
    belong to. A note whose topic was removed by a later edit still comes back —
    it just carries no topic title. Losing the note would be worse."""
    query: dict = {"user_id": user_id}
    if roadmapId:
        query["roadmapId"] = roadmapId
    if topicId:
        query["topicId"] = topicId
    if kind:
        query["kind"] = kind

    try:
        cursor = (
            get_db()[NOTES]
            .find(query)
            .sort([("createdAt", -1), ("_id", -1)])
            .skip(max(skip, 0))
            .limit(max(limit, 1))
        )
        docs = await cursor.to_list(None)
    except Exception as e:
        logger.error("list_notes error user=%s: %s", user_id, e)
        return []

    labels = await _label_map(user_id)
    for d in docs:
        d["_id"] = str(d["_id"])
        entry = labels.get(d.get("roadmapId")) or {}
        d["roadmapTitle"] = entry.get("title")
        d["topicTitle"] = (entry.get("topics") or {}).get(d.get("topicId"))
    return docs


async def update_note(note_id: str, user_id: str, updates: dict) -> bool:
    if not updates:
        return False
    try:
        res = await get_db()[NOTES].update_one(
            {"_id": ObjectId(note_id), "user_id": user_id},
            {"$set": {**updates, "updatedAt": datetime.now(timezone.utc).isoformat()}},
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error("update_note error id=%s: %s", note_id, e)
        return False


async def delete_note(note_id: str, user_id: str) -> bool:
    try:
        res = await get_db()[NOTES].delete_one(
            {"_id": ObjectId(note_id), "user_id": user_id}
        )
        return res.deleted_count > 0
    except Exception as e:
        logger.error("delete_note error id=%s: %s", note_id, e)
        return False


async def note_counts(user_id: str, roadmapId: str) -> dict:
    """{topicId: count} so the roadmap screen can show which topics have notes
    without pulling the notes themselves."""
    try:
        rows = await (
            get_db()[NOTES]
            .aggregate(
                [
                    {"$match": {"user_id": user_id, "roadmapId": roadmapId}},
                    {"$group": {"_id": "$topicId", "n": {"$sum": 1}}},
                ]
            )
            .to_list(None)
        )
        return {r["_id"]: r["n"] for r in rows if r.get("_id")}
    except Exception as e:
        logger.error("note_counts error user=%s: %s", user_id, e)
        return {}


# ── learner profile ──────────────────────────────────────────────────────────
_LEARNER_MEMORY_INSTRUCTIONS = (
    "Extract durable facts about the learner from their message — skill level "
    "(beginner/intermediate/advanced), preferred resource types "
    "(video/text/interactive), learning goals, weekly availability, and topics "
    "they already know."
)


async def write_memory(user_id: str, query: str, current: Optional[dict] = None):
    """Background task: pull durable learner facts out of the latest message and
    merge them into the user's memory doc. Runs after the response is sent, so it
    adds no latency to /query. Delegates to the shared memory store."""
    await extract_and_save(
        user_id,
        query,
        LearnerMemory,
        _LEARNER_MEMORY_INSTRUCTIONS,
        current,
        namespace=LEARNING_NS,
    )
