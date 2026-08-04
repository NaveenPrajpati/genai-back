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
import uuid
from datetime import date, timedelta, datetime, timezone
from math import ceil
from typing import Optional

from bson import ObjectId

from app.core.config import (
    CHECKPOINT_PASS_SCORE,
    DIGEST_MAX_UNREAD,
    DIGEST_QUIZ_EVERY,
    MAX_ACTIVE_ROADMAPS,
)
from app.database import get_db
from app.agents.memory_store import extract_and_save
from app.agents.trigger_store import next_run_at
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


async def learning_stats(user_id: str) -> dict:
    """Progress aggregated across ALL of a learner's roadmaps, for the landing
    screen.

    Deliberately not folded into `list_roadmaps`: that endpoint is paginated, and
    a global aggregate returned alongside one page of results reads as though it
    describes the page. This also spans `quiz_attempts`, which roadmaps don't
    touch. Two parallel requests beat one misleading response.
    """
    roadmaps = []
    try:
        cursor = get_db()["roadmaps"].find(
            {"user_id": user_id},
            projection={
                "status": 1,
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
    completed_days: list[str] = []  # ISO date per completed topic
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
                # "" for a topic completed without a timestamp: it still counts
                # towards the total, it just can't contribute to a streak.
                completed_days.append((t.get("completed_at") or "")[:10])
                due = t.get("next_review_at")
                if due and due <= stamp:
                    reviews_due += 1

    completed_topics = len(completed_days)
    dated = {d for d in completed_days if d}

    # Consecutive days ending today. An empty today doesn't break the streak —
    # it only breaks once a full day has been missed.
    today = datetime.now(timezone.utc).date()
    day = today if today.isoformat() in dated else today - timedelta(days=1)
    streak = 0
    while day.isoformat() in dated:
        streak += 1
        day -= timedelta(days=1)

    week_start = (today - timedelta(days=6)).isoformat()
    this_week = sum(1 for d in completed_days if d and d >= week_start)

    attempts, average = 0, 0
    try:
        rows = await (
            get_db()["quiz_attempts"]
            .find({"user_id": user_id}, projection={"score": 1})
            .to_list(None)
        )
        scores = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
        attempts = len(rows)
        average = round(sum(scores) / len(scores)) if scores else 0
    except Exception as e:
        logger.error("learning_stats quiz error user=%s: %s", user_id, e)

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
        "quizzes": {"attempts": attempts, "average_score": average},
    }


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
    roadmapId: str, topicId: str, user_id: str, score: int
) -> Optional[dict]:
    """Fold a graded checkpoint into the topic: mastery, completion, next review.

    Passing is what completes a topic — the point of gating completion on active
    recall rather than on a checkbox.

    Failing is deliberately asymmetric. A failed *first* attempt leaves the topic
    `in_progress` and simply doesn't complete it. A failed *review* does NOT
    un-complete a topic the learner already finished; it drags the next review
    back to the front of the ladder instead. Clawing back progress for an honest
    attempt would punish the exact behaviour this feature exists to encourage.
    """
    roadmap = await fetch_roadmap(roadmapId, user_id)
    topic = find_topic(roadmap, topicId)
    if topic is None:
        return None

    passed = score >= CHECKPOINT_PASS_SCORE
    was_completed = topic.get("progress_status") == "completed"
    review_count = (int(topic.get("review_count") or 0) + 1) if passed else 0
    status = "completed" if (passed or was_completed) else "in_progress"

    now = datetime.now(timezone.utc).isoformat()
    updates = {
        "topics.$.progress_status": status,
        "topics.$.mastery_score": score,
        "topics.$.review_count": review_count,
        "topics.$.next_review_at": next_review_at(review_count),
        "updated_at": now,
    }
    if status == "completed" and not topic.get("completed_at"):
        updates["topics.$.completed_at"] = now

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
    if passed and not was_completed:
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
        "next_review_at": updates["topics.$.next_review_at"],
        "was_review": was_completed,
        # The topic that just picked up the baton, if any.
        "advanced_to": advanced,
    }


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
def grade_quiz(questions: list[dict], selected: dict[int, int]) -> dict:
    """Score a submission against the stored questions. Shared by the chat path
    and POST /submit-quiz so the two can't grade the same answers differently.
    `review` carries only what the learner got wrong, unanswered included."""
    correct = 0
    review = []
    for idx, q in enumerate(questions):
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

    return {
        "total": len(questions),
        "correct": correct,
        "score": round(correct / len(questions) * 100) if questions else 0,
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


async def record_quiz_attempt(
    user_id: str,
    quizId: Optional[str],
    roadmapId: Optional[str],
    topicId: Optional[str],
    result: dict,
) -> None:
    """Persist one graded attempt. Nothing reads it yet — it's the record that
    makes score history and weak-topic detection possible later, for the cost of
    one insert instead of data we never captured."""
    try:
        await get_db()["quiz_attempts"].insert_one(
            {
                "user_id": user_id,
                "quizId": quizId,
                "roadmapId": roadmapId,
                "topicId": topicId,
                "total": result.get("total"),
                "correct": result.get("correct"),
                "score": result.get("score"),
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as e:
        logger.error("record_quiz_attempt error user=%s: %s", user_id, e)


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


def digest_quiz_window(prior: list[dict]) -> list[dict]:
    """The digests a new check should cover: everything since the last one.

    Never includes the digest the check is attached to — the learner hasn't read
    it yet, so quizzing on it would make marking impossible. #2 covers #1, #4
    covers #2 and #3, #6 covers #4 and #5. Re-asking about material an earlier
    check already cleared would make each check longer than the last for no gain.
    """
    return prior[max(len(prior) + 1 - 3, 0) :]


def digest_quiz_gate(prior: list[dict]) -> Optional[int]:
    """The sequence of an outstanding recall check blocking the next digest, or
    None when the way is clear.

    Marking a check-bearing digest requires passing its check, so `marked` is the
    record that it was passed — there is no separate flag to consult.

    Only a check-bearing digest is gated: #4 waits on #2's check, but #3 arrives
    while that check is still outstanding. That's what keeps the DIGEST_MAX_UNREAD
    buffer usable — the learner can read one ahead, just not indefinitely.
    """
    nxt = len(prior) + 1
    if not digest_carries_quiz(nxt) or nxt < 4:
        return None
    blocking = nxt - DIGEST_QUIZ_EVERY
    return blocking if prior[blocking - 1].get("status") != "marked" else None


async def topic_digest_states(
    user_id: str, roadmapId: str, topicId: Optional[str]
) -> list[dict]:
    """Every digest for a topic as `{status}`, oldest first.

    A projection, not `topic_digests`: the two questions asked before spending a
    web search and an LLM call — how many are unread, and is a check outstanding
    — need nothing else. Position is the sequence, since digests are only ever
    appended.
    """
    try:
        cursor = (
            get_db()[DIGESTS]
            .find(
                {"user_id": user_id, "roadmapId": roadmapId, "topicId": topicId},
                projection={"status": 1},
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
                str(q["_id"]): [
                    {"question": x.get("question"), "options": x.get("options")}
                    for x in q.get("questions") or []
                ]
                for q in found
            }
        except Exception as e:
            logger.error("list_digests quiz fetch error user=%s: %s", user_id, e)

    for d in docs:
        d["_id"] = str(d["_id"])
        d.setdefault("status", "unread")
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

        unread = 0
        reason = None
        if topic:
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

        current = topic or awaiting
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
                # Generation is only meaningful for a topic still being taught,
                # and only when nothing the generator checks would refuse it.
                "can_generate": bool(topic)
                and reason not in ("cap_reached", "awaiting_quiz"),
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
