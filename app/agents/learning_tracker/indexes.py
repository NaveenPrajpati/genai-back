"""The indexes the learning tracker's own collections need.

Every read in this feature is scoped by `user_id` and then narrowed by roadmap
and topic, and most of them sort. Unindexed, each one is a collection scan — and
the two that hurt first are the ones a learner triggers by *looking*:
`learning_stats` reads every quiz attempt and every roadmap the user owns on each
landing-screen load, and `list_digests` sorts the whole digest history to return
the newest twenty.

Declared here rather than inline at the call sites so the set can be read in one
go against the queries it serves, and created on startup because nothing else
creates them — Atlas will happily scan forever.

Only collections this feature owns. `triggers` is shared with the other agents
and is left to them.

Run by hand against whatever MONGO_URI points at:

    .venv/bin/python -m app.agents.learning_tracker.indexes
"""

import asyncio
import logging

from app.database import get_db

from .repository import DIGESTS, MISCONCEPTIONS, NOTES

logger = logging.getLogger(__name__)

# (collection, keys, name, why). The name is explicit so a re-run is idempotent
# and so a stale index is identifiable later; Mongo's generated names encode the
# keys but not the intent.
INDEXES: tuple[tuple[str, list[tuple[str, int]], str, str], ...] = (
    (
        "roadmaps",
        [("user_id", 1), ("status", 1), ("updated_at", -1)],
        "learning_roadmaps_by_owner",
        # list_roadmaps (user + optional status, newest first), active_roadmaps
        # (the cap check on every activation), resolve_roadmap_id (status $nin,
        # sorted), and the user_id prefix carries learning_stats' full scan.
        "the roadmap list, the active-roadmap cap, and 'what am I on?'",
    ),
    (
        "roadmaps",
        [("user_id", 1), ("topics.next_review_at", 1)],
        "learning_reviews_due",
        # due_reviews matches on a field inside the topics array — multikey, and
        # without it every roadmap document is unpacked on every reviews poll.
        "spaced-repetition reviews coming due",
    ),
    (
        DIGESTS,
        [("user_id", 1), ("roadmapId", 1), ("topicId", 1), ("createdAt", 1)],
        "learning_digests_by_topic",
        # topic_digests and topic_digest_states read a topic's history in order
        # before every generation; list_digests uses the same prefix and sorts
        # createdAt the other way, which an index serves in either direction.
        "a topic's digest history, and the archive",
    ),
    (
        "quizzes",
        [("user_id", 1), ("roadmapId", 1), ("topicId", 1), ("kind", 1)],
        "learning_quizzes_by_topic",
        # The checkpoint resume lookup and asked_questions, both of which filter
        # on kind — the filter that stops a digest check being handed back as a
        # checkpoint.
        "issuing a checkpoint without repeating a question",
    ),
    (
        "quiz_attempts",
        [("user_id", 1), ("roadmapId", 1), ("topicId", 1), ("createdAt", -1)],
        "learning_attempts_by_topic",
        # recent_attempts (the retry limits), topic_misses (the misconception
        # evidence window), failed_attempts (the re-teach escalation), and the
        # user_id prefix for learning_stats' mastery pass.
        "retry limits, mastery, and the misconception evidence",
    ),
    (
        "quiz_attempts",
        [("user_id", 1), ("quizId", 1)],
        "learning_attempts_by_quiz",
        # quiz_graded, on the path of every checkpoint submission.
        "the one-grading-per-set rule",
    ),
    (
        NOTES,
        [("user_id", 1), ("roadmapId", 1), ("topicId", 1), ("createdAt", -1)],
        "learning_notes_by_scope",
        # list_notes at all three scopes, and note_counts' aggregate match.
        "the notes list and the per-topic counts",
    ),
    (
        MISCONCEPTIONS,
        [("user_id", 1), ("roadmapId", 1), ("topicId", 1)],
        "learning_misconceptions_by_topic",
        # get_misconceptions on every checkpoint and every digest, and the key
        # save_misconceptions upserts on. Deliberately NOT unique: a duplicate
        # would be a bug, but failing index creation at boot over one is worse
        # than the duplicate.
        "the cached analysis behind digests and checkpoints",
    ),
)


async def ensure_indexes() -> int:
    """Create any that are missing. Returns how many are now in place.

    Best-effort and idempotent: `create_index` on an existing index is a no-op,
    and a failure is logged rather than raised. This runs during startup, and an
    index that didn't get created is a slow app — not a broken one, and certainly
    not a reason to refuse to boot.
    """
    created = 0
    db = get_db()
    for collection, keys, name, why in INDEXES:
        try:
            await db[collection].create_index(keys, name=name, background=True)
            created += 1
        except Exception as e:
            logger.warning("index %s on %s not created (%s) — %s", name, collection, e, why)
    logger.info("learning indexes ensured: %s/%s", created, len(INDEXES))
    return created


# The query shapes worth checking, as (label, collection, filter, sort). Written
# out rather than derived from the call sites: the point is to notice when a call
# site changes shape and stops matching the index that was built for it.
_SHAPES = (
    (
        "list_digests / topic_digests",
        DIGESTS,
        {"user_id": "u", "roadmapId": "r", "topicId": "t"},
        [("createdAt", -1), ("_id", -1)],
    ),
    (
        "recent_attempts (retry limits)",
        "quiz_attempts",
        {"user_id": "u", "roadmapId": "r", "topicId": "t"},
        [("createdAt", -1), ("_id", -1)],
    ),
    (
        "asked_questions / checkpoint resume",
        "quizzes",
        {"user_id": "u", "roadmapId": "r", "topicId": "t", "kind": {"$in": ["checkpoint"]}},
        [("_id", -1)],
    ),
    (
        "list_roadmaps",
        "roadmaps",
        {"user_id": "u", "status": "active"},
        [("updated_at", -1), ("_id", -1)],
    ),
    (
        "due_reviews (multikey)",
        "roadmaps",
        {"user_id": "u", "topics.next_review_at": {"$lte": "2026-01-01"}},
        None,
    ),
    (
        "list_notes",
        NOTES,
        {"user_id": "u"},
        [("createdAt", -1), ("_id", -1)],
    ),
    (
        "get_misconceptions",
        MISCONCEPTIONS,
        {"user_id": "u", "roadmapId": "r", "topicId": "t"},
        None,
    ),
)


def _stages(plan: dict) -> list[str]:
    """The winning plan flattened into its stage names."""
    out = []
    while plan:
        out.append(plan.get("stage"))
        plan = plan.get("inputStage") or (plan.get("inputStages") or [None])[0]
    return [s for s in out if s]


async def explain() -> int:
    """Ask the planner whether each shape actually uses an index.

    Read-only — `queryPlanner` verbosity plans the query without running it — so
    this is safe against a live database, and more honest than a synthetic one:
    it reports what your real data would do. Returns the number of shapes still
    on a collection scan.
    """
    db = get_db()
    scans = 0
    for label, collection, query, sort in _SHAPES:
        cursor = db[collection].find(query)
        if sort:
            cursor = cursor.sort(sort)
        try:
            plan = (await cursor.explain())["queryPlanner"]["winningPlan"]
        except Exception as e:
            print(f"  ? {label:38} explain failed: {e}")
            continue
        stages = _stages(plan)
        index = plan
        while index and index.get("stage") != "IXSCAN":
            index = index.get("inputStage") or (index.get("inputStages") or [None])[0]
        if index:
            print(f"  ✓ {label:38} {index.get('indexName')}")
        else:
            scans += 1
            # On an empty collection there is nothing to plan around, so this is
            # only meaningful once there is data.
            n = await db[collection].estimated_document_count()
            note = " (collection is empty)" if not n else ""
            print(f"  ✗ {label:38} {' → '.join(stages)}{note}")
    return scans


async def _main() -> None:
    import sys

    from app.database import close_db, connect_db

    await connect_db()
    try:
        db = get_db()
        if "--explain" in sys.argv:
            print("Query plans (read-only, nothing is executed):\n")
            scans = await explain()
            print(f"\n{len(_SHAPES) - scans}/{len(_SHAPES)} shapes served by an index")
            return
        created = await ensure_indexes()
        print(f"{created}/{len(INDEXES)} learning indexes in place\n")
        for collection in sorted({c for c, *_ in INDEXES}):
            names = sorted(n for n in await db[collection].index_information() if n != "_id_")
            print(f"  {collection}: {', '.join(names) or '(none)'}")
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(_main())
