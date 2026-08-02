"""Tests for the learning-tracker agent.

Offline: MongoDB and the LLM are replaced with fakes. These lock in the
invariants the tranche-1/2 hardening established, each of which was a live bug:

  * every roadmap read/write carries the caller's user_id (cross-tenant access)
  * a roadmap edit never costs the learner their progress
  * the server, not the model, owns ids, lifecycle status, and progress
  * every intent routes to a node that actually exists
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.agents.learning_tracker.repository as repo
import app.agents.learning_tracker.workflow as lt
from app.agents.learning_tracker.state import (
    RoadmapDraft,
    ResourceDraft,
    StageDraft,
    TopicDraft,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _draft(*topics: TopicDraft, title: str = "Rust") -> RoadmapDraft:
    return RoadmapDraft(
        title=title,
        summary="learn rust",
        stages=[StageDraft(order=1, title="Foundations")],
        topics=list(topics),
    )


def _topic(order: int, title: str, existing_id: str | None = None) -> TopicDraft:
    return TopicDraft(
        order=order,
        stage_order=1,
        title=title,
        description=f"about {title}",
        existing_id=existing_id,
    )


def _collection():
    """A Mongo collection double that records the filters it was queried with."""
    col = MagicMock()
    col.find_one = AsyncMock(return_value=None)
    col.update_one = AsyncMock(return_value=MagicMock(matched_count=1, modified_count=1))
    col.replace_one = AsyncMock(return_value=MagicMock(matched_count=1))
    col.insert_one = AsyncMock(return_value=MagicMock(inserted_id="rid"))
    return col


def _patch_db(monkeypatch, col):
    monkeypatch.setattr(repo, "get_db", lambda: {
        "roadmaps": col, "quizzes": col, "quiz_attempts": col
    })
    return col


_OID = "507f1f77bcf86cd799439011"


# --------------------------------------------------------------------------- #
# A1 — every roadmap access is scoped to the caller
# --------------------------------------------------------------------------- #
async def test_fetch_roadmap_scopes_to_the_owner(monkeypatch):
    col = _patch_db(monkeypatch, _collection())
    await repo.fetch_roadmap(_OID, "userA")
    assert col.find_one.await_args.args[0]["user_id"] == "userA"


async def test_replace_roadmap_cannot_take_over_another_users_roadmap(monkeypatch):
    """The bug this replaces stamped the caller's user_id onto a roadmap matched
    by _id alone — a full takeover by guessing an id."""
    col = _patch_db(monkeypatch, _collection())
    roadmap = repo.materialize_roadmap(_draft(_topic(1, "Ownership")))

    await repo.replace_roadmap(_OID, "attacker", roadmap)

    filt = col.replace_one.await_args.args[0]
    assert filt["user_id"] == "attacker"  # scoped, so a foreign roadmap won't match


@pytest.mark.parametrize(
    "call",
    [
        lambda: repo.set_topic_progress(_OID, "t1", "completed", "userA"),
        lambda: repo.set_topic_resources(_OID, "t1", [], "userA"),
        lambda: repo.set_roadmap_status(_OID, "userA", "archived"),
    ],
)
async def test_every_topic_write_is_user_scoped(monkeypatch, call):
    col = _patch_db(monkeypatch, _collection())
    await call()
    assert all(
        c.args[0].get("user_id") == "userA" for c in col.update_one.await_args_list
    )


async def test_fetch_quiz_scopes_to_the_owner(monkeypatch):
    col = _patch_db(monkeypatch, _collection())
    await repo.fetch_quiz("userA", _OID)
    assert col.find_one.await_args.args[0]["user_id"] == "userA"


async def test_malformed_roadmap_id_returns_none_rather_than_raising(monkeypatch):
    _patch_db(monkeypatch, _collection())
    assert await repo.fetch_roadmap("not-an-objectid", "userA") is None


# --------------------------------------------------------------------------- #
# A2 — a modify never costs the learner their progress
# --------------------------------------------------------------------------- #
_STORED = {
    "created_at": "2026-01-01T00:00:00+00:00",
    "status": "active",
    "topics": [
        {
            "id": "keep-1",
            "order": 1,
            "title": "Ownership",
            "progress_status": "completed",
            "mastery_score": 90,
            "completed_at": "2026-02-02T00:00:00+00:00",
        },
        {"id": "keep-2", "order": 2, "title": "Borrowing"},
    ],
}


def test_merge_preserves_progress_for_echoed_topics():
    merged = repo.merge_roadmap(
        _STORED,
        _draft(_topic(1, "Ownership", existing_id="keep-1"), _topic(2, "Traits")),
    )

    kept = merged.topics[0]
    assert kept.id == "keep-1"  # id survives, so PA source_ref dedup still holds
    assert kept.progress_status == "completed"
    assert kept.mastery_score == 90
    assert kept.completed_at == "2026-02-02T00:00:00+00:00"

    # A genuinely new topic starts clean with a fresh server-minted id — no
    # earned progress carried in from anywhere.
    added = merged.topics[1]
    assert added.mastery_score is None and added.completed_at is None
    assert added.id not in {"keep-1", "keep-2"}


def test_merge_falls_back_to_title_when_the_model_drops_the_id():
    merged = repo.merge_roadmap(_STORED, _draft(_topic(1, "  ownership  ")))
    assert merged.topics[0].id == "keep-1"
    assert merged.topics[0].progress_status == "completed"


def test_merge_ignores_an_unknown_existing_id():
    """An id the model invented must not silently bind to anything."""
    merged = repo.merge_roadmap(_STORED, _draft(_topic(1, "Macros", existing_id="nope")))
    assert merged.topics[0].id not in {"keep-1", "keep-2"}
    assert merged.topics[0].mastery_score is None
    assert merged.topics[0].completed_at is None


def test_merge_lets_each_stored_topic_be_claimed_once():
    """Repeating an id must not copy one topic's progress onto several."""
    merged = repo.merge_roadmap(
        _STORED,
        _draft(
            _topic(1, "Ownership", existing_id="keep-1"),
            _topic(2, "Ownership again", existing_id="keep-1"),
        ),
    )
    assert merged.topics[0].id == "keep-1"
    assert merged.topics[1].id != "keep-1"
    assert merged.topics[1].completed_at is None  # didn't inherit keep-1's progress


def test_merge_keeps_the_original_creation_time():
    merged = repo.merge_roadmap(_STORED, _draft(_topic(1, "Ownership")))
    assert merged.created_at == "2026-01-01T00:00:00+00:00"


# --------------------------------------------------------------------------- #
# A3 — the server owns identity, lifecycle, and progress
# --------------------------------------------------------------------------- #
def test_the_model_cannot_set_status_ids_or_progress():
    # RoadmapDraft has no status/id/progress fields at all — the LLM is never
    # offered them. What the server produces is active, id'd, and unstarted.
    assert not {"status", "id", "progress_status"} & set(
        RoadmapDraft.model_fields
    ) | set(TopicDraft.model_fields) & {"progress_status", "mastery_score"}

    roadmap = repo.materialize_roadmap(_draft(_topic(1, "Ownership")))
    assert roadmap.status == "active"  # never the old "archived" default
    assert roadmap.topics[0].id and roadmap.topics[0].completed_at is None
    assert roadmap.topics[0].mastery_score is None


def test_topics_get_distinct_ids_and_are_linked_to_their_stage():
    roadmap = repo.materialize_roadmap(
        _draft(_topic(1, "Ownership"), _topic(2, "Borrowing"))
    )
    ids = [t.id for t in roadmap.topics]
    assert len(set(ids)) == 2
    assert roadmap.topics[0].stage_id == roadmap.stages[0].id


def test_resources_survive_materialization():
    draft = _draft(_topic(1, "Ownership"))
    draft.topics[0].resources = [
        ResourceDraft(title="The Book", url="https://doc.rust-lang.org", resource_type="book")
    ]
    roadmap = repo.materialize_roadmap(draft)
    assert roadmap.topics[0].resources[0].resource_type == "book"


# --------------------------------------------------------------------------- #
# progress reading
# --------------------------------------------------------------------------- #
def test_active_topic_skips_completed_and_skipped():
    roadmap = {
        "topics": [
            {"id": "a", "order": 1, "title": "A", "progress_status": "completed"},
            {"id": "b", "order": 2, "title": "B", "progress_status": "skipped"},
            {"id": "c", "order": 3, "title": "C"},
        ]
    }
    assert repo.active_topic(roadmap)["id"] == "c"


def test_roadmap_progress_counts_only_completed():
    roadmap = {
        "topics": [
            {"id": "a", "order": 1, "title": "A", "progress_status": "completed"},
            {"id": "b", "order": 2, "title": "B", "progress_status": "skipped"},
            {"id": "c", "order": 3, "title": "C"},
        ]
    }
    progress = repo.roadmap_progress(roadmap)
    assert progress["completed_count"] == 1
    assert progress["total"] == 3
    assert progress["percent"] == 33
    assert progress["next_topic"] == "C"  # skipped is behind us, not next


def test_roadmap_progress_on_no_roadmap_is_empty_not_a_crash():
    assert repo.roadmap_progress(None) == {
        "next_topic": None,
        "next_topic_id": None,
        "completed_count": 0,
        "remaining": 0,
        "total": 0,
        "percent": 0,
    }


# --------------------------------------------------------------------------- #
# A4 — quiz grading
# --------------------------------------------------------------------------- #
_QUESTIONS = [
    {"question": "q0", "options": ["a", "b"], "answer": 1},
    {"question": "q1", "options": ["a", "b", "c"], "answer": 2},
]


def test_grade_quiz_scores_and_reviews_only_the_misses():
    result = repo.grade_quiz(_QUESTIONS, {0: 1, 1: 0})
    assert (result["correct"], result["total"], result["score"]) == (1, 2, 50)
    assert [r["question"] for r in result["review"]] == [1]
    assert result["review"][0]["correctOption"] == "c"


def test_grade_quiz_treats_unanswered_as_wrong_not_correct():
    """`selected.get(idx)` is None for a skipped question; None must never match
    an answer of None-ish shape."""
    result = repo.grade_quiz([{"question": "q", "options": ["a"], "answer": None}], {})
    assert result["correct"] == 0


def test_grade_quiz_handles_an_empty_quiz():
    assert repo.grade_quiz([], {})["score"] == 0


# --------------------------------------------------------------------------- #
# routing + graph wiring
# --------------------------------------------------------------------------- #
def test_every_intent_routes_to_a_node_that_exists():
    """submit_quiz used to route to a node that was never registered, which is a
    KeyError at runtime rather than a compile-time error."""
    nodes = set(lt.build_graph().nodes)
    assert set(lt.INTENT_ROUTES.values()) <= nodes


def test_every_classifier_intent_has_a_route():
    from app.agents.learning_tracker.state import IntentOutput

    declared = set(IntentOutput.model_fields["intent"].annotation.__args__)
    assert declared == set(lt.INTENT_ROUTES)


def test_unknown_intent_falls_back_instead_of_dead_ending():
    assert lt.decide_agent({"intent": "something_new"}) == "fallback_agent"
    assert lt.decide_agent({}) == "fallback_agent"


def test_onboarding_runs_once_then_never_again():
    assert lt.decide_onboarding({"memory": {}}) == "onboard"
    assert lt.decide_onboarding({"memory": {"skill_level": "beginner"}}) == "onboard"
    # set even when the learner skips, so declining doesn't re-prompt forever
    assert lt.decide_onboarding({"memory": {"onboarded": True}}) == "classify_intent"


async def test_load_memory_resolves_the_active_roadmap(monkeypatch):
    """A bare "what should I study next?" carries no roadmapId."""
    monkeypatch.setattr(lt, "get_profile", AsyncMock(return_value={}))
    monkeypatch.setattr(lt, "resolve_roadmap_id", AsyncMock(return_value="resolved"))

    out = await lt.load_memory({"user_id": "userA", "roadmapId": None})
    assert out["roadmapId"] == "resolved"


async def test_onboarding_keeps_only_the_keys_it_asked_for(monkeypatch):
    saved = AsyncMock()
    monkeypatch.setattr(lt, "save_profile", saved)
    monkeypatch.setattr(
        lt,
        "interrupt",
        lambda _payload: {"skill_level": "advanced", "is_admin": True},
    )

    await lt.onboard({"user_id": "userA", "memory": {}})

    profile = saved.await_args.args[1]
    assert profile == {"skill_level": "advanced", "onboarded": True}


async def test_skipping_onboarding_still_marks_it_done(monkeypatch):
    saved = AsyncMock()
    monkeypatch.setattr(lt, "save_profile", saved)
    monkeypatch.setattr(lt, "interrupt", lambda _payload: None)

    await lt.onboard({"user_id": "userA", "memory": {}})
    assert saved.await_args.args[1] == {"onboarded": True}


# --------------------------------------------------------------------------- #
# A7 — never report a save that didn't happen
# --------------------------------------------------------------------------- #
async def test_a_failed_persist_is_not_reported_as_approved(monkeypatch):
    monkeypatch.setattr(lt, "get_pending", AsyncMock(return_value=None))
    monkeypatch.setattr(lt, "create_pending", AsyncMock(return_value="ap1"))
    monkeypatch.setattr(lt, "build_roadmap", AsyncMock(return_value=_draft(_topic(1, "A"))))
    monkeypatch.setattr(lt, "insertRoadmapToDb", AsyncMock(return_value=None))  # Mongo down
    monkeypatch.setattr(lt, "interrupt", lambda _payload: "approved")
    resolved = AsyncMock()
    monkeypatch.setattr(lt, "resolve", resolved)
    sync = AsyncMock()
    monkeypatch.setattr(lt, "sync_roadmap_to_pa", sync)

    out = await lt.roadmap_agent(
        {"user_id": "userA", "thread_id": "t1", "intent": "create_roadmap", "query": "rust"}
    )

    assert out["roadmap_status"] == "save_failed"
    # The approval stays pending so a retry reuses the draft instead of paying
    # for a second generation, and no PA to-dos are created for a lost roadmap.
    resolved.assert_not_awaited()
    sync.assert_not_awaited()


async def test_modifying_a_roadmap_you_dont_own_creates_nothing(monkeypatch):
    monkeypatch.setattr(lt, "fetch_roadmap", AsyncMock(return_value=None))
    insert = AsyncMock()
    monkeypatch.setattr(lt, "insertRoadmapToDb", insert)

    out = await lt.roadmap_agent(
        {
            "user_id": "attacker",
            "thread_id": "t1",
            "intent": "modify_roadmap",
            "roadmapId": _OID,
            "query": "add a topic",
        }
    )

    assert out["roadmap_status"] == "not_found"
    insert.assert_not_awaited()


# --------------------------------------------------------------------------- #
# progress write/read contract
#
# A tracker stuck at 0% is the symptom of the writer and the readers disagreeing
# about which field holds progress. These assert they agree, field for field.
# --------------------------------------------------------------------------- #
async def test_completing_a_topic_writes_the_field_the_readers_count(monkeypatch):
    col = _patch_db(monkeypatch, _collection())
    # The rollup re-reads the roadmap; hand it back the topic we just wrote.
    col.find_one = AsyncMock(
        return_value={"status": "active", "topics": [{"progress_status": "completed"}]}
    )

    assert await repo.set_topic_progress(_OID, "t1", "completed", "userA") is True

    written = col.update_one.await_args_list[0].args[1]["$set"]
    assert written["topics.$.progress_status"] == "completed"
    assert written["topics.$.completed_at"]  # a streak needs this stamp

    # The exact document the writer produces, as one stored topic.
    stored = {
        "id": "t1",
        "order": 1,
        "title": "A",
        "progress_status": written["topics.$.progress_status"],
        "completed_at": written["topics.$.completed_at"],
    }
    assert repo.roadmap_progress({"topics": [stored]})["completed_count"] == 1
    assert repo.active_topic({"topics": [stored]}) is None  # no longer "next"


async def test_stats_count_the_same_written_topic(monkeypatch):
    """learning_stats reads through a projection, so it can drift from
    roadmap_progress independently."""
    col = _patch_db(monkeypatch, _collection())
    col.find_one = AsyncMock(
        return_value={"status": "active", "topics": [{"progress_status": "completed"}]}
    )
    await repo.set_topic_progress(_OID, "t1", "completed", "userA")
    written = col.update_one.await_args_list[0].args[1]["$set"]

    _stats_db(
        monkeypatch,
        [
            {
                "status": "active",
                "topics": [
                    {
                        "progress_status": written["topics.$.progress_status"],
                        "completed_at": written["topics.$.completed_at"],
                    }
                ],
            }
        ],
    )
    stats = await repo.learning_stats("userA")
    assert stats["topics"]["completed"] == 1
    assert stats["topics"]["percent"] == 100
    # Completing something today must move the streak off zero, or the counter
    # reads as broken however correct the totals are.
    assert stats["streak_days"] == 1
    assert stats["completed_this_week"] == 1


async def test_reopening_a_topic_clears_its_completion_stamp(monkeypatch):
    col = _patch_db(monkeypatch, _collection())
    col.find_one = AsyncMock(
        return_value={"status": "completed", "topics": [{"progress_status": "not_started"}]}
    )
    await repo.set_topic_progress(_OID, "t1", "not_started", "userA")

    written = col.update_one.await_args_list[0].args[1]["$set"]
    assert written["topics.$.completed_at"] is None
    # …and the roadmap comes back out of "completed".
    rolled = col.update_one.await_args_list[1].args[1]["$set"]
    assert rolled["status"] == "active"


async def test_a_freshly_generated_roadmap_starts_at_zero_not_complete():
    """materialize_roadmap is what every new roadmap goes through; if it emitted
    anything other than not_started the tracker would be wrong from birth."""
    roadmap = repo.materialize_roadmap(_draft(_topic(1, "A"), _topic(2, "B")))
    stored = {"topics": [t.model_dump() for t in roadmap.topics]}
    progress = repo.roadmap_progress(stored)
    assert progress == {
        "next_topic": "A",
        "next_topic_id": roadmap.topics[0].id,
        "completed_count": 0,
        "remaining": 2,
        "total": 2,
        "percent": 0,
    }


# --------------------------------------------------------------------------- #
# digests: acknowledgement and the catch-up queue
# --------------------------------------------------------------------------- #
def repo_coverage(covered: bool, missing=()):
    from app.agents.learning_tracker.state import CoverageOutput

    return CoverageOutput(covered=covered, missing=list(missing))


def _quiz_output(n: int = 2):
    from app.agents.learning_tracker.state import Question, QuizOutput

    return QuizOutput(
        quiz=[Question(question=f"q{i}", options=["a", "b"], answer=0) for i in range(n)]
    )


def _fake_tips(monkeypatch, trig, bullets=("a tip",)):
    """Replace the tips chain with a Runnable so no model is called."""
    from langchain_core.runnables import RunnableLambda

    from app.agents.learning_tracker.state import TopicTipsOutput

    fake = MagicMock()
    fake.with_structured_output.return_value = RunnableLambda(
        lambda _: TopicTipsOutput(bullets=list(bullets))
    )
    monkeypatch.setattr(trig, "llm", fake)


def _digest_db(monkeypatch, digests=(), roadmaps=(), unread=None):
    digest_col, roadmap_col = _collection(), _collection()
    digest_col.find = MagicMock(
        return_value=MagicMock(
            sort=lambda *_: MagicMock(
                limit=lambda *_: MagicMock(to_list=AsyncMock(return_value=list(digests)))
            )
        )
    )
    digest_col.find_one = AsyncMock(return_value=unread)
    roadmap_col.find = MagicMock(
        return_value=MagicMock(to_list=AsyncMock(return_value=list(roadmaps)))
    )
    monkeypatch.setattr(
        repo, "get_db", lambda: {repo.DIGESTS: digest_col, "roadmaps": roadmap_col}
    )
    return digest_col


async def test_unread_means_not_marked_so_older_digests_still_count(monkeypatch):
    """Digests written before `status` existed have never been acknowledged
    either, so they belong in the unread queue rather than being filtered out."""
    col = _digest_db(monkeypatch)
    await repo.list_digests("userA", status="unread")
    assert col.find.call_args.args[0]["status"] == {"$ne": "marked"}


async def test_the_catch_up_queue_only_spans_active_roadmaps(monkeypatch):
    col = _digest_db(monkeypatch, roadmaps=[{"_id": "r1"}, {"_id": "r2"}])
    await repo.list_digests("userA", status="unread", active_only=True)
    assert col.find.call_args.args[0]["roadmapId"] == {"$in": ["r1", "r2"]}


async def test_no_active_roadmaps_means_an_empty_queue_not_everything(monkeypatch):
    _digest_db(monkeypatch, digests=[{"_id": _OID}], roadmaps=[])
    assert await repo.list_digests("userA", status="unread", active_only=True) == []


async def test_digests_default_to_unread_when_listed(monkeypatch):
    _digest_db(
        monkeypatch,
        digests=[{"_id": _OID, "roadmapId": "r1"}],
        roadmaps=[{"_id": "r1", "title": "Rust", "topics": []}],
    )
    d = (await repo.list_digests("userA"))[0]
    assert d["status"] == "unread"
    assert d["roadmapTitle"] == "Rust"


async def test_marking_a_digest_is_user_scoped_and_stamps_when(monkeypatch):
    col = _digest_db(monkeypatch)
    assert await repo.mark_digest(_OID, "userA") is True

    filt, update = col.update_one.await_args.args
    assert filt["user_id"] == "userA"
    assert update["$set"]["status"] == "marked"
    assert update["$set"]["updatedAt"]


def _started(**over) -> dict:
    return {"id": "t1", "title": "A", "order": 1, "progress_status": "in_progress", **over}


def _digest_gen(monkeypatch, unread_count=0, prior=()):
    """Stub build_digest's collaborators, and record whether it spent anything."""
    import app.agents.learning_tracker.triggers as trig

    monkeypatch.setattr(trig, "unread_digest_count", AsyncMock(return_value=unread_count))
    monkeypatch.setattr(trig, "topic_digests", AsyncMock(return_value=list(prior)))
    search = MagicMock()
    search.ainvoke = AsyncMock(return_value={"results": []})
    monkeypatch.setattr(trig, "tavily_search_tool", search)
    # build_digest writes through its own module's get_db.
    col = _collection()
    monkeypatch.setattr(trig, "get_db", lambda: {trig.DIGESTS: col, "quizzes": col})
    monkeypatch.setattr(trig, "set_topic_progress", AsyncMock(return_value=True))
    return trig, search


# ── when the next digest lands ──────────────────────────────────────────────
def _at(iso: str):
    from datetime import datetime

    return datetime.fromisoformat(iso)


def test_next_run_is_today_when_the_hour_is_still_ahead():
    from app.agents.trigger_store import next_run_at

    trig = {"enabled": True, "schedule_hour": 9, "timezone": "UTC"}
    assert next_run_at(trig, _at("2026-08-02T06:00:00+00:00")).startswith("2026-08-02T09:00")


def test_next_run_rolls_to_tomorrow_once_the_hour_has_passed():
    from app.agents.trigger_store import next_run_at

    trig = {"enabled": True, "schedule_hour": 9, "timezone": "UTC"}
    assert next_run_at(trig, _at("2026-08-02T10:00:00+00:00")).startswith("2026-08-03T09:00")


def test_next_run_skips_today_when_it_already_fired():
    """Mirrors is_due's same-day guard — otherwise the countdown would point at
    an hour that has already been used up."""
    from app.agents.trigger_store import next_run_at

    trig = {
        "enabled": True,
        "schedule_hour": 9,
        "timezone": "UTC",
        "last_run_at": "2026-08-02T09:00:00+00:00",
    }
    assert next_run_at(trig, _at("2026-08-02T07:00:00+00:00")).startswith("2026-08-03T09:00")


def test_next_run_respects_the_learners_timezone():
    from app.agents.trigger_store import next_run_at

    trig = {"enabled": True, "schedule_hour": 9, "timezone": "Asia/Kolkata"}
    # 09:00 IST is 03:30 UTC.
    assert next_run_at(trig, _at("2026-08-02T00:00:00+00:00")).startswith("2026-08-02T03:30")


def test_a_disabled_trigger_has_no_next_run():
    from app.agents.trigger_store import next_run_at

    assert next_run_at({"enabled": False, "schedule_hour": 9}) is None


def test_next_run_honours_a_weekly_schedule():
    from app.agents.trigger_store import next_run_at

    # 2026-08-02 is a Sunday; dow=2 is Wednesday.
    trig = {"enabled": True, "schedule_hour": 9, "timezone": "UTC", "schedule_dow": 2}
    assert next_run_at(trig, _at("2026-08-02T10:00:00+00:00")).startswith("2026-08-05T09:00")


# ── what the home screen shows when the queue is clear ──────────────────────
def _focus_db(monkeypatch, roadmap=None, trigger=None, unread=0):
    col = _collection()
    col.find_one = AsyncMock(side_effect=[roadmap, trigger])
    col.count_documents = AsyncMock(return_value=unread)
    monkeypatch.setattr(
        repo, "get_db", lambda: {"roadmaps": col, "triggers": col, repo.DIGESTS: col}
    )
    monkeypatch.setattr(repo, "resolve_roadmap_id", AsyncMock(return_value=_OID))
    return col


_ENABLED_TRIGGER = {"enabled": True, "schedule_hour": 9, "timezone": "UTC"}


async def test_focus_names_the_roadmap_and_the_topic_underway(monkeypatch):
    _focus_db(
        monkeypatch,
        roadmap={
            "_id": _OID,
            "title": "Rust",
            "topics": [
                {"id": "t1", "order": 1, "title": "Ownership", "progress_status": "in_progress"},
                {"id": "t2", "order": 2, "title": "Traits"},
            ],
        },
        trigger=_ENABLED_TRIGGER,
    )
    focus = await repo.learning_focus("userA")

    assert focus["roadmapTitle"] == "Rust"
    assert focus["topic"]["title"] == "Ownership"
    assert focus["can_generate"] is True
    assert focus["next_at"]
    assert focus["blocked_reason"] is None


async def test_focus_explains_a_full_backlog_rather_than_offering_more(monkeypatch):
    from app.core.config import DIGEST_MAX_UNREAD

    _focus_db(
        monkeypatch,
        roadmap={
            "_id": _OID,
            "title": "Rust",
            "topics": [{"id": "t1", "order": 1, "progress_status": "in_progress"}],
        },
        trigger=_ENABLED_TRIGGER,
        unread=DIGEST_MAX_UNREAD,
    )
    focus = await repo.learning_focus("userA")
    assert focus["blocked_reason"] == "cap_reached"
    assert focus["can_generate"] is False


async def test_focus_points_at_the_checkpoint_when_a_topic_is_fully_taught(monkeypatch):
    _focus_db(
        monkeypatch,
        roadmap={
            "_id": _OID,
            "title": "Rust",
            "topics": [
                {"id": "t1", "order": 1, "title": "Ownership", "progress_status": "needs_review"}
            ],
        },
        trigger=_ENABLED_TRIGGER,
    )
    focus = await repo.learning_focus("userA")

    assert focus["blocked_reason"] == "needs_review"
    assert focus["topic"]["title"] == "Ownership"  # still worth naming
    assert focus["can_generate"] is False


async def test_focus_reports_a_finished_roadmap(monkeypatch):
    _focus_db(
        monkeypatch,
        roadmap={
            "_id": _OID,
            "title": "Rust",
            "topics": [{"id": "t1", "order": 1, "progress_status": "completed"}],
        },
        trigger=_ENABLED_TRIGGER,
    )
    assert (await repo.learning_focus("userA"))["blocked_reason"] == "roadmap_complete"


async def test_focus_flags_digests_being_switched_off(monkeypatch):
    _focus_db(
        monkeypatch,
        roadmap={
            "_id": _OID,
            "title": "Rust",
            "topics": [{"id": "t1", "order": 1, "progress_status": "in_progress"}],
        },
        trigger={"enabled": False, "schedule_hour": 9, "timezone": "UTC"},
    )
    focus = await repo.learning_focus("userA")

    assert focus["blocked_reason"] == "digests_off"
    assert focus["next_at"] is None
    # Still pullable by hand — the schedule is off, not the feature.
    assert focus["can_generate"] is True


async def test_focus_on_no_roadmap_says_so(monkeypatch):
    monkeypatch.setattr(repo, "resolve_roadmap_id", AsyncMock(return_value=None))
    monkeypatch.setattr(repo, "fetch_roadmap", AsyncMock(return_value=None))
    focus = await repo.learning_focus("userA")
    assert focus["blocked_reason"] == "no_roadmap"
    assert focus["roadmapTitle"] is None


# ── exactly one topic underway ──────────────────────────────────────────────
def test_a_new_roadmap_opens_with_its_first_topic_underway():
    """So the learner has something in flight, and digests, without a separate step."""
    roadmap = repo.materialize_roadmap(
        _draft(_topic(1, "A"), _topic(2, "B"), _topic(3, "C"))
    )
    assert [t.progress_status for t in roadmap.topics] == [
        "in_progress",
        "not_started",
        "not_started",
    ]


def _nodes(*statuses):
    from app.agents.learning_tracker.state import TopicNode

    return [
        TopicNode(id=f"t{i}", order=i + 1, title=f"T{i}", description="", progress_status=s)
        for i, s in enumerate(statuses)
    ]


def test_a_second_in_progress_topic_is_demoted():
    topics = _nodes("in_progress", "in_progress", "not_started")
    repo.enforce_single_in_progress(topics)
    assert [t.progress_status for t in topics] == [
        "in_progress",
        "not_started",
        "not_started",
    ]


def test_the_slot_passes_to_the_first_unfinished_topic():
    topics = _nodes("completed", "not_started", "not_started")
    repo.enforce_single_in_progress(topics)
    assert topics[1].progress_status == "in_progress"


def test_a_topic_awaiting_its_checkpoint_holds_the_slot():
    """Otherwise the learner accumulates half-finished topics instead of
    closing one out."""
    topics = _nodes("needs_review", "not_started")
    repo.enforce_single_in_progress(topics)
    assert [t.progress_status for t in topics] == ["needs_review", "not_started"]


def test_a_finished_roadmap_starts_nothing():
    topics = _nodes("completed", "skipped")
    repo.enforce_single_in_progress(topics)
    assert [t.progress_status for t in topics] == ["completed", "skipped"]


def test_an_edit_that_drops_the_started_topic_still_leaves_one_underway():
    existing = {
        "status": "active",
        "topics": [{"id": "gone", "order": 1, "title": "Gone", "progress_status": "in_progress"}],
    }
    merged = repo.merge_roadmap(existing, _draft(_topic(1, "Fresh"), _topic(2, "Later")))
    assert [t.progress_status for t in merged.topics] == ["in_progress", "not_started"]


async def test_starting_a_topic_swaps_the_slot_in_one_write(monkeypatch):
    col = _patch_db(monkeypatch, _collection())
    assert await repo.start_topic(_OID, "t2", "userA") is True

    _, update = col.update_one.await_args.args
    kwargs = col.update_one.await_args.kwargs
    assert update["$set"]["topics.$[target].progress_status"] == "in_progress"
    assert update["$set"]["topics.$[other].progress_status"] == "not_started"
    # Both halves in one update, so the roadmap is never briefly two-or-none.
    assert kwargs["array_filters"][0]["other.id"] == {"$ne": "t2"}


async def test_set_topic_progress_routes_starting_through_the_swap(monkeypatch):
    col = _patch_db(monkeypatch, _collection())
    col.find_one = AsyncMock(return_value={"status": "active", "topics": []})

    await repo.set_topic_progress(_OID, "t1", "in_progress", "userA")
    assert "array_filters" in col.update_one.await_args_list[0].kwargs


async def test_passing_a_checkpoint_hands_the_slot_to_the_next_topic(monkeypatch):
    col = _collection()
    col.find_one = AsyncMock(
        return_value={
            "_id": _OID,
            "status": "active",
            "topics": [
                {"id": "t1", "order": 1, "title": "A", "progress_status": "in_progress"},
                {"id": "t2", "order": 2, "title": "B", "progress_status": "not_started"},
            ],
        }
    )
    _patch_db(monkeypatch, col)

    out = await repo.apply_checkpoint(_OID, "t1", "userA", 100)

    assert out["passed"] is True
    assert out["advanced_to"] == {"topicId": "t2", "title": "B"}


async def test_passing_a_review_does_not_move_the_slot(monkeypatch):
    """A review of an already-finished topic isn't progress through the roadmap."""
    col = _collection()
    col.find_one = AsyncMock(
        return_value={
            "_id": _OID,
            "status": "active",
            "topics": [
                {"id": "t1", "order": 1, "progress_status": "completed", "completed_at": "x"},
                {"id": "t2", "order": 2, "progress_status": "not_started"},
            ],
        }
    )
    _patch_db(monkeypatch, col)

    out = await repo.apply_checkpoint(_OID, "t1", "userA", 100)
    assert out["was_review"] is True
    assert out["advanced_to"] is None


async def test_full_coverage_sends_the_topic_to_needs_review(monkeypatch):
    """The drip-feed is done; the checkpoint is what completes it now."""
    trig, _ = _digest_gen(monkeypatch)
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(True)))
    _fake_tips(monkeypatch, trig)

    await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert trig.set_topic_progress.await_args.args[2] == "needs_review"


async def test_partial_coverage_leaves_the_topic_underway(monkeypatch):
    trig, _ = _digest_gen(monkeypatch)
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    _fake_tips(monkeypatch, trig)

    await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)
    trig.set_topic_progress.assert_not_awaited()


# ── which topics get digests ────────────────────────────────────────────────
@pytest.mark.parametrize("status", ["not_started", "completed", "skipped", "needs_review"])
async def test_only_a_started_topic_gets_digests(monkeypatch, status):
    """A topic nobody has picked up shouldn't be filling an inbox."""
    trig, search = _digest_gen(monkeypatch)
    out = await trig.build_digest(
        "userA", {"_id": _OID, "topics": [_started(progress_status=status)]}
    )
    assert out is None
    search.ainvoke.assert_not_awaited()


async def test_digests_stop_once_three_are_waiting(monkeypatch):
    """Otherwise a learner who ignores them for a week returns to seven."""
    from app.core.config import DIGEST_MAX_UNREAD

    trig, search = _digest_gen(monkeypatch, unread_count=DIGEST_MAX_UNREAD)
    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]})

    assert out is None
    search.ainvoke.assert_not_awaited()  # no search, no LLM call, no spend


async def test_digests_resume_once_the_backlog_drops(monkeypatch):
    from app.core.config import DIGEST_MAX_UNREAD

    trig, search = _digest_gen(monkeypatch, unread_count=DIGEST_MAX_UNREAD - 1)
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    _fake_tips(monkeypatch, trig)

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)
    assert out is not None
    assert out["sequence"] == 1


async def test_a_backlog_read_failure_holds_off_rather_than_piling_on(monkeypatch):
    """unread_digest_count fails closed — an unknown backlog is treated as full."""
    col = _digest_db(monkeypatch)
    col.count_documents = AsyncMock(side_effect=RuntimeError("mongo down"))
    from app.core.config import DIGEST_MAX_UNREAD

    assert await repo.unread_digest_count("userA", "r1", "t1") == DIGEST_MAX_UNREAD


def test_in_progress_topic_picks_the_started_one_in_order():
    roadmap = {
        "topics": [
            {"id": "a", "order": 1, "progress_status": "completed"},
            {"id": "b", "order": 2, "progress_status": "in_progress"},
            {"id": "c", "order": 3, "progress_status": "in_progress"},
            {"id": "d", "order": 4, "progress_status": "not_started"},
        ]
    }
    assert repo.in_progress_topic(roadmap)["id"] == "b"
    assert repo.in_progress_topic({"topics": []}) is None


# ── the recall check on later digests ───────────────────────────────────────
async def test_the_first_digest_carries_no_recall_check(monkeypatch):
    trig, _ = _digest_gen(monkeypatch)
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    _fake_tips(monkeypatch, trig)

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)
    assert out["sequence"] == 1
    assert out["quizId"] is None  # nothing to recall yet


async def test_later_digests_are_quizzed_on_the_earlier_ones_only(monkeypatch):
    """Quizzing on the digest they haven't acknowledged yet would make marking
    it impossible."""
    trig, _ = _digest_gen(
        monkeypatch, prior=[{"bullets": ["earlier point"]}]
    )
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    _fake_tips(monkeypatch, trig, bullets=["brand new point"])
    seen = {}

    async def _quiz(topic_title, bullets):
        seen["bullets"] = bullets
        return _quiz_output()

    monkeypatch.setattr(trig, "build_digest_quiz", _quiz)

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert out["sequence"] == 2
    assert out["quizId"]
    assert seen["bullets"] == ["earlier point"]
    assert "brand new point" not in seen["bullets"]


async def test_a_failed_quiz_generation_still_ships_the_digest(monkeypatch):
    """A digest nobody can acknowledge is worse than one without a recall check."""
    trig, _ = _digest_gen(monkeypatch, prior=[{"bullets": ["earlier"]}])
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    _fake_tips(monkeypatch, trig)
    monkeypatch.setattr(trig, "build_digest_quiz", AsyncMock(side_effect=RuntimeError("boom")))

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)
    assert out is not None and out["quizId"] is None


# ── coverage ────────────────────────────────────────────────────────────────
async def test_coverage_is_recorded_on_every_digest(monkeypatch):
    trig, _ = _digest_gen(monkeypatch)
    monkeypatch.setattr(
        trig, "check_coverage", AsyncMock(return_value=repo_coverage(True, ["x"]))
    )
    _fake_tips(monkeypatch, trig)

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)
    assert out["coverage_complete"] is True
    assert out["missing_outcomes"] == ["x"]


async def test_a_topic_with_no_outcomes_is_never_declared_covered():
    """Nothing concrete to measure against, so don't end the drip-feed on a guess."""
    from app.agents.learning_tracker.service import check_coverage

    out = await check_coverage({"title": "A", "learning_outcomes": []}, ["a bullet"])
    assert out.covered is False


async def test_a_coverage_failure_does_not_block_the_digest(monkeypatch):
    trig, _ = _digest_gen(monkeypatch)
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(side_effect=RuntimeError("boom")))
    _fake_tips(monkeypatch, trig)

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)
    assert out is not None and out["coverage_complete"] is False


# --------------------------------------------------------------------------- #
# notes
# --------------------------------------------------------------------------- #
def _notes_db(monkeypatch, notes=(), roadmaps=()):
    notes_col, roadmap_col = _collection(), _collection()
    notes_col.find = MagicMock(
        return_value=MagicMock(
            sort=lambda *_: MagicMock(
                skip=lambda *_: MagicMock(
                    limit=lambda *_: MagicMock(to_list=AsyncMock(return_value=list(notes)))
                )
            )
        )
    )
    notes_col.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    roadmap_col.find = MagicMock(
        return_value=MagicMock(to_list=AsyncMock(return_value=list(roadmaps)))
    )
    monkeypatch.setattr(
        repo, "get_db", lambda: {repo.NOTES: notes_col, "roadmaps": roadmap_col}
    )
    return notes_col


async def test_a_note_records_who_and_what_it_belongs_to(monkeypatch):
    col = _notes_db(monkeypatch)
    note = await repo.create_note("userA", "r1", "t1", "snippet", "let x = 5;")

    doc = col.insert_one.await_args.args[0]
    assert doc["user_id"] == "userA"
    assert (doc["kind"], doc["body"]) == ("snippet", "let x = 5;")
    assert doc["resolved"] is False
    assert note["_id"]


async def test_notes_are_decorated_with_where_they_came_from(monkeypatch):
    """The consolidated view needs to say which topic a note is about, and titles
    are resolved at read time so a renamed topic doesn't leave a stale label."""
    _notes_db(
        monkeypatch,
        notes=[{"_id": _OID, "roadmapId": "r1", "topicId": "t1", "body": "n"}],
        roadmaps=[{"_id": "r1", "title": "Rust", "topics": [{"id": "t1", "title": "Ownership"}]}],
    )
    note = (await repo.list_notes("userA"))[0]
    assert note["roadmapTitle"] == "Rust"
    assert note["topicTitle"] == "Ownership"


async def test_a_note_survives_the_topic_it_was_written_against(monkeypatch):
    """A roadmap edit can drop a topic. Losing the learner's writing with it
    would be far worse than showing it without a topic label."""
    _notes_db(
        monkeypatch,
        notes=[{"_id": _OID, "roadmapId": "r1", "topicId": "gone", "body": "still here"}],
        roadmaps=[{"_id": "r1", "title": "Rust", "topics": []}],
    )
    notes = await repo.list_notes("userA")
    assert len(notes) == 1
    assert notes[0]["body"] == "still here"
    assert notes[0]["topicTitle"] is None


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, {"user_id": "userA"}),
        ({"roadmapId": "r1"}, {"user_id": "userA", "roadmapId": "r1"}),
        ({"topicId": "t1"}, {"user_id": "userA", "topicId": "t1"}),
        ({"kind": "question"}, {"user_id": "userA", "kind": "question"}),
    ],
)
async def test_note_queries_are_always_scoped_to_the_caller(monkeypatch, kwargs, expected):
    col = _notes_db(monkeypatch)
    await repo.list_notes("userA", **kwargs)
    assert col.find.call_args.args[0] == expected


@pytest.mark.parametrize(
    "call",
    [
        lambda: repo.update_note(_OID, "userA", {"resolved": True}),
        lambda: repo.delete_note(_OID, "userA"),
    ],
)
async def test_editing_and_deleting_a_note_is_user_scoped(monkeypatch, call):
    col = _notes_db(monkeypatch)
    await call()
    filt = (col.update_one.await_args or col.delete_one.await_args).args[0]
    assert filt["user_id"] == "userA"


async def test_update_note_with_nothing_to_change_is_a_no_op(monkeypatch):
    col = _notes_db(monkeypatch)
    assert await repo.update_note(_OID, "userA", {}) is False
    col.update_one.assert_not_awaited()


# --------------------------------------------------------------------------- #
# personalization made visible
# --------------------------------------------------------------------------- #
_PACED = {"availability": {"minutes_per_day": 60, "days_per_week": 7}}


def _paced_roadmap(*minutes: int, done: int = 0) -> dict:
    return {
        "topics": [
            {
                "id": f"t{i}",
                "estimated_minutes": m,
                "progress_status": "completed" if i < done else "not_started",
            }
            for i, m in enumerate(minutes)
        ]
    }


def test_forecast_turns_remaining_minutes_into_a_date():
    f = repo.completion_forecast(_paced_roadmap(60, 60, 60), _PACED)
    assert f["remaining_minutes"] == 180
    assert f["study_days"] == 3
    assert f["calendar_days"] == 3  # 7 days a week, so calendar == study days


def test_forecast_stretches_over_the_weeks_a_part_time_pace_implies():
    """Six study days at three days a week is a fortnight, not six days."""
    f = repo.completion_forecast(
        _paced_roadmap(*[60] * 6),
        {"availability": {"minutes_per_day": 60, "days_per_week": 3}},
    )
    assert f["study_days"] == 6
    assert f["calendar_days"] == 14


def test_forecast_only_counts_what_is_left():
    f = repo.completion_forecast(_paced_roadmap(60, 60, 60, done=2), _PACED)
    assert f["remaining_minutes"] == 60


def test_no_pace_on_file_means_no_forecast_rather_than_a_made_up_one():
    assert repo.completion_forecast(_paced_roadmap(60), {}) is None
    assert repo.completion_forecast(_paced_roadmap(60), {"availability": {}}) is None
    # Nothing left to do is also not a forecast.
    assert repo.completion_forecast(_paced_roadmap(60, done=1), _PACED) is None


def test_forecast_says_whether_a_stated_deadline_is_reachable():
    soon = {"availability": {"minutes_per_day": 60, "days_per_week": 7, "deadline": "2020-01-01"}}
    assert repo.completion_forecast(_paced_roadmap(60, 60), soon)["on_track"] is False

    far = {"availability": {"minutes_per_day": 60, "days_per_week": 7, "deadline": "2099-01-01"}}
    assert repo.completion_forecast(_paced_roadmap(60, 60), far)["on_track"] is True


def test_snapshot_keeps_only_what_shapes_a_roadmap():
    snap = repo.profile_snapshot(
        {
            "skill_level": "beginner",
            "goals": ["ship a service"],
            "preferred_quiz_difficulty": "hard",  # not a roadmap input
            "known_topics": [],  # blank, so not recorded
            "onboarded": True,
        }
    )
    assert snap == {"skill_level": "beginner", "goals": ["ship a service"]}


def test_drift_reports_the_inputs_that_changed():
    snapshot = {"skill_level": "beginner", "goals": ["a"]}
    assert repo.profile_drift(snapshot, {"skill_level": "beginner", "goals": ["a"]}) == []
    assert repo.profile_drift(snapshot, {"skill_level": "advanced", "goals": ["a"]}) == [
        "skill_level"
    ]
    # A field filled in after the fact is drift too — that's new information the
    # roadmap was never built with.
    assert repo.profile_drift(snapshot, {**snapshot, "availability": {"minutes_per_day": 30}}) == [
        "availability"
    ]
    # Changing something outside the personalization set is not drift.
    assert repo.profile_drift(snapshot, {**snapshot, "preferred_quiz_difficulty": "easy"}) == []


def test_a_roadmap_with_no_snapshot_is_never_called_stale():
    """Pre-existing roadmaps have no record of what built them, so claiming the
    profile has moved on would be a guess."""
    assert repo.profile_drift(None, {"skill_level": "advanced"}) == []


def test_generated_roadmaps_record_what_personalized_them():
    memory = {"skill_level": "beginner", "preferred_quiz_difficulty": "hard"}
    roadmap = repo.materialize_roadmap(
        _draft(_topic(1, "A")), personalization=repo.profile_snapshot(memory)
    )
    assert roadmap.personalization == {"skill_level": "beginner"}


def test_editing_a_roadmap_re_personalizes_it():
    existing = {"status": "active", "topics": [], "personalization": {"skill_level": "beginner"}}
    merged = repo.merge_roadmap(existing, _draft(_topic(1, "A")), {"skill_level": "advanced"})
    assert merged.personalization == {"skill_level": "advanced"}

    # …but an edit made without a profile to hand keeps the original record
    # rather than blanking it.
    kept = repo.merge_roadmap(existing, _draft(_topic(1, "A")))
    assert kept.personalization == {"skill_level": "beginner"}


# --------------------------------------------------------------------------- #
# checkpoints & spaced repetition
# --------------------------------------------------------------------------- #
def _roadmap_with(topic: dict) -> dict:
    return {"_id": _OID, "status": "active", "title": "Rust", "topics": [topic]}


def _checkpoint_db(monkeypatch, topic: dict):
    """Wire fetch_roadmap + the positional update for apply_checkpoint."""
    col = _collection()
    col.find_one = AsyncMock(return_value=_roadmap_with(topic))
    _patch_db(monkeypatch, col)
    return col


def _written(col) -> dict:
    return col.update_one.await_args_list[0].args[1]["$set"]


def test_review_ladder_expands_with_each_consecutive_pass():
    from datetime import datetime, timezone

    def days_out(count: int) -> int:
        due = datetime.fromisoformat(repo.next_review_at(count))
        return round((due - datetime.now(timezone.utc)).total_seconds() / 86400)

    assert [days_out(n) for n in (1, 2, 3, 4, 5)] == list(repo.REVIEW_LADDER_DAYS)
    # A failure (count reset to 0) comes back tomorrow, not in five weeks.
    assert days_out(0) == repo.REVIEW_LADDER_DAYS[0]
    # The ladder tops out rather than running off the end.
    assert days_out(99) == repo.REVIEW_LADDER_DAYS[-1]


async def test_passing_a_checkpoint_is_what_completes_a_topic(monkeypatch):
    col = _checkpoint_db(monkeypatch, {"id": "t1", "progress_status": "not_started"})

    out = await repo.apply_checkpoint(_OID, "t1", "userA", 100)

    assert out["passed"] is True
    assert out["progress_status"] == "completed"
    assert out["review_count"] == 1
    assert out["was_review"] is False
    written = _written(col)
    assert written["topics.$.mastery_score"] == 100
    assert written["topics.$.completed_at"]
    assert written["topics.$.next_review_at"]


async def test_failing_a_first_attempt_does_not_complete_the_topic(monkeypatch):
    col = _checkpoint_db(monkeypatch, {"id": "t1", "progress_status": "not_started"})

    out = await repo.apply_checkpoint(_OID, "t1", "userA", 25)

    assert out["passed"] is False
    assert out["progress_status"] == "in_progress"
    assert _written(col).get("topics.$.completed_at") is None
    # Still scheduled — a topic you struggled with should come back soonest.
    assert out["review_count"] == 0


async def test_failing_a_review_does_not_take_completion_away(monkeypatch):
    """Clawing back progress for an honest attempt would punish the exact
    behaviour spaced repetition exists to encourage."""
    col = _checkpoint_db(
        monkeypatch,
        {
            "id": "t1",
            "progress_status": "completed",
            "completed_at": "2026-01-01T00:00:00+00:00",
            "review_count": 3,
        },
    )

    out = await repo.apply_checkpoint(_OID, "t1", "userA", 30)

    assert out["passed"] is False
    assert out["progress_status"] == "completed"  # still done
    assert out["was_review"] is True
    assert out["review_count"] == 0  # but back to the front of the ladder
    # The original completion date is not overwritten by the failed review.
    assert "topics.$.completed_at" not in _written(col)


async def test_passing_a_review_pushes_the_next_one_further_out(monkeypatch):
    _checkpoint_db(
        monkeypatch,
        {"id": "t1", "progress_status": "completed", "completed_at": "x", "review_count": 2},
    )
    out = await repo.apply_checkpoint(_OID, "t1", "userA", 90)
    assert out["review_count"] == 3
    assert out["was_review"] is True


async def test_a_checkpoint_for_an_unknown_topic_changes_nothing(monkeypatch):
    col = _checkpoint_db(monkeypatch, {"id": "t1", "progress_status": "not_started"})
    assert await repo.apply_checkpoint(_OID, "nope", "userA", 100) is None
    col.update_one.assert_not_awaited()


async def test_due_reviews_only_surfaces_completed_topics_past_their_date(monkeypatch):
    past, future = "2020-01-01T00:00:00+00:00", "2099-01-01T00:00:00+00:00"
    col = _collection()
    col.find = MagicMock(
        return_value=MagicMock(
            to_list=AsyncMock(
                return_value=[
                    {
                        "_id": _OID,
                        "title": "Rust",
                        "topics": [
                            {"id": "due", "title": "Ownership", "progress_status": "completed", "next_review_at": past},
                            {"id": "later", "title": "Traits", "progress_status": "completed", "next_review_at": future},
                            # Never completed, so it isn't a review candidate even
                            # if something stamped a date on it.
                            {"id": "unstarted", "title": "Macros", "progress_status": "not_started", "next_review_at": past},
                        ],
                    }
                ]
            )
        )
    )
    _patch_db(monkeypatch, col)

    due = await repo.due_reviews("userA")
    assert [d["topicId"] for d in due] == ["due"]
    assert due[0]["roadmapTitle"] == "Rust"


async def test_stats_report_how_many_reviews_are_due(monkeypatch):
    past, future = "2020-01-01T00:00:00+00:00", "2099-01-01T00:00:00+00:00"
    _stats_db(
        monkeypatch,
        [
            {
                "status": "active",
                "topics": [
                    {"progress_status": "completed", "completed_at": f"{_day(0)}T09:00:00+00:00", "next_review_at": past},
                    {"progress_status": "completed", "completed_at": f"{_day(1)}T09:00:00+00:00", "next_review_at": future},
                ],
            }
        ],
    )
    assert (await repo.learning_stats("userA"))["reviews_due"] == 1


async def test_progress_route_refuses_to_complete_without_a_checkpoint(monkeypatch):
    """The gate is the feature: completion has to be earned, not asserted."""
    from fastapi import HTTPException

    wrote = AsyncMock()
    monkeypatch.setattr(lt_router, "set_topic_progress", wrote)

    body = lt_router.ProgressUpdate(roadmapId=_OID, topicId="t1", status="completed")
    with pytest.raises(HTTPException) as err:
        await lt_router.update_progress(body, {"uid": "userA"})

    assert err.value.status_code == 409
    wrote.assert_not_awaited()

    # The legacy boolean shape defaulted to completing, so it has to be caught too.
    with pytest.raises(HTTPException):
        await lt_router.update_progress(
            lt_router.ProgressUpdate(roadmapId=_OID, topicId="t1", covered=True),
            {"uid": "userA"},
        )


async def test_chat_cannot_complete_a_topic_either(monkeypatch):
    """The HTTP gate is worthless if 'I finished pointers' in chat walks around
    it — the agent path writes to the repository directly."""
    roadmap = {
        "_id": _OID,
        "topics": [{"id": "t1", "title": "Pointers", "progress_status": "not_started"}],
    }
    monkeypatch.setattr(lt, "fetch_roadmap", AsyncMock(return_value=roadmap))
    wrote = AsyncMock(return_value=True)
    monkeypatch.setattr(lt, "set_topic_progress", wrote)

    # The node pipes a prompt into the model, so the stand-in has to be a real
    # Runnable rather than a bare mock.
    from langchain_core.runnables import RunnableLambda

    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = RunnableLambda(
        lambda _: MagicMock(topicId="t1")
    )
    monkeypatch.setattr(lt, "fast_llm", fake_llm)

    out = await lt.progress_agent(
        {
            "user_id": "userA",
            "intent": "update_progress",
            "roadmapId": _OID,
            "query": "I finished pointers",
        }
    )

    assert out["log_status"] == "checkpoint_required"
    assert wrote.await_args.args[2] == "in_progress"  # never "completed"


@pytest.mark.parametrize("status", ["in_progress", "skipped", "not_started"])
async def test_progress_route_still_owns_every_other_transition(monkeypatch, status):
    """Only completion is gated. Starting, skipping, and reopening stay direct —
    the gate is on claiming knowledge, not on retracting it."""
    wrote = AsyncMock(return_value=True)
    monkeypatch.setattr(lt_router, "set_topic_progress", wrote)

    out = await lt_router.update_progress(
        lt_router.ProgressUpdate(roadmapId=_OID, topicId="t1", status=status),
        {"uid": "userA"},
    )
    assert out["progress_status"] == status
    assert wrote.await_args.args[2] == status


# --------------------------------------------------------------------------- #
# landing-screen stats
# --------------------------------------------------------------------------- #
def _stats_db(monkeypatch, roadmaps, attempts=()):
    """Stub the two collections learning_stats reads."""
    roadmap_col, quiz_col = MagicMock(), MagicMock()
    roadmap_col.find = MagicMock(
        return_value=MagicMock(to_list=AsyncMock(return_value=list(roadmaps)))
    )
    quiz_col.find = MagicMock(
        return_value=MagicMock(to_list=AsyncMock(return_value=list(attempts)))
    )
    monkeypatch.setattr(
        repo, "get_db", lambda: {"roadmaps": roadmap_col, "quiz_attempts": quiz_col}
    )


def _day(offset: int) -> str:
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc).date() - timedelta(days=offset)).isoformat()


async def test_stats_counts_topics_across_every_roadmap(monkeypatch):
    _stats_db(
        monkeypatch,
        [
            {
                "status": "active",
                "topics": [
                    {"progress_status": "completed", "completed_at": f"{_day(0)}T09:00:00+00:00"},
                    {"progress_status": "not_started"},
                ],
            },
            {"status": "completed", "topics": [{"progress_status": "completed", "completed_at": f"{_day(1)}T09:00:00+00:00"}]},
        ],
        attempts=[{"score": 80}, {"score": 60}],
    )

    stats = await repo.learning_stats("userA")
    assert stats["roadmaps"] == {"total": 2, "active": 1, "completed": 1}
    assert stats["topics"] == {"total": 3, "completed": 2, "percent": 67}
    assert stats["quizzes"] == {"attempts": 2, "average_score": 70}


async def test_streak_counts_consecutive_days_back_from_today(monkeypatch):
    _stats_db(
        monkeypatch,
        [
            {
                "status": "active",
                "topics": [
                    {"progress_status": "completed", "completed_at": f"{_day(d)}T09:00:00+00:00"}
                    for d in (0, 1, 2, 5)  # 3-day run, then a gap
                ],
            }
        ],
    )
    stats = await repo.learning_stats("userA")
    assert stats["streak_days"] == 3
    assert stats["completed_this_week"] == 4


async def test_a_quiet_today_does_not_break_the_streak(monkeypatch):
    """The streak should only break once a full day has been missed — otherwise
    it would read as 0 every morning until the learner studied."""
    _stats_db(
        monkeypatch,
        [
            {
                "status": "active",
                "topics": [
                    {"progress_status": "completed", "completed_at": f"{_day(d)}T09:00:00+00:00"}
                    for d in (1, 2)
                ],
            }
        ],
    )
    assert (await repo.learning_stats("userA"))["streak_days"] == 2


async def test_stats_on_an_empty_account_are_zeros_not_a_crash(monkeypatch):
    _stats_db(monkeypatch, [])
    stats = await repo.learning_stats("userA")
    assert stats["topics"]["percent"] == 0
    assert stats["streak_days"] == 0
    assert stats["quizzes"]["average_score"] == 0


async def test_a_completed_topic_with_no_timestamp_still_counts(monkeypatch):
    """Topics completed before completed_at existed, or via a path that didn't
    stamp it, must not vanish from the totals."""
    _stats_db(monkeypatch, [{"status": "active", "topics": [{"progress_status": "completed"}]}])
    stats = await repo.learning_stats("userA")
    assert stats["topics"]["completed"] == 1
    assert stats["streak_days"] == 0  # undated, so it can't contribute to a run


# --------------------------------------------------------------------------- #
# turn responses — never leak internal state, never bury a second pause
# --------------------------------------------------------------------------- #
import app.routers.learning_tracker as lt_router

# What the graph state actually holds mid-run. `current_user` is the dangerous
# part: the raw state used to be returned verbatim.
_STATE = {
    "intent": "create_roadmap",
    "roadmap_status": "approved",
    "roadmapId": "r1",
    "topic_explaination": "hello",
    "current_user": {
        "uid": "u1",
        "token_version": 3,
        "email_verify_code_hash": "$2b$12$secret",
        "email": "learner@example.com",
    },
    "memory": {"skill_level": "beginner"},
    "query": "learn rust",
    "thread_id": "t1",
}


def test_turn_response_never_carries_account_internals():
    body = lt_router._turn(dict(_STATE), "t1")
    serialized = json.dumps(body)
    for secret in ("token_version", "email_verify_code_hash", "learner@example.com"):
        assert secret not in serialized
    assert "current_user" not in body["result"]
    # The learner's profile is available from GET /memory; it is not turn output.
    assert "memory" not in body["result"]


def test_turn_response_keeps_what_the_client_renders():
    result = lt_router._turn(dict(_STATE), "t1")["result"]
    assert result["intent"] == "create_roadmap"
    assert result["roadmapId"] == "r1"
    # log_status and roadmap drive the update_progress card — the streaming
    # projection used to omit both.
    assert "log_status" in result and "roadmap" in result


def test_a_pause_after_resuming_surfaces_as_a_pause():
    """Answering onboarding runs straight into roadmap approval. That second
    interrupt must come back as a proposal, not as a state dump the client
    renders as raw JSON."""
    interrupt = MagicMock()
    interrupt.value = {"type": "save_roadmap", "approvalId": "a1", "roadmap": {}}
    body = lt_router._turn({**_STATE, "__interrupt__": [interrupt]}, "t1")

    assert body["status"] == "needs_approval"
    assert body["proposal"]["type"] == "save_roadmap"
    assert "result" not in body  # nothing to render as a finished turn yet
    assert "token_version" not in json.dumps(body)


def test_onboarding_pause_is_distinguishable_from_an_approval():
    interrupt = MagicMock()
    interrupt.value = {"type": "onboarding", "questions": [], "skippable": True}
    body = lt_router._turn({"__interrupt__": [interrupt]}, "t1")
    assert body["status"] == "needs_input"


async def test_roadmap_node_ignores_another_skills_pending_approval(monkeypatch):
    """A supervisor thread carries approvals from every skill."""
    seen = {}

    async def _get_pending(thread_id, action_types=None):
        seen["action_types"] = action_types
        return None

    monkeypatch.setattr(lt, "get_pending", _get_pending)
    monkeypatch.setattr(lt, "create_pending", AsyncMock(return_value="ap1"))
    monkeypatch.setattr(lt, "build_roadmap", AsyncMock(return_value=_draft(_topic(1, "A"))))
    monkeypatch.setattr(lt, "interrupt", lambda _payload: "rejected")
    monkeypatch.setattr(lt, "resolve", AsyncMock())

    await lt.roadmap_agent(
        {"user_id": "userA", "thread_id": "t1", "intent": "create_roadmap", "query": "rust"}
    )
    assert seen["action_types"] == ["save_roadmap", "update_roadmap"]
