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

    # A genuinely new topic starts clean with a fresh server-minted id.
    added = merged.topics[1]
    assert added.progress_status == "not_started"
    assert added.id not in {"keep-1", "keep-2"}


def test_merge_falls_back_to_title_when_the_model_drops_the_id():
    merged = repo.merge_roadmap(_STORED, _draft(_topic(1, "  ownership  ")))
    assert merged.topics[0].id == "keep-1"
    assert merged.topics[0].progress_status == "completed"


def test_merge_ignores_an_unknown_existing_id():
    """An id the model invented must not silently bind to anything."""
    merged = repo.merge_roadmap(_STORED, _draft(_topic(1, "Macros", existing_id="nope")))
    assert merged.topics[0].id not in {"keep-1", "keep-2"}
    assert merged.topics[0].progress_status == "not_started"


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
    assert merged.topics[1].progress_status == "not_started"


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
    assert roadmap.topics[0].progress_status == "not_started"
    assert roadmap.topics[0].id and roadmap.topics[0].completed_at is None


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
