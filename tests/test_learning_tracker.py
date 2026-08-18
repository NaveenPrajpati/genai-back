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
from bson import ObjectId

import app.agents.learning_tracker.repository as repo
import app.agents.learning_tracker.workflow as lt
import app.agents.trigger_store as trigger_store
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


def _cursor(docs=()):
    """A cursor double answering every chain the reads use: `.find(...)` straight
    to `to_list`, and through any of sort / skip / limit."""
    cur = MagicMock()
    cur.to_list = AsyncMock(return_value=[dict(d) for d in docs])
    cur.sort.return_value = cur
    cur.limit.return_value = cur
    cur.skip.return_value = cur
    return cur


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


async def test_load_context_resolves_the_active_roadmap(monkeypatch):
    """A bare "what should I study next?" carries no roadmapId."""
    monkeypatch.setattr(lt, "get_profile", AsyncMock(return_value={}))
    monkeypatch.setattr(lt, "resolve_roadmap_id", AsyncMock(return_value="resolved"))
    monkeypatch.setattr(lt, "learner_context", AsyncMock(return_value={}))

    out = await lt.load_context({"user_id": "userA", "roadmapId": None})
    assert out["roadmapId"] == "resolved"


async def test_load_context_carries_the_learner_position(monkeypatch):
    """Every node downstream reads `context`; without it the agent can answer a
    question but cannot know what the learner is in the middle of."""
    monkeypatch.setattr(lt, "get_profile", AsyncMock(return_value={}))
    monkeypatch.setattr(lt, "resolve_roadmap_id", AsyncMock(return_value="r1"))
    monkeypatch.setattr(
        lt, "learner_context", AsyncMock(return_value={"unread_digests": 3})
    )

    out = await lt.load_context({"user_id": "userA", "roadmapId": "r1"})
    assert out["context"] == {"unread_digests": 3}


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
# at most MAX_ACTIVE_ROADMAPS run at once
#
# `active` is what the digest sweep drips on, so every path that can mint an
# active roadmap has to respect the cap — not just the button that says "resume".
# --------------------------------------------------------------------------- #
def _active_db(monkeypatch, active: list[dict]):
    """A learner already running `active` roadmaps."""
    col = _collection()
    col.find.return_value.to_list = AsyncMock(return_value=active)
    _patch_db(monkeypatch, col)
    return col


def _other(n: int) -> list[dict]:
    return [{"_id": ObjectId(), "title": f"Roadmap {i}"} for i in range(n)]


async def test_resuming_past_the_cap_is_refused(monkeypatch):
    col = _active_db(monkeypatch, _other(repo.MAX_ACTIVE_ROADMAPS))

    with pytest.raises(repo.ActiveRoadmapLimit) as excinfo:
        await repo.set_roadmap_status(_OID, "userA", "active")

    # Named, so the client can say which to park rather than "go and look".
    assert len(excinfo.value.active) == repo.MAX_ACTIVE_ROADMAPS
    col.update_one.assert_not_awaited()


async def test_a_free_slot_lets_a_roadmap_resume(monkeypatch):
    col = _active_db(monkeypatch, _other(repo.MAX_ACTIVE_ROADMAPS - 1))
    assert await repo.set_roadmap_status(_OID, "userA", "active") is True
    assert col.update_one.await_args.args[1]["$set"]["status"] == "active"


async def test_reactivating_an_already_active_roadmap_is_not_a_refusal(monkeypatch):
    """It holds one of the slots it would be counted against."""
    others = _other(repo.MAX_ACTIVE_ROADMAPS - 1)
    _active_db(monkeypatch, [{"_id": ObjectId(_OID), "title": "Rust"}, *others])
    assert await repo.set_roadmap_status(_OID, "userA", "active") is True


@pytest.mark.parametrize("status", ["paused", "archived", "completed"])
async def test_parking_a_roadmap_is_never_capped(monkeypatch, status):
    """The cap exists to limit what's running; it must not trap a learner who is
    already over it — otherwise there's no way back under."""
    _active_db(monkeypatch, _other(repo.MAX_ACTIVE_ROADMAPS + 1))
    assert await repo.set_roadmap_status(_OID, "userA", status) is True


async def test_a_new_roadmap_is_parked_rather_than_lost_at_the_cap(monkeypatch):
    """Approving a roadmap while the slots are full still saves it — refusing
    would throw away what the learner just built with the tutor."""
    col = _active_db(monkeypatch, _other(repo.MAX_ACTIVE_ROADMAPS))
    await repo.insertRoadmapToDb(
        repo.materialize_roadmap(_draft(_topic(1, "Ownership"))), "userA"
    )
    assert col.insert_one.await_args.args[0]["status"] == "paused"


async def test_a_new_roadmap_starts_active_when_there_is_room(monkeypatch):
    col = _active_db(monkeypatch, _other(repo.MAX_ACTIVE_ROADMAPS - 1))
    await repo.insertRoadmapToDb(
        repo.materialize_roadmap(_draft(_topic(1, "Ownership"))), "userA"
    )
    assert col.insert_one.await_args.args[0]["status"] == "active"


async def test_reopening_a_finished_roadmap_parks_it_when_the_slots_are_full(
    monkeypatch,
):
    """The rollup would otherwise hand out a third active roadmap off the back of
    a topic edit — an activation nobody asked for."""
    col = _active_db(monkeypatch, _other(repo.MAX_ACTIVE_ROADMAPS))
    col.find_one = AsyncMock(
        return_value={"status": "completed", "topics": [{"progress_status": "in_progress"}]}
    )

    await repo._rollup_roadmap_status(_OID, "userA")

    assert col.update_one.await_args.args[1]["$set"]["status"] == "paused"


async def test_reopening_a_finished_roadmap_resumes_it_when_a_slot_is_free(monkeypatch):
    col = _active_db(monkeypatch, _other(repo.MAX_ACTIVE_ROADMAPS - 1))
    col.find_one = AsyncMock(
        return_value={"status": "completed", "topics": [{"progress_status": "in_progress"}]}
    )

    await repo._rollup_roadmap_status(_OID, "userA")

    assert col.update_one.await_args.args[1]["$set"]["status"] == "active"


async def test_stats_report_the_cap_so_the_list_can_show_it(monkeypatch):
    col = _collection()
    col.find.return_value.to_list = AsyncMock(return_value=[])
    _patch_db(monkeypatch, col)
    stats = await repo.learning_stats("userA")
    assert stats["roadmaps"]["max_active"] == repo.MAX_ACTIVE_ROADMAPS


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
        # Passing the checkpoint that completed the topic also closes its
        # outstanding digests (see close_topic_digests), so a completion day is
        # a marked day too — by that route rather than by being a completion.
        marked=[{"updatedAt": written["topics.$.completed_at"]}],
    )
    stats = await repo.learning_stats("userA")
    assert stats["topics"]["completed"] == 1
    assert stats["topics"]["percent"] == 100
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


def _tips_output(bullets):
    from app.agents.learning_tracker.state import TopicTipsOutput

    return TopicTipsOutput(bullets=list(bullets))


def _started(**over) -> dict:
    return {"id": "t1", "title": "A", "order": 1, "progress_status": "in_progress", **over}


def _digest_gen(monkeypatch, unread_count=0, prior=()):
    """Stub build_digest's collaborators, and record whether it spent anything.

    `build_digest` reads the backlog off the digest history rather than counting
    it separately, so an unread count is expressed as history: `prior` entries
    are acknowledged, and `unread_count` unacknowledged ones follow.
    """
    import app.agents.learning_tracker.triggers as trig

    history = [{"status": "marked", **d} for d in prior]
    history += [{"status": "unread", "bullets": []} for _ in range(unread_count)]
    monkeypatch.setattr(trig, "topic_digests", AsyncMock(return_value=history))
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
def _focus_db(monkeypatch, roadmap=None, roadmaps=None, trigger=None, unread=0, digests=None):
    docs = roadmaps if roadmaps is not None else ([roadmap] if roadmap else [])
    # The focus read derives the backlog from the digest history, so `unread` is
    # expressed as that many unacknowledged digests unless a history is given.
    history = (
        digests
        if digests is not None
        else [{"status": "unread"} for _ in range(unread)]
    )
    col = _collection()
    col.find_one = AsyncMock(return_value=trigger)  # the digest trigger
    # Two chains off one double: roadmaps paginate (…skip.limit), digest states
    # don't (…sort straight to to_list).
    col.find.return_value.sort.return_value.to_list = AsyncMock(return_value=list(history))
    col.find.return_value.sort.return_value.skip.return_value.limit.return_value.to_list = AsyncMock(
        return_value=[dict(d) for d in docs]
    )
    monkeypatch.setattr(
        repo, "get_db", lambda: {"roadmaps": col, "triggers": col, repo.DIGESTS: col}
    )
    return col


async def _focus_one(user_id="userA"):
    """The single-roadmap entry, for the cases that only set up one."""
    return (await repo.learning_focus(user_id))["roadmaps"][0]


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
    first = focus["roadmaps"][0]

    assert first["roadmapTitle"] == "Rust"
    assert first["topic"]["title"] == "Ownership"
    assert first["can_generate"] is True
    assert first["blocked_reason"] is None
    assert focus["next_at"]
    assert focus["blocked_reason"] is None


async def test_focus_covers_every_active_roadmap_not_just_the_latest(monkeypatch):
    """The daily sweep digests each active roadmap, so the home screen shows each
    of them — surfacing one hid the queues building on the others."""
    _focus_db(
        monkeypatch,
        roadmaps=[
            {
                "_id": _OID,
                "title": "Rust",
                "topics": [
                    {"id": "t1", "order": 1, "title": "Ownership", "progress_status": "in_progress"}
                ],
            },
            {
                "_id": _OID,
                "title": "Go",
                "topics": [
                    {"id": "t9", "order": 1, "title": "Goroutines", "progress_status": "needs_review"}
                ],
            },
        ],
        trigger=_ENABLED_TRIGGER,
    )
    focus = await repo.learning_focus("userA")

    assert [r["roadmapTitle"] for r in focus["roadmaps"]] == ["Rust", "Go"]
    # Each carries its own verdict — one blocked roadmap doesn't speak for the rest.
    assert focus["roadmaps"][0]["blocked_reason"] is None
    assert focus["roadmaps"][1]["blocked_reason"] == "needs_review"
    assert focus["blocked_reason"] is None


async def test_focus_only_counts_roadmaps_still_active(monkeypatch):
    col = _focus_db(monkeypatch, roadmap={"_id": _OID, "title": "Rust", "topics": []})
    await repo.learning_focus("userA")

    assert col.find.call_args.args[0] == {"user_id": "userA", "status": "active"}


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
    focus = await _focus_one()
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
    focus = await _focus_one()

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
    assert (await _focus_one())["blocked_reason"] == "roadmap_complete"


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
    assert focus["roadmaps"][0]["blocked_reason"] == "digests_off"
    assert focus["next_at"] is None
    # Still pullable by hand — the schedule is off, not the feature.
    assert focus["roadmaps"][0]["can_generate"] is True


async def test_focus_on_no_roadmap_says_so(monkeypatch):
    _focus_db(monkeypatch, roadmaps=[], trigger=_ENABLED_TRIGGER)
    focus = await repo.learning_focus("userA")
    assert focus["blocked_reason"] == "no_roadmap"
    assert focus["roadmaps"] == []


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
    """The drip-feed is done; the checkpoint is what completes it now.

    From the second digest: the first is held by the floor so that every topic
    carries at least one recall check — see `coverage_may_end`."""
    trig, _ = _digest_gen(monkeypatch, prior=[{"bullets": ["already sent"]}])
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
    # One below the cap, so it sends — and it's the next in the run, not a
    # restart: the backlog *is* the history those digests came from.
    assert out["sequence"] == DIGEST_MAX_UNREAD


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

    async def _quiz(topic_title, bullets, questioncount=None):
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
    trig, _ = _digest_gen(monkeypatch, prior=[{"bullets": ["already sent"]}])
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

    out = await repo.apply_checkpoint(_OID, "t1", "userA", 25, missed=["Q about moves"])

    assert out["passed"] is False
    # Held at needs_review, not dropped back to in_progress: coverage already
    # declared the topic taught, so what's owed is revision of what was missed.
    assert out["progress_status"] == "needs_review"
    assert out["needs_revision"] is True
    written = _written(col)
    assert written.get("topics.$.completed_at") is None
    # Nothing was completed, so there is no next review to schedule.
    assert "topics.$.next_review_at" not in written
    assert written["topics.$.checkpoint_attempts"] == 1
    assert written["topics.$.weak_points"] == ["Q about moves"]
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
def _stats_db(monkeypatch, roadmaps, attempts=(), marked=(), tz="UTC"):
    """Stub the three collections learning_stats reads, plus the trigger row it
    takes the learner's timezone from.

    `marked` is the acknowledged digests behind the streak; each is `{"updatedAt":
    <iso>}`, which is when it was noticed.
    """
    roadmap_col, quiz_col, digest_col = MagicMock(), MagicMock(), MagicMock()
    roadmap_col.find = MagicMock(
        return_value=MagicMock(to_list=AsyncMock(return_value=list(roadmaps)))
    )
    quiz_col.find = MagicMock(
        return_value=MagicMock(to_list=AsyncMock(return_value=list(attempts)))
    )
    digest_col.find = MagicMock(
        return_value=MagicMock(to_list=AsyncMock(return_value=list(marked)))
    )
    monkeypatch.setattr(
        repo,
        "get_db",
        lambda: {
            "roadmaps": roadmap_col,
            "quiz_attempts": quiz_col,
            repo.DIGESTS: digest_col,
        },
    )
    # user_zone reads through trigger_store's own get_db, not the repository's.
    trigger_col = MagicMock()
    trigger_col.find_one = AsyncMock(return_value={"timezone": tz})
    monkeypatch.setattr(trigger_store, "get_db", lambda: {"triggers": trigger_col})
    return digest_col


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
    assert stats["roadmaps"] == {
        "total": 2,
        "active": 1,
        "completed": 1,
        "paused": 0,
        "max_active": repo.MAX_ACTIVE_ROADMAPS,
    }
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
                    for d in (0, 1, 2, 5)
                ],
            }
        ],
        marked=[
            {"updatedAt": f"{_day(d)}T09:00:00+00:00"} for d in (0, 1, 2, 5)
        ],  # 3-day run, then a gap
    )
    stats = await repo.learning_stats("userA")
    assert stats["streak_days"] == 3
    assert stats["completed_this_week"] == 4


async def test_the_streak_counts_digests_read_not_topics_finished(monkeypatch):
    """The streak measures showing up, and showing up here means reading the
    day's digest.

    Keyed to topic completions it was unreachable rather than merely wrong: a
    topic costs several instalments plus a passed checkpoint, the drip-feed sends
    one a day, so finishing one on each of two consecutive days is not something
    a learner does. Someone reading every day for a week saw 🔥1 at best.
    """
    _stats_db(
        monkeypatch,
        [
            {
                "status": "active",
                # One topic finished, weeks ago — no help to a streak either way.
                "topics": [
                    {"progress_status": "completed", "completed_at": f"{_day(30)}T09:00:00+00:00"}
                ],
            }
        ],
        marked=[{"updatedAt": f"{_day(d)}T20:00:00+00:00"} for d in range(5)],
    )
    stats = await repo.learning_stats("userA")
    assert stats["streak_days"] == 5
    # And the reverse: finishing a topic is not itself a day of showing up.
    assert stats["completed_this_week"] == 0


async def test_several_digests_read_in_one_day_are_still_one_day(monkeypatch):
    """Catching up on a backlog is not a streak. Three digests marked this
    afternoon is one day of showing up, not three."""
    _stats_db(
        monkeypatch,
        [],
        marked=[{"updatedAt": f"{_day(0)}T{h}:00:00+00:00"} for h in ("09", "14", "21")],
    )
    assert (await repo.learning_stats("userA"))["streak_days"] == 1


async def test_a_quiet_today_does_not_break_the_streak(monkeypatch):
    """The streak should only break once a full day has been missed — otherwise
    it would read as 0 every morning until the learner opened that day's digest."""
    _stats_db(
        monkeypatch,
        [],
        marked=[{"updatedAt": f"{_day(d)}T09:00:00+00:00"} for d in (1, 2)],
    )
    assert (await repo.learning_stats("userA"))["streak_days"] == 2


async def test_a_digest_with_no_timestamp_cannot_contribute_to_a_streak(monkeypatch):
    _stats_db(monkeypatch, [], marked=[{}, {"updatedAt": None}, {"updatedAt": "nonsense"}])
    assert (await repo.learning_stats("userA"))["streak_days"] == 0


async def test_stats_on_an_empty_account_are_zeros_not_a_crash(monkeypatch):
    _stats_db(monkeypatch, [])
    stats = await repo.learning_stats("userA")
    assert stats["topics"]["percent"] == 0
    assert stats["streak_days"] == 0
    assert stats["quizzes"]["average_score"] == 0


# --------------------------------------------------------------------------- #
# which day something happened on — the learner's, not the server's
# --------------------------------------------------------------------------- #
def test_a_day_is_the_learners_day_not_the_servers():
    """Asia/Kolkata is UTC+5:30, so an evening in UTC is already tomorrow there.

    On UTC boundaries a learner reading their digest after midnight local has it
    filed under yesterday — and the run they were building breaks the next day
    for no reason they can see.
    """
    from zoneinfo import ZoneInfo

    stamp = "2026-08-17T19:00:00+00:00"  # 00:30 on the 18th in Kolkata
    assert repo._local_day(stamp, ZoneInfo("UTC")) == "2026-08-17"
    assert repo._local_day(stamp, ZoneInfo("Asia/Kolkata")) == "2026-08-18"
    # And the other way: a Los Angeles morning is still the previous day there.
    assert repo._local_day("2026-08-18T06:00:00+00:00", ZoneInfo("America/Los_Angeles")) == (
        "2026-08-17"
    )


def test_a_timestamp_written_before_they_were_tz_aware_reads_as_utc():
    from zoneinfo import ZoneInfo

    assert repo._local_day("2026-08-17T19:00:00", ZoneInfo("Asia/Kolkata")) == "2026-08-18"


@pytest.mark.parametrize("stamp", [None, "", "not a date", "2026-13-45"])
def test_an_unusable_timestamp_is_dropped_rather_than_raising(stamp):
    from zoneinfo import ZoneInfo

    assert repo._local_day(stamp, ZoneInfo("UTC")) is None


def test_consecutive_days_stops_at_the_first_gap():
    from datetime import date

    today = date(2026, 8, 17)
    days = {"2026-08-17", "2026-08-16", "2026-08-15", "2026-08-13"}
    assert repo.consecutive_days(days, today) == 3
    # A quiet today counts back from yesterday instead of resetting to zero.
    assert repo.consecutive_days(days - {"2026-08-17"}, today) == 2
    # Two quiet days is a genuinely broken run.
    assert repo.consecutive_days(days - {"2026-08-17", "2026-08-16"}, today) == 0
    assert repo.consecutive_days(set(), today) == 0


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


# --------------------------------------------------------------------------- #
# a store that actually applies the writes
#
# The fakes above record calls; these tests need the *result* of a positional
# `$set`/`$inc`, because the behaviour under test is a counter crossing a
# threshold rather than a query being shaped correctly.
# --------------------------------------------------------------------------- #
def _live_roadmap(monkeypatch, topic: dict, **roadmap):
    """A one-topic roadmap whose updates land, so state can be read back."""
    import copy

    doc = {
        "_id": ObjectId(_OID),
        "user_id": "userA",
        "title": "Rust",
        "status": "active",
        "topics": [copy.deepcopy(topic)],
        **roadmap,
    }
    col = _collection()
    col.find_one = AsyncMock(side_effect=lambda *a, **k: copy.deepcopy(doc))

    async def _update(filt, update, **kw):
        t = doc["topics"][0]
        for key, value in (update.get("$set") or {}).items():
            if key.startswith("topics.$."):
                t[key.removeprefix("topics.$.")] = value
        for key, value in (update.get("$inc") or {}).items():
            if key.startswith("topics.$."):
                field = key.removeprefix("topics.$.")
                t[field] = int(t.get(field) or 0) + value
        return MagicMock(matched_count=1, modified_count=1)

    col.update_one = AsyncMock(side_effect=_update)
    monkeypatch.setattr(repo, "get_db", lambda: {"roadmaps": col})
    monkeypatch.setattr(repo, "_rollup_roadmap_status", AsyncMock())
    monkeypatch.setattr(repo, "start_topic", AsyncMock(return_value=False))
    return doc


# --------------------------------------------------------------------------- #
# the digest archive, filtered
# --------------------------------------------------------------------------- #
async def test_the_archive_narrows_to_one_roadmap_and_then_one_topic(monkeypatch):
    """Filtering in the query, not the client: `limit` has to mean the newest N
    of what was asked for, or a rarely-written-about topic comes back empty."""
    col = _digest_db(monkeypatch)
    await repo.list_digests("userA", roadmapId="r1", topicId="t1")

    query = col.find.call_args.args[0]
    assert query["roadmapId"] == "r1"
    assert query["topicId"] == "t1"
    assert query["user_id"] == "userA"  # still scoped to the caller


async def test_a_roadmap_filter_narrows_the_catch_up_queue_rather_than_widening_it(
    monkeypatch,
):
    """Asking for a parked roadmap under active_only must return nothing, not
    fall back to every active roadmap."""
    _digest_db(monkeypatch, digests=[{"_id": _OID}], roadmaps=[{"_id": "r1"}])
    assert await repo.list_digests("userA", active_only=True, roadmapId="parked") == []


async def test_a_roadmap_filter_inside_the_active_set_still_applies(monkeypatch):
    col = _digest_db(monkeypatch, roadmaps=[{"_id": "r1"}, {"_id": "r2"}])
    await repo.list_digests("userA", active_only=True, roadmapId="r2")
    assert col.find.call_args.args[0]["roadmapId"] == {"$in": ["r2"]}


# --------------------------------------------------------------------------- #
# recovery from a failed checkpoint
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "topic,blocked",
    [
        ({}, False),
        ({"checkpoint_attempts": 1, "revisions_done": 0}, True),
        ({"checkpoint_attempts": 1, "revisions_done": 1}, False),
        ({"checkpoint_attempts": 2, "revisions_done": 1}, True),
        # Written before the fields existed: absence is not a debt.
        ({"progress_status": "needs_review"}, False),
    ],
)
def test_a_failed_attempt_owes_revision_before_the_next_one(topic, blocked):
    assert repo.revision_outstanding(topic) is blocked


async def test_failing_holds_the_topic_at_needs_review(monkeypatch):
    """Dropping back to in_progress reopens the teaching drip-feed on a topic
    coverage already finished — one more arbitrary digest, then straight back
    here. What's owed is revision of what was actually missed."""
    doc = _live_roadmap(monkeypatch, _started(progress_status="needs_review"))

    out = await repo.apply_checkpoint(_OID, "t1", "userA", 25, missed=["what moves?"])

    assert out["progress_status"] == "needs_review"
    assert out["needs_revision"] is True
    topic = doc["topics"][0]
    assert topic["checkpoint_attempts"] == 1
    assert topic["weak_points"] == ["what moves?"]
    # Nothing was completed, so nothing goes on the resurface ladder.
    assert topic.get("next_review_at") is None
    assert repo.revision_outstanding(topic) is True


async def test_marking_the_revision_digest_reopens_the_retry(monkeypatch):
    doc = _live_roadmap(
        monkeypatch,
        _started(progress_status="needs_review", checkpoint_attempts=1, revisions_done=0),
    )
    assert await repo.record_revision(_OID, "t1", "userA") is True
    assert repo.revision_outstanding(doc["topics"][0]) is False


async def test_a_spare_revision_banks_no_credit_against_a_future_failure(monkeypatch):
    """Otherwise a second digest read now buys a skipped revision later."""
    doc = _live_roadmap(
        monkeypatch,
        _started(progress_status="needs_review", checkpoint_attempts=1, revisions_done=1),
    )
    assert await repo.record_revision(_OID, "t1", "userA") is False
    assert doc["topics"][0]["revisions_done"] == 1


async def test_a_revision_digest_is_not_a_teaching_digest(monkeypatch):
    """It asks what they got wrong, not what hasn't been covered — and needs no
    web search, since the material is their own missed questions."""
    import app.agents.learning_tracker.triggers as trig

    monkeypatch.setattr(trig, "topic_digests", AsyncMock(return_value=[]))
    search = MagicMock()
    search.ainvoke = AsyncMock(return_value={"results": []})
    monkeypatch.setattr(trig, "tavily_search_tool", search)
    col = _collection()
    monkeypatch.setattr(trig, "get_db", lambda: {trig.DIGESTS: col, "quizzes": col})
    monkeypatch.setattr(trig, "send_push_notification", AsyncMock())
    _fake_tips(monkeypatch, trig, bullets=["try it this way instead"])

    topic = _started(
        progress_status="needs_review",
        checkpoint_attempts=1,
        revisions_done=0,
        weak_points=["what moves?"],
    )
    out = await trig.build_revision_digest(
        "userA", {"_id": _OID, "summary": "", "topics": [topic]}, "t1", notify=False
    )

    assert out["kind"] == "revision"
    assert out["sequence"] is None  # must not shift the recall-check cadence
    assert out["quizId"] is None  # the retry is the assessment
    search.ainvoke.assert_not_awaited()


async def test_no_revision_digest_when_none_is_owed(monkeypatch):
    import app.agents.learning_tracker.triggers as trig

    monkeypatch.setattr(trig, "topic_digests", AsyncMock(return_value=[]))
    out = await trig.build_revision_digest(
        "userA", {"_id": _OID, "topics": [_started(progress_status="needs_review")]}, "t1"
    )
    assert out is None


async def test_only_one_revision_digest_waits_at_a_time(monkeypatch):
    """A stack of them is a worse answer to a failed checkpoint than one."""
    import app.agents.learning_tracker.triggers as trig

    monkeypatch.setattr(
        trig,
        "topic_digests",
        AsyncMock(return_value=[{"kind": "revision", "status": "unread", "bullets": []}]),
    )
    topic = _started(
        progress_status="needs_review", checkpoint_attempts=1, revisions_done=0
    )
    out = await trig.build_revision_digest(
        "userA", {"_id": _OID, "topics": [topic]}, "t1", notify=False
    )
    assert out is None


# --------------------------------------------------------------------------- #
# checkpoint attempt policy
#
# A completion gate is only a gate if failing costs something. Each of these
# closes one way of walking through it.
# --------------------------------------------------------------------------- #
def _attempts(*ages_minutes):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    return [
        {"createdAt": (now - timedelta(minutes=m)).isoformat()} for m in ages_minutes
    ]


def test_a_first_attempt_is_never_blocked():
    assert repo.retry_block([]) is None


def test_bouncing_straight_back_in_hits_the_cooldown():
    blocked = repo.retry_block(_attempts(1))
    assert blocked["reason"] == "cooldown"
    assert blocked["retry_at"]


def test_the_cooldown_expires():
    from app.core.config import CHECKPOINT_RETRY_COOLDOWN_MINUTES as COOL

    assert repo.retry_block(_attempts(COOL + 1)) is None


def test_an_afternoon_of_grinding_hits_the_daily_cap():
    from app.core.config import (
        CHECKPOINT_MAX_ATTEMPTS_PER_DAY as CAP,
        CHECKPOINT_RETRY_COOLDOWN_MINUTES as COOL,
    )

    blocked = repo.retry_block(_attempts(*(COOL + 1 + i * 60 for i in range(CAP))))
    assert blocked["reason"] == "daily_limit"
    assert blocked["attempts_today"] == CAP


def test_yesterdays_attempts_do_not_count_against_today():
    from app.core.config import CHECKPOINT_MAX_ATTEMPTS_PER_DAY as CAP

    two_days = 60 * 24 * 2
    assert repo.retry_block(_attempts(*([two_days] * (CAP + 1)))) is None


async def test_an_unreadable_attempt_history_does_not_hand_out_a_free_retry(monkeypatch):
    col = _collection()
    col.find = MagicMock(side_effect=RuntimeError("mongo down"))
    monkeypatch.setattr(repo, "get_db", lambda: {"quiz_attempts": col})

    attempts = await repo.recent_attempts("userA", "r1", "t1")
    # Fails closed: an unknown history reads as "at the cap", not "never tried".
    assert repo.retry_block(attempts) is not None


# ── feedback that doesn't hand over the answer ───────────────────────────────
_GRADED_QS = [
    {
        "question": "What does move do?",
        "options": ["copies", "transfers"],
        "answer": 1,
        "outcome": "Ownership transfer",
        "hint": "Re-read what happens to the original binding.",
    }
]
_GRADED = {
    "total": 1,
    "correct": 0,
    "score": 0,
    "review": [
        {"question": 0, "selected": 0, "correctAnswer": 1, "correctOption": "transfers"}
    ],
}


def test_a_failed_checkpoint_never_returns_the_answer():
    """Handing back `correctOption` on a failure turns the retry into
    transcription, which is exactly what a completion gate must not permit."""
    out = repo.redact_review(_GRADED, reveal=False, questions=_GRADED_QS)
    blob = json.dumps(out)

    assert "transfers" not in blob
    assert "correctAnswer" not in blob and "correctOption" not in blob
    assert out["answers_revealed"] is False


def test_a_failed_checkpoint_still_says_what_to_go_over():
    """Withholding the answer without saying anything leaves nowhere to go."""
    out = repo.redact_review(_GRADED, reveal=False, questions=_GRADED_QS)
    assert out["review"][0]["outcome"] == "Ownership transfer"
    assert out["review"][0]["hint"]


def test_passing_reveals_the_answers():
    out = repo.redact_review(_GRADED, reveal=True, questions=_GRADED_QS)
    assert out["review"][0]["correctOption"] == "transfers"


def test_redaction_never_touches_the_grade():
    out = repo.redact_review(_GRADED, reveal=False, questions=_GRADED_QS)
    assert (out["score"], out["correct"], out["total"]) == (0, 0, 1)


# ── review questions don't repeat ────────────────────────────────────────────
async def test_previous_questions_are_fed_back_so_reviews_vary(monkeypatch):
    """Regenerating is not varying: the same description and outcomes lead the
    model to the same handful of questions, and over a spaced-repetition ladder
    that measures memory of the answer rather than the material."""
    col = _collection()
    col.find.return_value.sort.return_value.limit.return_value.to_list = AsyncMock(
        return_value=[
            {"questions": [{"question": "What is ownership?"}, {"question": "dup"}]},
            {"questions": [{"question": "dup"}, {"question": "When does a move happen?"}]},
        ]
    )
    monkeypatch.setattr(repo, "get_db", lambda: {"quizzes": col})

    asked = await repo.asked_questions("userA", "r1", "t1")

    assert asked == ["What is ownership?", "dup", "When does a move happen?"]
    assert col.find.call_args.args[0]["user_id"] == "userA"


# --------------------------------------------------------------------------- #
# misconceptions
# --------------------------------------------------------------------------- #
def test_an_attempt_records_what_was_wrong_in_words_not_indices():
    """Index 2 is meaningless once the quiz is out of scope; this is what makes
    an attempt readable later, and is the whole basis of the analysis."""
    misses = repo.attempt_misses(_GRADED_QS, _GRADED)
    assert misses == [
        {"question": "What does move do?", "chosen": "copies", "correct": "transfers"}
    ]


def test_a_skipped_question_is_not_recorded_as_a_wrong_belief():
    """A blank is different evidence from a wrong pick, and flattening the two
    invents a belief the learner never expressed."""
    skipped = {**_GRADED, "review": [{**_GRADED["review"][0], "selected": None}]}
    assert repo.attempt_misses(_GRADED_QS, skipped)[0]["chosen"] is None


async def test_thin_evidence_is_not_analysed(monkeypatch):
    """One bad day is not a misconception, and a model pressed to find a pattern
    will paraphrase the latest mistake and call it a diagnosis."""
    from app.agents.learning_tracker import service

    monkeypatch.setattr(service, "topic_misses", AsyncMock(return_value=[{"question": "q"}]))
    extract = AsyncMock()
    monkeypatch.setattr(service, "extract_misconceptions", extract)

    assert await service.refresh_misconceptions("userA", "r1", "t1", "Rust") is None
    extract.assert_not_awaited()


async def test_unchanged_evidence_is_not_reanalysed(monkeypatch):
    """Same input, same patterns — re-deriving them is a wasted LLM call."""
    from app.agents.learning_tracker import service
    from app.core.config import MISCONCEPTION_MIN_EVIDENCE as MIN

    misses = [{"question": f"q{i}"} for i in range(MIN)]
    monkeypatch.setattr(service, "topic_misses", AsyncMock(return_value=misses))
    monkeypatch.setattr(
        service,
        "get_misconceptions",
        AsyncMock(return_value={"patterns": [{"label": "cached"}], "misses_analyzed": MIN}),
    )
    extract = AsyncMock()
    monkeypatch.setattr(service, "extract_misconceptions", extract)

    out = await service.refresh_misconceptions("userA", "r1", "t1", "Rust")

    assert out == [{"label": "cached"}]
    extract.assert_not_awaited()


async def test_a_failed_analysis_keeps_the_patterns_it_already_had(monkeypatch):
    from app.agents.learning_tracker import service
    from app.core.config import MISCONCEPTION_MIN_EVIDENCE as MIN

    monkeypatch.setattr(
        service, "topic_misses", AsyncMock(return_value=[{"question": f"q{i}"} for i in range(MIN)])
    )
    monkeypatch.setattr(
        service,
        "get_misconceptions",
        AsyncMock(return_value={"patterns": [{"label": "kept"}], "misses_analyzed": 0}),
    )
    monkeypatch.setattr(
        service, "extract_misconceptions", AsyncMock(side_effect=RuntimeError("boom"))
    )
    save = AsyncMock()
    monkeypatch.setattr(service, "save_misconceptions", save)

    assert await service.refresh_misconceptions("userA", "r1", "t1", "Rust") == [
        {"label": "kept"}
    ]
    save.assert_not_awaited()


# --------------------------------------------------------------------------- #
# mastery
#
# The lifetime quiz average counts a week-one failure as evidence about today
# and stops moving once there's any history. These are the cases that broke it.
# --------------------------------------------------------------------------- #
def _scored(*pairs):
    """(days_ago, score) → attempt rows."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    return [
        {"score": s, "createdAt": (now - timedelta(days=d)).isoformat()} for d, s in pairs
    ]


def test_a_topic_with_no_attempts_has_no_mastery():
    """None, not zero — no evidence is not the same as knowing nothing."""
    assert repo.topic_mastery([]) is None


def test_recent_attempts_weigh_more_than_old_ones():
    """A learner who struggled and then got it is measured on where they are."""
    m = repo.topic_mastery(_scored((30, 40), (20, 55), (1, 95)))
    flat = round((40 + 55 + 95) / 3)

    assert m["mastery"] > flat
    assert m["trend"] == "improving"


def test_slipping_is_visible_even_while_the_average_looks_healthy():
    m = repo.topic_mastery(_scored((30, 95), (20, 90), (1, 55)))
    flat = round((95 + 90 + 55) / 3)

    assert m["mastery"] < flat
    assert m["trend"] == "slipping"


def test_a_single_attempt_has_no_trend():
    assert repo.topic_mastery(_scored((2, 90)))["trend"] == "new"


def test_a_topic_reviewed_on_schedule_has_not_faded():
    from datetime import datetime, timedelta, timezone

    due = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    m = repo.topic_mastery(_scored((5, 95)), next_review_at=due)
    assert m["retention"] == 1.0 and m["mastery"] == m["score"]


def test_an_overdue_topic_decays():
    """Aced in March and untouched since is not knowledge you still have."""
    from datetime import datetime, timedelta, timezone

    overdue = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    m = repo.topic_mastery(_scored((60, 95)), next_review_at=overdue)

    assert m["score"] == 95  # what they scored is unchanged
    assert m["mastery"] < 60  # what they're assumed to hold is not
    assert m["overdue_days"] == 40


def test_the_summary_leads_with_the_weakest_topics():
    topics = [
        {**repo.topic_mastery(_scored((1, 90))), "title": "strong"},
        {**repo.topic_mastery(_scored((1, 40))), "title": "weak"},
        {**repo.topic_mastery(_scored((1, 65))), "title": "middling"},
    ]
    summary = repo.mastery_summary(topics)

    assert [t["title"] for t in summary["weakest"]] == ["weak", "middling", "strong"]
    assert summary["topics_scored"] == 3


def test_the_summary_reports_nothing_rather_than_zero_when_untested():
    assert repo.mastery_summary([])["score"] is None


# --------------------------------------------------------------------------- #
# coverage decides before the spend
# --------------------------------------------------------------------------- #
async def test_a_covered_topic_costs_nothing_to_decline(monkeypatch):
    """Checking after generating means the answer arrives once a search and a
    tips call have been spent, and the learner receives a digest the topic
    didn't need — the drip-feed always overshoots by one."""
    trig, search = _digest_gen(
        monkeypatch,
        prior=[{"bullets": ["a"]}, {"bullets": ["b"], "coverage_complete": True}],
    )
    tips = AsyncMock()
    monkeypatch.setattr(trig, "check_coverage", tips)

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert out is None
    search.ainvoke.assert_not_awaited()
    # The stored verdict was computed over exactly these bullets, so there is
    # nothing to re-derive.
    tips.assert_not_awaited()
    assert trig.set_topic_progress.await_args.args[2] == "needs_review"


async def test_a_stored_not_covered_verdict_is_trusted_too(monkeypatch):
    """Otherwise every digest pays for the same verdict twice."""
    trig, search = _digest_gen(
        monkeypatch, prior=[{"bullets": ["a"], "coverage_complete": False}]
    )
    coverage = AsyncMock(return_value=repo_coverage(False))
    monkeypatch.setattr(trig, "check_coverage", coverage)
    _fake_tips(monkeypatch, trig)

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert out is not None
    assert coverage.await_count == 1  # the post-generation one only


async def test_a_digest_written_before_the_field_existed_is_re_checked(monkeypatch):
    trig, search = _digest_gen(
        monkeypatch, prior=[{"bullets": ["a"]}, {"bullets": ["b"]}]
    )
    coverage = AsyncMock(return_value=repo_coverage(True))
    monkeypatch.setattr(trig, "check_coverage", coverage)

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert out is None
    coverage.assert_awaited()  # re-derived rather than trusted
    search.ainvoke.assert_not_awaited()


# --------------------------------------------------------------------------- #
# deleting a roadmap
# --------------------------------------------------------------------------- #
async def test_deleting_a_roadmap_takes_its_children_with_it(monkeypatch):
    """Orphans put digests for a deleted roadmap in the catch-up queue and its
    graded attempts in the mastery average for a topic nobody can open."""
    col = _collection()
    col.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    col.delete_many = AsyncMock(return_value=MagicMock(deleted_count=2))
    col.count_documents = AsyncMock(return_value=3)
    monkeypatch.setattr(
        repo,
        "get_db",
        lambda: {
            "roadmaps": col,
            repo.DIGESTS: col,
            repo.NOTES: col,
            "quizzes": col,
            "quiz_attempts": col,
            repo.MISCONCEPTIONS: col,
            repo.TODOS: col,
        },
    )

    removed = await repo.delete_roadmap(_OID, "userA")

    assert removed["roadmap"] == 1
    for child in ("digests", "notes", "quizzes", "attempts", "misconceptions"):
        assert removed[child] == 2
    # To-dos live in another feature and are counted, not deleted.
    assert removed["linked_tasks"] == 3


async def test_deleting_someone_elses_roadmap_removes_nothing(monkeypatch):
    col = _collection()
    col.delete_one = AsyncMock(return_value=MagicMock(deleted_count=0))
    col.delete_many = AsyncMock()
    monkeypatch.setattr(repo, "get_db", lambda: {"roadmaps": col})

    assert await repo.delete_roadmap(_OID, "attacker") is None
    col.delete_many.assert_not_awaited()  # the cascade never starts


# --------------------------------------------------------------------------- #
# the Feynman checkpoint
# --------------------------------------------------------------------------- #
async def test_explaining_it_well_buys_a_longer_interval(monkeypatch):
    """The only reason to do an optional exercise. Without the payout it's a
    text box nobody fills in."""
    from app.core.config import FEYNMAN_LADDER_BONUS as BONUS

    plain = _live_roadmap(monkeypatch, _started(progress_status="needs_review"))
    base = await repo.apply_checkpoint(_OID, "t1", "userA", 100)

    explained = _live_roadmap(
        monkeypatch, _started(progress_status="needs_review", feynman_passed=True)
    )
    bonus = await repo.apply_checkpoint(_OID, "t1", "userA", 100)

    assert bonus["review_count"] == base["review_count"] + BONUS
    assert bonus["feynman_bonus"] == BONUS
    assert bonus["next_review_at"] > base["next_review_at"]
    # Spent on use, so the next review has to earn its own.
    assert explained["topics"][0]["feynman_passed"] is False


async def test_a_poor_explanation_costs_the_learner_nothing(monkeypatch):
    """An exercise that can lose you progress is one nobody volunteers for."""
    doc = _live_roadmap(monkeypatch, _started(feynman_passed=True))

    await repo.record_explanation(
        _OID, "t1", "userA", {"score": 20}, "a poor attempt", passed=False
    )

    assert doc["topics"][0]["feynman_passed"] is True  # credit already earned stands
    assert doc["topics"][0]["feynman_score"] == 20


async def test_a_passing_explanation_records_the_credit(monkeypatch):
    doc = _live_roadmap(monkeypatch, _started())

    await repo.record_explanation(
        _OID, "t1", "userA", {"score": 85}, "in my own words…", passed=True
    )

    topic = doc["topics"][0]
    assert topic["feynman_passed"] is True
    # Kept because it's the only record of how this learner talks about the topic.
    assert topic["feynman_text"] == "in my own words…"


def test_an_edit_does_not_wipe_what_the_learner_earned():
    """Otherwise revising a roadmap is a way to reset the review ladder — and to
    skip the revision owed after a failed checkpoint."""
    stored = {
        "_id": _OID,
        "topics": [
            {
                "id": "t1",
                "order": 1,
                "title": "Ownership",
                "progress_status": "completed",
                "review_count": 3,
                "checkpoint_attempts": 2,
                "revisions_done": 1,
                "feynman_passed": True,
            }
        ],
    }
    merged = repo.merge_roadmap(stored, _draft(_topic(1, "Ownership")))
    topic = merged.topics[0]

    assert topic.review_count == 3
    assert (topic.checkpoint_attempts, topic.revisions_done) == (2, 1)
    assert topic.feynman_passed is True


# --------------------------------------------------------------------------- #
# the written one-liner on later digest checks
# --------------------------------------------------------------------------- #
_MIXED_QS = [
    {"question": "A?", "options": ["x", "y"], "answer": 0, "kind": "choice"},
    {"question": "B?", "options": ["x", "y"], "answer": 1, "kind": "choice"},
    {"question": "Say why", "options": [], "answer": 0, "kind": "open", "expected": "…"},
]


def test_the_written_question_is_found_by_position():
    assert [i for i, _ in repo.open_questions(_MIXED_QS)] == [2]
    assert repo.open_questions([{"question": "A?", "options": [], "answer": 0}]) == []


def test_a_perfect_tap_round_is_not_dragged_down_by_the_written_question():
    """Counting it in the denominator scores a flawless round at 2/3 — and
    against an all-or-nothing pass mark that makes the digest unmarkable, which
    jams the topic for good."""
    graded = repo.grade_quiz(_MIXED_QS, {0: 0, 1: 1})
    assert (graded["total"], graded["score"]) == (2, 100)


def test_the_written_question_is_never_graded_as_a_wrong_tap():
    graded = repo.grade_quiz(_MIXED_QS, {0: 0, 1: 1})
    assert graded["review"] == []


def test_taps_are_still_graded_normally_alongside_it():
    graded = repo.grade_quiz(_MIXED_QS, {0: 1, 1: 1})
    assert graded["score"] == 50
    assert [r["question"] for r in graded["review"]] == [0]


def test_a_quiz_with_nothing_to_tap_cannot_be_scored_against():
    """0/0 against a pass mark of 100 is a digest that can never be marked."""
    only_open = [_MIXED_QS[2]]
    assert repo.grade_quiz(only_open, {}) == {
        "total": 0,
        "correct": 0,
        "score": 0,
        "review": [],
    }


def test_quizzes_written_before_open_questions_existed_grade_unchanged():
    legacy = [{"question": "A?", "options": ["x", "y"], "answer": 0}]
    assert repo.grade_quiz(legacy, {0: 0})["score"] == 100
    assert repo.grade_quiz(legacy, {0: 1})["score"] == 0


async def test_a_one_liner_rides_along_from_the_fourth_digest(monkeypatch):
    """Late rather than from the start: the habit forms on tap-only checks, and
    by the fourth there is enough material that a sentence is worth the
    keystroke."""
    from app.agents.learning_tracker.state import Question

    trig, _ = _digest_gen(
        monkeypatch, prior=[{"bullets": [f"b{i}"]} for i in range(3)]
    )
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    _fake_tips(monkeypatch, trig)
    monkeypatch.setattr(trig, "build_digest_quiz", AsyncMock(return_value=_quiz_output()))
    monkeypatch.setattr(
        trig,
        "build_oneliner",
        AsyncMock(
            return_value=Question(question="Say why", options=[], answer=0, kind="open")
        ),
    )

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert out["sequence"] == 4
    quiz_doc = next(
        c.args[0]
        for c in trig.get_db()["quizzes"].insert_one.await_args_list
        if c.args[0].get("kind") == "digest"
    )
    assert [q["kind"] for q in quiz_doc["questions"]][-1] == "open"


async def test_a_failed_one_liner_still_ships_the_check(monkeypatch):
    """The taps are the gate; losing the sentence costs signal, not the
    learner's ability to acknowledge the digest."""
    trig, _ = _digest_gen(monkeypatch, prior=[{"bullets": [f"b{i}"]} for i in range(3)])
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    _fake_tips(monkeypatch, trig)
    monkeypatch.setattr(trig, "build_digest_quiz", AsyncMock(return_value=_quiz_output()))
    monkeypatch.setattr(
        trig, "build_oneliner", AsyncMock(side_effect=RuntimeError("boom"))
    )

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)
    assert out is not None and out["quizId"]


async def test_an_empty_recall_check_is_never_attached(monkeypatch):
    """A quiz with no questions scores 0 against a pass mark of 100, so its
    digest can never be marked — and an unmarkable digest holds a slot under
    DIGEST_MAX_UNREAD forever, stopping the topic getting any further ones."""
    from app.agents.learning_tracker.state import QuizOutput

    trig, _ = _digest_gen(monkeypatch, prior=[{"bullets": ["earlier"]}])
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    _fake_tips(monkeypatch, trig)
    monkeypatch.setattr(
        trig, "build_digest_quiz", AsyncMock(return_value=QuizOutput(quiz=[]))
    )

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)
    assert out is not None and out["quizId"] is None


# --------------------------------------------------------------------------- #
# misconceptions steer generation, not just reporting
#
# The tracker is only worth its LLM calls if something consumes it. These assert
# the loop is closed at both ends — teaching aimed at what the learner gets
# wrong, and checkpoints that come back to it later.
# --------------------------------------------------------------------------- #
_PATTERN = {
    "label": "Confuses move with copy",
    "detail": "Believes assignment copies; it transfers ownership.",
    "probe": "ask what the original binding can do afterwards",
}


def _days_ago(days):
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _tips_capture(monkeypatch, trig):
    """Swap the tips prompt for something that records what it was rendered with."""
    seen = {}

    class _Chain:
        def __or__(self, _other):
            return self

        async def ainvoke(self, variables, *a, **k):
            seen.update(variables)
            return _tips_output(["a new bullet"])

    monkeypatch.setattr(trig, "_TIPS_PROMPT", _Chain())
    return seen


async def test_a_digest_teaches_against_what_the_learner_gets_wrong(monkeypatch):
    """Teaching that ignores what someone already believes is teaching into a
    headwind — the misconception is what the new bullet has to displace."""
    trig, _ = _digest_gen(monkeypatch, prior=[{"bullets": ["b1"], "coverage_complete": False}])
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    monkeypatch.setattr(
        trig, "get_misconceptions", AsyncMock(return_value={"patterns": [_PATTERN]})
    )
    monkeypatch.setattr(trig, "mark_misconceptions_addressed", AsyncMock())
    seen = _tips_capture(monkeypatch, trig)

    await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert "Confuses move with copy" in seen["misconceptions"]
    assert "transfers ownership" in seen["misconceptions"]


async def test_a_learner_with_no_recorded_confusions_gets_an_unchanged_prompt(monkeypatch):
    trig, _ = _digest_gen(monkeypatch, prior=[{"bullets": ["b1"], "coverage_complete": False}])
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    monkeypatch.setattr(trig, "get_misconceptions", AsyncMock(return_value=None))
    marked = AsyncMock()
    monkeypatch.setattr(trig, "mark_misconceptions_addressed", marked)
    seen = _tips_capture(monkeypatch, trig)

    await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert seen["misconceptions"] == ""
    marked.assert_not_awaited()


async def test_the_same_correction_is_not_re_taught_in_the_next_digest(monkeypatch):
    """Repeating it every digest is nagging, and crowds out the material the
    drip-feed is supposed to be advancing through."""
    from app.core.config import MISCONCEPTION_RETEACH_DAYS as QUIET

    trig, _ = _digest_gen(monkeypatch, prior=[{"bullets": ["b1"], "coverage_complete": False}])
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    monkeypatch.setattr(
        trig,
        "get_misconceptions",
        AsyncMock(return_value={"patterns": [{**_PATTERN, "last_taught_at": _days_ago(1)}]}),
    )
    monkeypatch.setattr(trig, "mark_misconceptions_addressed", AsyncMock())
    seen = _tips_capture(monkeypatch, trig)

    await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)
    assert seen["misconceptions"] == ""
    assert QUIET > 1  # the window this test is exercising


async def test_a_confusion_comes_back_once_the_quiet_period_is_up(monkeypatch):
    """A single right answer is not a corrected belief. Coming back to it later
    is the only way to tell one that stuck from one papered over for a session."""
    from app.core.config import MISCONCEPTION_RETEACH_DAYS as QUIET

    trig, _ = _digest_gen(monkeypatch, prior=[{"bullets": ["b1"], "coverage_complete": False}])
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    monkeypatch.setattr(
        trig,
        "get_misconceptions",
        AsyncMock(
            return_value={"patterns": [{**_PATTERN, "last_taught_at": _days_ago(QUIET + 2)}]}
        ),
    )
    monkeypatch.setattr(trig, "mark_misconceptions_addressed", AsyncMock())
    seen = _tips_capture(monkeypatch, trig)

    await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)
    assert "Confuses move with copy" in seen["misconceptions"]


async def test_teaching_against_a_confusion_starts_its_own_clock(monkeypatch):
    trig, _ = _digest_gen(monkeypatch, prior=[{"bullets": ["b1"], "coverage_complete": False}])
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    monkeypatch.setattr(
        trig, "get_misconceptions", AsyncMock(return_value={"patterns": [_PATTERN]})
    )
    marked = AsyncMock()
    monkeypatch.setattr(trig, "mark_misconceptions_addressed", marked)
    _tips_capture(monkeypatch, trig)

    await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert marked.await_args.args[3] == ["Confuses move with copy"]
    assert marked.await_args.args[4] == "last_taught_at"


def test_teaching_a_confusion_does_not_suppress_testing_it():
    """Teach then test is the good pairing — one clock per surface, or a digest
    that explains the confusion would cancel the checkpoint that checks it."""
    from app.core.config import (
        MISCONCEPTION_REPROBE_DAYS as PROBE,
        MISCONCEPTION_RETEACH_DAYS as TEACH,
    )

    just_taught = [{**_PATTERN, "last_taught_at": _days_ago(0)}]

    assert repo.due_misconceptions(just_taught, "last_taught_at", TEACH) == []
    assert len(repo.due_misconceptions(just_taught, "last_probed_at", PROBE)) == 1


def test_a_confusion_nobody_has_addressed_is_always_due():
    assert len(repo.due_misconceptions([_PATTERN], "last_probed_at", 7)) == 1


def test_an_unparseable_stamp_does_not_silence_a_confusion_forever():
    """Failing open: the cost is one repetition, where failing closed would hide
    a real misconception permanently."""
    broken = [{**_PATTERN, "last_probed_at": "not-a-date"}]
    assert len(repo.due_misconceptions(broken, "last_probed_at", 7)) == 1


async def test_re_analysis_does_not_reset_the_decay_clocks(monkeypatch):
    """Patterns are rewritten wholesale on each new piece of evidence. Losing
    the stamps would mean the learner who keeps making mistakes is the one whose
    misconceptions get hammered hardest."""
    # Captured once: _days_ago() is wall-clock, so calling it twice differs.
    yesterday = _days_ago(1)
    stored = {
        "patterns": [
            {**_PATTERN, "last_probed_at": yesterday},
            {"label": "dropped", "last_probed_at": yesterday},
        ]
    }
    col = _collection()
    monkeypatch.setattr(repo, "get_db", lambda: {repo.MISCONCEPTIONS: col})
    monkeypatch.setattr(repo, "get_misconceptions", AsyncMock(return_value=stored))

    await repo.save_misconceptions(
        "userA", "r1", "t1", [dict(_PATTERN), {"label": "new", "detail": "d"}], 9
    )

    written = {
        p["label"]: p for p in col.update_one.await_args.args[1]["$set"]["patterns"]
    }
    assert written["Confuses move with copy"]["last_probed_at"] == yesterday
    assert "last_probed_at" not in written["new"]  # never addressed, so due now
    assert "dropped" not in written  # gone from the analysis, gone from the store


# --------------------------------------------------------------------------- #
# "explain differently" escalation
#
# Two failures on one set of tips is a fact about the tips. Fail-closed on the
# learner's side — they still have to pass — adaptive on the agent's.
# --------------------------------------------------------------------------- #
def test_a_failed_recall_check_is_counted(monkeypatch):
    from app.core.config import DIGEST_QUIZ_PASS_SCORE as PASS

    rows = [{"score": 100}, {"score": 50}, {"score": 0}]
    assert sum(1 for r in rows if r["score"] < PASS) == 2  # the shape counted


async def test_only_sub_pass_attempts_count_towards_escalation(monkeypatch):
    from app.core.config import DIGEST_QUIZ_PASS_SCORE as PASS

    col = _collection()
    col.find.return_value.to_list = AsyncMock(
        return_value=[{"score": 100}, {"score": 50}, {"score": 0}]
    )
    monkeypatch.setattr(repo, "get_db", lambda: {"quiz_attempts": col})

    assert await repo.failed_attempts("userA", "r1", "t1", "Q2", PASS) == 2
    assert col.find.call_args.args[0]["quizId"] == "Q2"
    assert col.find.call_args.args[0]["user_id"] == "userA"


async def test_an_unreadable_history_does_not_escalate(monkeypatch):
    """Better to let them try again than to spend an LLM call on a guess."""
    col = _collection()
    col.find = MagicMock(side_effect=RuntimeError("mongo down"))
    monkeypatch.setattr(repo, "get_db", lambda: {"quiz_attempts": col})

    assert await repo.failed_attempts("userA", "r1", "t1", "Q2", 100) == 0


def _reteach_env(monkeypatch, style=None, digests=None):
    import app.agents.learning_tracker.triggers as trig
    from app.agents.learning_tracker.state import Question, QuizOutput

    monkeypatch.setattr(
        trig,
        "topic_digests",
        AsyncMock(
            return_value=digests
            if digests is not None
            else [
                {"sequence": 1, "bullets": ["old bullet 1"], "quizId": None},
                {"sequence": 2, "bullets": ["old bullet 2"], "quizId": "Q2"},
            ]
        ),
    )
    monkeypatch.setattr(
        trig,
        "fetch_roadmap",
        AsyncMock(return_value={"topics": [{"id": "t1", "title": "Ownership"}]}),
    )
    monkeypatch.setattr(
        trig, "get_profile", AsyncMock(return_value={"preferred_explanation_style": style})
    )
    monkeypatch.setattr(
        trig,
        "build_digest_quiz",
        AsyncMock(
            return_value=QuizOutput(
                quiz=[Question(question="new q", options=["a", "b"], answer=0)]
            )
        ),
    )
    monkeypatch.setattr(trig, "replace_quiz_questions", AsyncMock(return_value=True))
    col = _collection()
    monkeypatch.setattr(trig, "get_db", lambda: {trig.DIGESTS: col, "quizzes": col})
    monkeypatch.setattr(trig, "send_push_notification", AsyncMock())

    seen = {}

    async def _reteach(topic, bullets, style=None, missed=None):
        seen.update(topic=topic, bullets=list(bullets), style=style, missed=list(missed or []))
        return _tips_output(["a different angle"])

    monkeypatch.setattr(trig, "build_reteach", _reteach)
    return trig, seen, col


async def test_re_teaching_covers_the_material_the_failed_check_was_built_from(monkeypatch):
    """Not the whole topic. Anything extra makes a topic they are already stuck
    on bigger."""
    trig, seen, _ = _reteach_env(monkeypatch)

    await trig.escalate_to_reteach("userA", _OID, "t1", "Q2")

    assert seen["bullets"] == ["old bullet 1"]


async def test_re_teaching_uses_the_style_the_learner_asked_for(monkeypatch):
    """The one moment where "how do you like things explained" has something
    concrete to change."""
    trig, seen, _ = _reteach_env(monkeypatch, style="examples_first")
    await trig.escalate_to_reteach("userA", _OID, "t1", "Q2")
    assert seen["style"] == "examples_first"


async def test_re_teaching_is_aimed_at_the_questions_they_missed(monkeypatch):
    trig, seen, _ = _reteach_env(monkeypatch)
    await trig.escalate_to_reteach(
        "userA", _OID, "t1", "Q2", missed=["what does move do?"]
    )
    assert seen["missed"] == ["what does move do?"]


async def test_the_re_teach_digest_stays_outside_the_drip_feed(monkeypatch):
    """It covers ground already sent, so counting it would shift where the
    recall-check cadence falls for everything after."""
    trig, _, col = _reteach_env(monkeypatch)

    out = await trig.escalate_to_reteach("userA", _OID, "t1", "Q2")

    assert out["kind"] == "reteach"
    assert out["sequence"] is None
    assert out["quizId"] is None  # the gate is the check it re-explains


async def test_the_learner_still_has_to_pass_the_same_check(monkeypatch):
    """Fail-closed: escalation rewrites the questions in place rather than
    dropping the gate or adding a second one."""
    trig, _, _ = _reteach_env(monkeypatch)

    await trig.escalate_to_reteach("userA", _OID, "t1", "Q2")

    assert trig.replace_quiz_questions.await_args.args[0] == "Q2"
    rebuilt_from = trig.build_digest_quiz.await_args.args[1]
    # Over old material AND new, so it tests the same ideas while becoming
    # answerable from the new explanation.
    assert "old bullet 1" in rebuilt_from
    assert "a different angle" in rebuilt_from


async def test_nothing_to_re_explain_is_a_no_op(monkeypatch):
    trig, _, col = _reteach_env(monkeypatch, digests=[])
    assert await trig.escalate_to_reteach("userA", _OID, "t1", "Q2") is None
    col.insert_one.assert_not_awaited()


async def test_a_failed_quiz_rebuild_still_ships_the_new_explanation(monkeypatch):
    """The re-explanation is the valuable half; keeping the old questions is
    survivable, losing the new teaching is not."""
    trig, _, col = _reteach_env(monkeypatch)
    monkeypatch.setattr(
        trig, "build_digest_quiz", AsyncMock(side_effect=RuntimeError("boom"))
    )

    out = await trig.escalate_to_reteach("userA", _OID, "t1", "Q2")
    assert out is not None and out["bullets"] == ["a different angle"]


# --------------------------------------------------------------------------- #
# what a question looks like on the way out
# --------------------------------------------------------------------------- #
def test_a_question_leaves_without_its_answer_but_with_its_kind():
    """`kind` is not decoration: an `open` question has no options, and a client
    that can't tell the two apart waits for a tap that has nowhere to land — the
    digest then can't be marked, and an unmarkable digest jams the topic."""
    out = repo.public_question(
        {
            "question": "Why?",
            "options": [],
            "kind": "open",
            "answer": 0,
            "expected": "because …",
            "hint": "re-read the second tip",
            "outcome": "explains why",
        }
    )
    assert out == {"question": "Why?", "options": [], "kind": "open"}


def test_a_question_with_no_kind_reads_as_multiple_choice():
    assert repo.public_question({"question": "q", "options": ["a", "b"]})["kind"] == "choice"


async def test_the_check_on_a_listed_digest_keeps_its_kind(monkeypatch):
    """The read path is how a digest is answered after a reload, so a written
    question that arrives looking like a tap question can never be completed."""
    digest_col, quiz_col = _collection(), _collection()
    digest_col.find = MagicMock(
        return_value=MagicMock(
            sort=lambda *_: MagicMock(
                limit=lambda *_: MagicMock(
                    to_list=AsyncMock(
                        return_value=[{"_id": _OID, "roadmapId": "r1", "quizId": _OID}]
                    )
                )
            )
        )
    )
    quiz_col.find = MagicMock(
        return_value=MagicMock(
            to_list=AsyncMock(
                return_value=[
                    {
                        "_id": ObjectId(_OID),
                        "questions": [
                            {"question": "tap", "options": ["a", "b"], "answer": 1},
                            {
                                "question": "say why",
                                "options": [],
                                "kind": "open",
                                "expected": "because …",
                            },
                        ],
                    }
                ]
            )
        )
    )
    roadmap_col = _collection()
    roadmap_col.find = MagicMock(return_value=MagicMock(to_list=AsyncMock(return_value=[])))
    monkeypatch.setattr(
        repo,
        "get_db",
        lambda: {repo.DIGESTS: digest_col, "quizzes": quiz_col, "roadmaps": roadmap_col},
    )

    quiz = (await repo.list_digests("userA"))[0]["quiz"]

    assert [q["kind"] for q in quiz] == ["choice", "open"]
    # And nothing that would let the learner answer without reading.
    assert not any("answer" in q or "expected" in q for q in quiz)


async def test_a_freshly_generated_digest_does_not_ship_the_answer_key(monkeypatch):
    """POST /digests/generate skips the read projection, so the check used to be
    issued and given away in the same response."""
    trig, _ = _digest_gen(monkeypatch, prior=[{"bullets": ["earlier"]}])
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    _fake_tips(monkeypatch, trig)
    monkeypatch.setattr(trig, "build_digest_quiz", AsyncMock(return_value=_quiz_output()))

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert out["quiz"] and all(
        set(q) == {"question", "options", "kind"} for q in out["quiz"]
    )


# --------------------------------------------------------------------------- #
# the drip-feed's own numbering
# --------------------------------------------------------------------------- #
def test_revision_and_reteach_sit_outside_the_sequence():
    history = [
        {"sequence": 1},
        {"sequence": 2},
        {"kind": "reteach", "sequence": None},
        {"kind": "revision", "sequence": None},
    ]
    assert [d["sequence"] for d in repo.teaching_digests(history)] == [1, 2]


async def test_a_reteach_does_not_take_the_next_digests_number(monkeypatch):
    """It re-explains ground already sent. Counting it would move where the
    recall checks fall for everything after it — the cadence the whole gate is
    indexed on."""
    trig, _ = _digest_gen(
        monkeypatch,
        prior=[
            {"sequence": 1, "bullets": ["one"]},
            {"kind": "reteach", "sequence": None, "bullets": ["said differently"]},
        ],
    )
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False)))
    _fake_tips(monkeypatch, trig)
    monkeypatch.setattr(trig, "build_digest_quiz", AsyncMock(return_value=_quiz_output()))

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert out["sequence"] == 2


def test_the_gate_points_at_the_check_that_is_actually_outstanding():
    """With a re-teach in the history, counting positions instead of the
    drip-feed asked about the wrong digest — or let #4 through unchecked."""
    history = [
        {"sequence": 1, "status": "marked"},
        {"sequence": 2, "status": "unread"},  # carries the check
        {"kind": "reteach", "sequence": None, "status": "marked"},
        {"sequence": 3, "status": "marked"},
    ]
    assert repo.digest_quiz_gate(history) == 2

    history[1]["status"] = "marked"
    assert repo.digest_quiz_gate(history) is None


# --------------------------------------------------------------------------- #
# a checkpoint set is sat once
# --------------------------------------------------------------------------- #
async def test_a_graded_set_is_recognised(monkeypatch):
    col = _patch_db(monkeypatch, _collection())
    col.count_documents = AsyncMock(return_value=1)

    assert await repo.quiz_graded("userA", _OID) is True
    # Scoped to the caller, or one learner's attempt would close another's quiz.
    assert col.count_documents.await_args.args[0]["user_id"] == "userA"


async def test_an_ungraded_set_can_still_be_sat(monkeypatch):
    col = _patch_db(monkeypatch, _collection())
    col.count_documents = AsyncMock(return_value=0)
    assert await repo.quiz_graded("userA", _OID) is False


async def test_an_unreadable_history_refuses_the_re_grade(monkeypatch):
    """Fails closed: the alternative is a free re-run of a set whose misses the
    learner has already been shown, which is the completion gate opened from the
    outside."""
    col = _patch_db(monkeypatch, _collection())
    col.count_documents = AsyncMock(side_effect=RuntimeError("mongo down"))
    assert await repo.quiz_graded("userA", _OID) is True


# --------------------------------------------------------------------------- #
# what a roadmap owed revision offers
# --------------------------------------------------------------------------- #
_OWED = {"id": "t1", "order": 1, "title": "Ownership", "progress_status": "needs_review",
         "checkpoint_attempts": 1, "revisions_done": 0}


async def test_a_leftover_teaching_digest_is_not_the_revision_they_are_owed(monkeypatch):
    """`can_generate` has to say what pressing the button would actually do —
    and what it produces here is the revision digest, which doesn't exist yet."""
    _focus_db(
        monkeypatch,
        roadmap={"_id": _OID, "title": "Rust", "topics": [_OWED]},
        trigger=_ENABLED_TRIGGER,
        digests=[{"status": "unread", "sequence": 3}],
    )
    focus = await _focus_one()

    assert focus["blocked_reason"] == "needs_revision"
    assert focus["can_generate"] is True


async def test_revision_tips_already_waiting_are_not_generated_twice(monkeypatch):
    _focus_db(
        monkeypatch,
        roadmap={"_id": _OID, "title": "Rust", "topics": [_OWED]},
        trigger=_ENABLED_TRIGGER,
        digests=[{"status": "unread", "kind": "revision"}],
    )
    assert (await _focus_one())["can_generate"] is False


# --------------------------------------------------------------------------- #
# the written answer, end to end
#
# Until the read projection carried `kind`, no client could tell an open question
# from a tap one, so `written` was never sent and this whole path — the gate, the
# grader, the attempt it files — had never executed. These drive it through the
# real route, with only Mongo and the judge faked.
# --------------------------------------------------------------------------- #
def _mark_client(monkeypatch, *, questions, judgement=None, digest=None):
    """POST /learning/digests/{id}/mark against the real router.

    TestClient runs background tasks before returning, so `_record_written_answer`
    actually executes here rather than being asserted at second hand.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.dependencies import get_current_user
    from app.routers import learning_tracker as route

    _chain = _cursor
    digests, quizzes, attempts, roadmaps = (_collection() for _ in range(4))
    digests.find_one = AsyncMock(
        return_value={
            "_id": ObjectId(_OID),
            "user_id": "u1",
            "roadmapId": _OID,
            "topicId": "t1",
            "topicTitle": "Ownership",
            "quizId": _OID,
            "sequence": 4,
            "status": "unread",
            **(digest or {}),
        }
    )
    digests.find = MagicMock(return_value=_chain([]))
    quizzes.find_one = AsyncMock(
        return_value={"_id": ObjectId(_OID), "user_id": "u1", "questions": questions}
    )
    roadmaps.find_one = AsyncMock(
        return_value={"_id": ObjectId(_OID), "topics": [_started()]}
    )
    roadmaps.find = MagicMock(return_value=_chain([]))
    monkeypatch.setattr(
        repo,
        "get_db",
        lambda: {
            repo.DIGESTS: digests,
            "quizzes": quizzes,
            "quiz_attempts": attempts,
            "roadmaps": roadmaps,
        },
    )

    from app.agents.learning_tracker.state import ExplanationJudgement

    graded = AsyncMock(
        return_value=judgement or ExplanationJudgement(score=20, feedback="not quite")
    )
    monkeypatch.setattr(route, "grade_oneliner", graded)
    refreshed = AsyncMock()
    monkeypatch.setattr(route, "refresh_misconceptions", refreshed)

    app = FastAPI()
    app.include_router(route.router)
    app.dependency_overrides[get_current_user] = lambda: {"uid": "u1"}
    return TestClient(app), {
        "graded": graded,
        "refreshed": refreshed,
        "attempts": attempts,
        "digests": digests,
    }


_TAP = {"question": "Which is it?", "options": ["a", "b"], "answer": 1, "kind": "choice"}
_OPEN = {
    "question": "Say why in your own words",
    "options": [],
    "kind": "open",
    "answer": 0,
    "expected": "because the value moves",
}


def test_a_written_answer_reaches_the_grader_with_what_it_should_convey(monkeypatch):
    """`expected` never leaves the server, so the grader has to be handed it here
    — without it the judge marks a sentence against nothing."""
    client, spy = _mark_client(monkeypatch, questions=[_TAP, _OPEN])

    res = client.post(
        f"/learning/digests/{_OID}/mark",
        # Keyed by position, as JSON — which makes the keys strings on the wire.
        json={"answers": [{"question": 0, "answer": 1}], "written": {"1": "it moves"}},
    )

    assert res.status_code == 200
    question, expected, answer = spy["graded"].await_args.args
    assert question == "Say why in your own words"
    assert expected == "because the value moves"
    assert answer == "it moves"


def test_a_wrong_sentence_is_filed_as_evidence_not_as_a_refusal(monkeypatch):
    """The taps are the gate. A wrong sentence still marks the digest — and still
    reaches the misconception tracker, which is the reason for asking."""
    client, spy = _mark_client(monkeypatch, questions=[_TAP, _OPEN])

    res = client.post(
        f"/learning/digests/{_OID}/mark",
        json={"answers": [{"question": 0, "answer": 1}], "written": {"1": "no idea"}},
    )

    assert res.status_code == 200
    spy["digests"].update_one.assert_awaited()  # marked despite the wrong answer

    attempt = spy["attempts"].insert_one.await_args.args[0]
    assert attempt["kind"] == "written"
    assert attempt["score"] == 20
    assert attempt["correct"] == 0
    # Their sentence next to what the question wanted — richer evidence than any
    # distractor index, which is the whole point of asking in prose.
    assert attempt["misses"] == [
        {
            "question": "Say why in your own words",
            "chosen": "no idea",
            "correct": "because the value moves",
        }
    ]
    spy["refreshed"].assert_awaited()


def test_a_good_sentence_files_no_miss(monkeypatch):
    from app.agents.learning_tracker.state import ExplanationJudgement

    client, spy = _mark_client(
        monkeypatch,
        questions=[_TAP, _OPEN],
        judgement=ExplanationJudgement(score=90, feedback="yes"),
    )

    res = client.post(
        f"/learning/digests/{_OID}/mark",
        json={
            "answers": [{"question": 0, "answer": 1}],
            "written": {"1": "because the value moves"},
        },
    )

    assert res.status_code == 200
    attempt = spy["attempts"].insert_one.await_args.args[0]
    assert attempt["correct"] == 1 and attempt["misses"] == []
    spy["refreshed"].assert_not_awaited()  # nothing to diagnose


def test_a_blank_sentence_is_refused_and_the_digest_stays_unread(monkeypatch):
    """A blank is noise in the tracker rather than a reading of how someone
    thinks, so it has to be attempted — and the mark can't slip through anyway."""
    client, spy = _mark_client(monkeypatch, questions=[_TAP, _OPEN])

    res = client.post(
        f"/learning/digests/{_OID}/mark",
        json={"answers": [{"question": 0, "answer": 1}], "written": {"1": "   "}},
    )

    assert res.status_code == 422
    assert res.json()["detail"]["blocked_reason"] == "written_answer_required"
    spy["digests"].update_one.assert_not_awaited()
    spy["graded"].assert_not_awaited()


def test_failing_the_taps_stops_before_the_sentence_is_read(monkeypatch):
    """The recall check is the gate; grading a sentence for a learner who is
    being sent back anyway spends a model call on nothing."""
    client, spy = _mark_client(monkeypatch, questions=[_TAP, _OPEN])

    res = client.post(
        f"/learning/digests/{_OID}/mark",
        json={"answers": [{"question": 0, "answer": 0}], "written": {"1": "it moves"}},
    )

    assert res.status_code == 422
    assert res.json()["detail"]["quiz_result"]["score"] == 0
    spy["graded"].assert_not_awaited()
    spy["digests"].update_one.assert_not_awaited()


def test_a_check_with_nothing_to_tap_still_collects_the_sentence(monkeypatch):
    """An all-open set scores 0 against a pass mark of 100, so it can't be gated
    on — but the sentence is evidence and still has to be asked for."""
    client, spy = _mark_client(monkeypatch, questions=[_OPEN])

    blank = client.post(
        f"/learning/digests/{_OID}/mark", json={"answers": [], "written": {}}
    )
    assert blank.status_code == 422

    res = client.post(
        f"/learning/digests/{_OID}/mark",
        json={"answers": [], "written": {"0": "it moves"}},
    )
    assert res.status_code == 200
    assert res.json()["result"]["quiz_result"] is None  # nothing was gated on
    spy["graded"].assert_awaited()


# --------------------------------------------------------------------------- #
# the HTTP surface
#
# Everything above tests the repository and the agent. These go through the
# router itself, because that is where the refusals live: the gates on issuing
# and grading a checkpoint, the ownership checks on resuming a paused run, and
# the shape of what leaves the server. A gate that is correct in the repository
# and unreachable — or bypassable — through the route is not a gate.
# --------------------------------------------------------------------------- #
class _DB(dict):
    """A database double. A collection nobody set up answers empty rather than
    raising, so a test only has to describe the state it actually cares about."""

    def __missing__(self, name):
        col = _collection()
        col.find = MagicMock(return_value=_cursor())
        col.count_documents = AsyncMock(return_value=0)
        self[name] = col
        return col


def _api(monkeypatch, db=None, user="u1", agent=None):
    """The learning router, wired to a fake database and an authenticated caller.

    Both `get_db`s are patched: the repository reads through its own, and the
    router reaches for quizzes and triggers through the one imported into it.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.dependencies import get_current_user
    from app.routers import learning_tracker as route

    db = db if db is not None else _DB()
    monkeypatch.setattr(repo, "get_db", lambda: db)
    monkeypatch.setattr(route, "get_db", lambda: db)

    app = FastAPI()
    app.include_router(route.router)
    app.state.learning_agent = agent or MagicMock()
    app.dependency_overrides[get_current_user] = lambda: {"uid": user}
    return TestClient(app), db, route


def _quizzes(*docs):
    """A quizzes collection that honours the filters the router actually sends —
    `_id` and `kind` — so "which set comes back" is answered by the query rather
    than by the double."""
    col = _collection()

    async def find_one(filt, *a, **kw):
        for d in docs:
            kinds = (filt.get("kind") or {}).get("$in") if isinstance(filt.get("kind"), dict) else None
            if kinds and d.get("kind") not in kinds:
                continue
            if "_id" in filt and str(d["_id"]) != str(filt["_id"]):
                continue
            return dict(d)
        return None

    col.find_one = AsyncMock(side_effect=find_one)
    col.find = MagicMock(return_value=_cursor(docs))
    col.insert_one = AsyncMock(return_value=MagicMock(inserted_id=ObjectId(_OID)))
    return col


_Q = {"question": "Which is it?", "options": ["a", "b"], "answer": 1,
      "kind": "choice", "outcome": "tells them apart", "hint": "re-read the second tip"}


def _roadmap_doc(topic=None, **over):
    return {
        "_id": ObjectId(_OID),
        "user_id": "u1",
        "title": "Rust",
        "status": "active",
        "topics": [topic or {"id": "t1", "order": 1, "title": "Ownership",
                             "description": "…", "learning_outcomes": ["moves"],
                             "progress_status": "needs_review"}],
        **over,
    }


def _issue_env(monkeypatch, *, quizzes, topic=None, attempts=(), generated=None):
    from app.agents.learning_tracker.state import Question, QuizOutput

    db = _DB()
    db["roadmaps"].find_one = AsyncMock(return_value=_roadmap_doc(topic))
    db["quizzes"] = quizzes
    db["quiz_attempts"].find = MagicMock(return_value=_cursor(attempts))
    client, db, route = _api(monkeypatch, db)
    monkeypatch.setattr(route, "get_profile", AsyncMock(return_value={}))
    monkeypatch.setattr(
        route,
        "build_checkpoint",
        AsyncMock(
            return_value=generated
            or QuizOutput(quiz=[Question(question="fresh one", options=["a", "b"], answer=0)])
        ),
    )
    return client, db, route


def _checkpoint(topicId="t1"):
    return {"_id": ObjectId(_OID), "user_id": "u1", "roadmapId": _OID,
            "topicId": topicId, "kind": "checkpoint", "questions": [_Q]}


# ── issuing a checkpoint ────────────────────────────────────────────────────
def test_a_digest_check_is_never_handed_back_as_the_checkpoint(monkeypatch):
    """The newest quiz on a topic is just as often the digest's recall check —
    two questions about last week's tips. Completing a topic with it would make
    the gate meaningless."""
    digest_check = {"_id": ObjectId(_OID), "user_id": "u1", "roadmapId": _OID,
                    "topicId": "t1", "kind": "digest", "questions": [_Q]}
    client, _, route = _issue_env(monkeypatch, quizzes=_quizzes(digest_check))

    res = client.post(f"/learning/topics/t1/checkpoint", json={"roadmapId": _OID})

    assert res.status_code == 200
    # A fresh set was generated instead of the digest's being reused.
    route.build_checkpoint.assert_awaited()
    assert [q["question"] for q in res.json()["result"]["questions"]] == ["fresh one"]


def test_a_chat_quiz_is_not_a_checkpoint_either(monkeypatch):
    """`tutor_agent` stamps quizzes raised in chat with the topic underway and no
    kind at all."""
    chat_quiz = {"_id": ObjectId(_OID), "user_id": "u1", "roadmapId": _OID,
                 "topicId": "t1", "questions": [_Q]}
    client, _, route = _issue_env(monkeypatch, quizzes=_quizzes(chat_quiz))

    res = client.post(f"/learning/topics/t1/checkpoint", json={"roadmapId": _OID})

    assert res.status_code == 200
    route.build_checkpoint.assert_awaited()


def test_an_unfinished_checkpoint_is_resumed_rather_than_regenerated(monkeypatch):
    """Reopening the app mid-attempt is not a retry, and regenerating would cost
    an LLM call to replace a set they were halfway through."""
    client, _, route = _issue_env(monkeypatch, quizzes=_quizzes(_checkpoint()))

    res = client.post(f"/learning/topics/t1/checkpoint", json={"roadmapId": _OID})

    assert res.status_code == 200
    route.build_checkpoint.assert_not_awaited()
    assert res.json()["result"]["quizId"] == _OID


def test_an_issued_set_leaves_without_its_answers(monkeypatch):
    """Grading is server-side; a set that ships its own key grades itself."""
    client, _, _ = _issue_env(monkeypatch, quizzes=_quizzes(_checkpoint()))

    questions = client.post(
        f"/learning/topics/t1/checkpoint", json={"roadmapId": _OID}
    ).json()["result"]["questions"]

    assert questions and all(set(q) == {"question", "options"} for q in questions)


def test_a_topic_owed_revision_is_refused_a_retry(monkeypatch):
    """Without this, "retry" is re-rolling the same questions until they land by
    accident."""
    owed = {"id": "t1", "order": 1, "title": "Ownership", "description": "…",
            "progress_status": "needs_review", "checkpoint_attempts": 1,
            "revisions_done": 0, "weak_points": ["what moves on assignment"]}
    client, _, route = _issue_env(monkeypatch, quizzes=_quizzes(), topic=owed)

    res = client.post(f"/learning/topics/t1/checkpoint", json={"roadmapId": _OID})

    assert res.status_code == 409
    detail = res.json()["detail"]
    assert detail["blocked_reason"] == "needs_revision"
    # Named, so "go and revise" is something the learner can act on.
    assert detail["weak_points"] == ["what moves on assignment"]
    route.build_checkpoint.assert_not_awaited()  # refused before any spend


def test_the_daily_cap_is_refused_before_anything_is_generated(monkeypatch):
    from datetime import datetime, timezone

    from app.core.config import CHECKPOINT_MAX_ATTEMPTS_PER_DAY as CAP

    now = datetime.now(timezone.utc).isoformat()
    client, _, route = _issue_env(
        monkeypatch,
        quizzes=_quizzes(),
        attempts=[{"createdAt": now, "score": 10, "quizId": "old"} for _ in range(CAP)],
    )

    res = client.post(f"/learning/topics/t1/checkpoint", json={"roadmapId": _OID})

    assert res.status_code == 429
    detail = res.json()["detail"]
    assert detail["blocked_reason"] == "daily_limit"
    assert detail["retry_at"] and detail["limit"] == CAP
    # Refusing after paying for the call would burn it and still refuse.
    route.build_checkpoint.assert_not_awaited()


def test_a_missing_topic_is_a_404(monkeypatch):
    client, _, _ = _issue_env(monkeypatch, quizzes=_quizzes())
    res = client.post("/learning/topics/nope/checkpoint", json={"roadmapId": _OID})
    assert res.status_code == 404


# ── grading a checkpoint ────────────────────────────────────────────────────
def _submit_env(monkeypatch, *, already_graded=False, topic=None):
    db = _DB()
    db["quizzes"] = _quizzes(_checkpoint())
    db["roadmaps"].find_one = AsyncMock(return_value=_roadmap_doc(topic))
    db["quiz_attempts"].count_documents = AsyncMock(return_value=1 if already_graded else 0)
    client, db, route = _api(monkeypatch, db)
    monkeypatch.setattr(route, "refresh_misconceptions", AsyncMock())
    return client, db, route


def test_a_set_cannot_be_graded_twice(monkeypatch):
    """The cooldown and the daily cap are enforced where a set is issued. Without
    this the learner re-posts the same quizId with the misses they were just
    shown, and walks past both — and past the revision a failure owes."""
    client, db, _ = _submit_env(monkeypatch, already_graded=True)

    res = client.post(
        "/learning/checkpoint/submit",
        json={"quizId": _OID, "answers": [{"question": 0, "answer": 1}]},
    )

    assert res.status_code == 409
    assert res.json()["detail"]["blocked_reason"] == "already_graded"
    db["quiz_attempts"].insert_one.assert_not_awaited()
    db["roadmaps"].update_one.assert_not_awaited()  # nothing was applied


def test_a_failure_returns_the_hint_and_never_the_answer(monkeypatch):
    """Handing back `correctOption` on a failure turns the retry into
    transcription, which is exactly what a completion gate must not allow."""
    client, _, _ = _submit_env(monkeypatch)

    result = client.post(
        "/learning/checkpoint/submit",
        json={"quizId": _OID, "answers": [{"question": 0, "answer": 0}]},
    ).json()["result"]

    assert result["passed"] is False
    assert result["answers_revealed"] is False
    miss = result["review"][0]
    assert "correctAnswer" not in miss and "correctOption" not in miss
    assert miss["hint"] == "re-read the second tip"
    assert miss["outcome"] == "tells them apart"
    # The failure opened a revision debt, and says so — the client needs it to
    # stop offering a retry the server will refuse.
    assert result["needs_revision"] is True
    assert result["weak_points"] == ["Which is it?"]


def test_passing_completes_the_topic_and_returns_the_answers(monkeypatch):
    client, _, _ = _submit_env(monkeypatch)

    result = client.post(
        "/learning/checkpoint/submit",
        json={"quizId": _OID, "answers": [{"question": 0, "answer": 1}]},
    ).json()["result"]

    assert result["passed"] is True
    assert result["progress_status"] == "completed"
    assert result["answers_revealed"] is True
    assert result["next_review_at"]


def test_a_chat_quiz_cannot_be_submitted_as_a_checkpoint(monkeypatch):
    """It has no topic to attach a result to."""
    db = _DB()
    db["quizzes"] = _quizzes({"_id": ObjectId(_OID), "user_id": "u1", "questions": [_Q]})
    client, _, _ = _api(monkeypatch, db)

    res = client.post(
        "/learning/checkpoint/submit",
        json={"quizId": _OID, "answers": [{"question": 0, "answer": 1}]},
    )
    assert res.status_code == 404


# ── resuming someone else's paused run ──────────────────────────────────────
def _snapshot(user_id="u1", nxt=("roadmap_agent",)):
    snap = MagicMock()
    snap.next = nxt
    snap.values = {"user_id": user_id}
    return snap


def test_an_approval_cannot_be_resolved_by_another_learner(monkeypatch):
    """thread_ids are guessable; approving on someone else's thread would save a
    roadmap into their account — or reject one out of it."""
    agent = MagicMock()
    agent.aget_state = AsyncMock(return_value=_snapshot())
    agent.ainvoke = AsyncMock(return_value={})
    client, _, route = _api(monkeypatch, agent=agent)
    monkeypatch.setattr(
        route, "get_pending", AsyncMock(return_value={"user_id": "someone-else"})
    )

    res = client.post(
        "/learning/approvals", json={"thread_id": "t", "decision": "approved"}
    )

    assert res.status_code == 403
    agent.ainvoke.assert_not_awaited()


def test_an_approval_that_no_longer_exists_is_a_404(monkeypatch):
    client, _, route = _api(monkeypatch)
    monkeypatch.setattr(route, "get_pending", AsyncMock(return_value=None))

    res = client.post(
        "/learning/approvals", json={"thread_id": "t", "decision": "approved"}
    )
    assert res.status_code == 404


def test_onboarding_answers_cannot_be_posted_into_another_learners_thread(monkeypatch):
    """Onboarding raises no approval row, so the checkpointed user_id is the only
    identity the graph actually ran under."""
    agent = MagicMock()
    agent.aget_state = AsyncMock(return_value=_snapshot(user_id="someone-else"))
    agent.ainvoke = AsyncMock(return_value={})
    client, _, _ = _api(monkeypatch, agent=agent)

    res = client.post(
        "/learning/onboarding", json={"thread_id": "t", "answers": {"skill_level": "advanced"}}
    )

    assert res.status_code == 403
    agent.ainvoke.assert_not_awaited()


def test_nothing_waiting_on_a_thread_is_a_404(monkeypatch):
    agent = MagicMock()
    agent.aget_state = AsyncMock(return_value=_snapshot(nxt=()))
    client, _, _ = _api(monkeypatch, agent=agent)

    res = client.post("/learning/onboarding", json={"thread_id": "t", "answers": None})
    assert res.status_code == 404


# ── what else the routes refuse, and what they hand back ────────────────────
def test_a_topic_cannot_be_completed_by_asserting_it(monkeypatch):
    """Completion is earned by passing the checkpoint, or a finished roadmap means
    something was tapped rather than recalled."""
    client, db, _ = _api(monkeypatch)

    res = client.post(
        "/learning/progress",
        json={"roadmapId": _OID, "topicId": "t1", "status": "completed"},
    )

    assert res.status_code == 409
    db["roadmaps"].update_one.assert_not_awaited()


def test_reopening_a_topic_stays_free(monkeypatch):
    """The gate is on claiming knowledge, not on retracting it."""
    db = _DB()
    db["roadmaps"].find_one = AsyncMock(return_value=_roadmap_doc())
    client, _, _ = _api(monkeypatch, db)

    res = client.post(
        "/learning/progress",
        json={"roadmapId": _OID, "topicId": "t1", "status": "not_started"},
    )
    assert res.status_code == 200 and res.json()["progress_status"] == "not_started"


def test_the_misconception_report_withholds_how_it_will_be_probed(monkeypatch):
    """A learner who reads `probe` knows what the next question is coming for."""
    db = _DB()
    db[repo.MISCONCEPTIONS].find = MagicMock(
        return_value=_cursor(
            [{"roadmapId": _OID, "topicId": "t1", "misses_analyzed": 4,
              "patterns": [{"label": "moves vs copies", "detail": "…",
                            "probe": "ask what the old binding holds"}]}]
        )
    )
    client, _, _ = _api(monkeypatch, db)

    patterns = client.get("/learning/misconceptions").json()["result"][0]["patterns"]

    assert patterns[0]["label"] == "moves vs copies"
    assert "probe" not in patterns[0]


def test_an_explanation_withholds_the_probe_too(monkeypatch):
    """Same rule on the way out of the Feynman judge, which returns the same
    Misconception shape."""
    from app.agents.learning_tracker.state import ExplanationJudgement, Misconception

    db = _DB()
    db["roadmaps"].find_one = AsyncMock(return_value=_roadmap_doc())
    client, _, route = _api(monkeypatch, db)
    monkeypatch.setattr(route, "refresh_misconceptions", AsyncMock())
    monkeypatch.setattr(
        route,
        "judge_explanation",
        AsyncMock(
            return_value=ExplanationJudgement(
                score=80,
                feedback="clear",
                misconceptions=[
                    Misconception(label="moves vs copies", detail="…",
                                  probe="ask what the old binding holds")
                ],
            )
        ),
    )

    result = client.post(
        f"/learning/topics/t1/explain",
        json={"roadmapId": _OID, "text": " ".join(["word"] * 40)},
    ).json()["result"]

    assert result["passed"] is True and result["ladder_bonus"]
    assert "probe" not in result["misconceptions"][0]


def test_too_few_words_are_refused_rather_than_scored(monkeypatch):
    """Graded, this would produce a confident verdict about nothing and store it
    as evidence."""
    from app.core.config import FEYNMAN_MIN_WORDS

    db = _DB()
    db["roadmaps"].find_one = AsyncMock(return_value=_roadmap_doc())
    client, _, route = _api(monkeypatch, db)
    monkeypatch.setattr(route, "judge_explanation", AsyncMock())

    res = client.post(
        f"/learning/topics/t1/explain", json={"roadmapId": _OID, "text": "it moves"}
    )

    assert res.status_code == 422
    assert res.json()["detail"]["min_words"] == FEYNMAN_MIN_WORDS
    route.judge_explanation.assert_not_awaited()


def test_resuming_past_the_cap_names_what_is_holding_the_slots(monkeypatch):
    """The learner has to park one, and it isn't the server's call which."""
    from app.core.config import MAX_ACTIVE_ROADMAPS as CAP

    db = _DB()
    # Distinct ids: activating an already-active roadmap excludes itself from the
    # count, so holders sharing the id under test would read as a free slot.
    db["roadmaps"].find = MagicMock(
        return_value=_cursor(
            [{"_id": ObjectId(), "title": f"Running {i}"} for i in range(CAP)]
        )
    )
    client, _, _ = _api(monkeypatch, db)

    res = client.patch(f"/learning/roadmaps/{_OID}", json={"status": "active"})

    assert res.status_code == 409
    detail = res.json()["detail"]
    assert detail["limit"] == CAP
    assert [r["title"] for r in detail["active"]] == [f"Running {i}" for i in range(CAP)]


# --------------------------------------------------------------------------- #
# the floor under the drip-feed
#
# A live run turned this up: a two-outcome topic was declared covered by its very
# first digest, so the topic went straight to the checkpoint — and the recall
# check rides the SECOND digest, which meant it never had one. Acknowledging that
# topic's only digest meant "dismissed", not "read", and neither the written
# question nor the re-teach path could ever fire.
# --------------------------------------------------------------------------- #
def test_coverage_cannot_end_the_feed_before_the_first_check():
    from app.core.config import DIGEST_MIN_BEFORE_CHECKPOINT as FLOOR

    assert repo.coverage_may_end(FLOOR - 1) is False
    assert repo.coverage_may_end(FLOOR) is True
    # The guarantee the floor exists for: it is never below the check cadence, or
    # a topic could still reach the checkpoint without one.
    from app.core.config import DIGEST_QUIZ_EVERY

    assert FLOOR >= DIGEST_QUIZ_EVERY


async def test_a_narrow_topic_still_gets_its_recall_check(monkeypatch):
    """Covered on the first digest is a correct answer to the wrong question."""
    trig, _ = _digest_gen(monkeypatch)
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(True)))
    _fake_tips(monkeypatch, trig)

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert out is not None and out["sequence"] == 1
    # Held, not lost: the topic stays underway so the next digest — the one that
    # carries the check — can still be generated.
    trig.set_topic_progress.assert_not_awaited()
    # And the card must not offer the checkpoint while the server intends to send
    # one more, or the two disagree on screen.
    assert out["coverage_complete"] is False


async def test_the_floor_lets_go_once_the_check_has_been_sent(monkeypatch):
    trig, _ = _digest_gen(monkeypatch, prior=[{"bullets": ["the first digest"]}])
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(True)))
    _fake_tips(monkeypatch, trig)
    monkeypatch.setattr(trig, "build_digest_quiz", AsyncMock(return_value=_quiz_output()))

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert out["sequence"] == 2
    assert out["quizId"]  # the check the floor was bought for
    assert out["coverage_complete"] is True
    assert trig.set_topic_progress.await_args.args[2] == "needs_review"


async def test_the_floor_does_not_hold_a_topic_that_is_still_being_taught(monkeypatch):
    """It delays the handover; it never invents extra digests once coverage is
    genuinely incomplete — that path is unchanged."""
    trig, _ = _digest_gen(monkeypatch)
    monkeypatch.setattr(trig, "check_coverage", AsyncMock(return_value=repo_coverage(False, ["x"])))
    _fake_tips(monkeypatch, trig)

    out = await trig.build_digest("userA", {"_id": _OID, "topics": [_started()]}, notify=False)

    assert out["coverage_complete"] is False and out["missing_outcomes"] == ["x"]
    trig.set_topic_progress.assert_not_awaited()


# --------------------------------------------------------------------------- #
# indexes
# --------------------------------------------------------------------------- #
def test_every_declared_index_leads_with_the_owner():
    """Each of these queries is scoped by user_id first. An index that doesn't
    lead with it can't serve the query it was written for."""
    from app.agents.learning_tracker import indexes

    for collection, keys, name, why in indexes.INDEXES:
        assert keys[0] == ("user_id", 1), f"{name} does not lead with user_id"
        assert name.startswith("learning_"), f"{name} is not namespaced"
        assert why, f"{name} has no stated purpose"


def test_index_names_are_unique():
    """They double as the idempotency key on a re-run."""
    from app.agents.learning_tracker import indexes

    names = [n for _, _, n, _ in indexes.INDEXES]
    assert len(names) == len(set(names))


async def test_a_failed_index_creation_does_not_stop_the_app(monkeypatch):
    """This runs during startup. A missing index is a slow app, not a broken one,
    and certainly not a reason to refuse to boot."""
    from app.agents.learning_tracker import indexes

    col = _collection()
    col.create_index = AsyncMock(side_effect=RuntimeError("no permission"))
    monkeypatch.setattr(indexes, "get_db", lambda: {c: col for c, *_ in indexes.INDEXES})

    assert await indexes.ensure_indexes() == 0


async def test_indexes_are_created_for_every_collection_the_feature_owns(monkeypatch):
    from app.agents.learning_tracker import indexes

    seen = []
    col = _collection()

    async def _create(keys, name=None, background=None):
        seen.append(name)

    col.create_index = AsyncMock(side_effect=_create)
    monkeypatch.setattr(indexes, "get_db", lambda: {c: col for c, *_ in indexes.INDEXES})

    created = await indexes.ensure_indexes()

    assert created == len(indexes.INDEXES)
    assert seen == [n for _, _, n, _ in indexes.INDEXES]


async def test_a_digest_missing_its_body_does_not_blank_the_queue(monkeypatch):
    """The card maps over bullets and resources without checking."""
    _digest_db(
        monkeypatch,
        digests=[{"_id": _OID, "roadmapId": "r1"}],
        roadmaps=[{"_id": "r1", "title": "Rust", "topics": []}],
    )
    d = (await repo.list_digests("userA"))[0]
    assert d["bullets"] == [] and d["resources"] == []


def test_every_explained_shape_has_an_index_that_could_serve_it():
    """The two lists are written out separately on purpose — this is what notices
    when a call site changes shape and drifts off the index built for it."""
    from app.agents.learning_tracker import indexes

    by_collection: dict[str, list[list]] = {}
    for collection, keys, *_ in indexes.INDEXES:
        by_collection.setdefault(collection, []).append([k for k, _ in keys])

    for label, collection, query, _sort in indexes._SHAPES:
        fields = set(query)
        assert any(
            fields.issubset(set(keys)) for keys in by_collection.get(collection, [])
        ), f"{label}: no declared index covers {sorted(fields)} on {collection}"


# --------------------------------------------------------------------------- #
# what completing a topic does to its leftovers
#
# Digests are a nudge to study the topic you're on. Passing its checkpoint proves
# the material by a harder route than the tips were softening, so anything still
# waiting on that topic has nothing left to ask for. Left open they nagged from
# the catch-up queue — and marking one late graded a recall check against a
# completed topic, denting a mastery score already earned, and could escalate to
# a re-teach digest for a topic nobody was studying any more.
# --------------------------------------------------------------------------- #
def _closing_db(monkeypatch, topic=None, closed=2):
    """A roadmap that completes on a pass, and a digest collection that reports
    how many were still waiting."""
    roadmaps, digests = _collection(), _collection()
    roadmaps.find_one = AsyncMock(
        return_value={
            "_id": _OID,
            "status": "active",
            "topics": [topic or _started(progress_status="needs_review")],
        }
    )
    digests.update_many = AsyncMock(return_value=MagicMock(modified_count=closed))
    monkeypatch.setattr(
        repo, "get_db", lambda: {"roadmaps": roadmaps, repo.DIGESTS: digests}
    )
    return roadmaps, digests


async def test_passing_closes_the_digests_still_waiting_on_that_topic(monkeypatch):
    _, digests = _closing_db(monkeypatch)

    out = await repo.apply_checkpoint(_OID, "t1", "userA", 100)

    assert out["digests_closed"] == 2
    filt, update = digests.update_many.await_args.args
    # Scoped to the caller and to this topic — one learner completing a topic
    # must not close another's queue, or their own other topics.
    assert filt["user_id"] == "userA"
    assert filt["topicId"] == "t1"
    assert filt["status"] == {"$ne": "marked"}
    assert update["$set"]["status"] == "marked"
    # Marked, but not pretending it was read: `status: "marked"` is what
    # `digest_quiz_gate` reads as "the check was passed", and these were closed
    # by the checkpoint rather than answered.
    assert update["$set"]["closed_by"] == "checkpoint"


async def test_failing_leaves_the_queue_alone(monkeypatch):
    """The topic is still being studied, and those tips are still the way through
    it."""
    _, digests = _closing_db(monkeypatch)

    out = await repo.apply_checkpoint(_OID, "t1", "userA", 10)

    assert out["digests_closed"] == 0
    digests.update_many.assert_not_awaited()


async def test_a_review_of_a_finished_topic_closes_nothing(monkeypatch):
    """Only the transition closes the feed. A topic completed weeks ago has no
    drip-feed left to close, and a spaced review isn't a second completion."""
    _, digests = _closing_db(
        monkeypatch, topic=_started(progress_status="completed", completed_at="x")
    )

    out = await repo.apply_checkpoint(_OID, "t1", "userA", 100)

    assert out["was_review"] is True and out["digests_closed"] == 0
    digests.update_many.assert_not_awaited()


async def test_a_failed_tidy_up_does_not_undo_the_pass(monkeypatch):
    """The checkpoint has already been applied. A queue that couldn't be tidied is
    a stale nudge, not a topic that failed to complete."""
    _, digests = _closing_db(monkeypatch)
    digests.update_many = AsyncMock(side_effect=RuntimeError("mongo down"))

    out = await repo.apply_checkpoint(_OID, "t1", "userA", 100)

    assert out["passed"] is True and out["progress_status"] == "completed"
    assert out["digests_closed"] == 0


async def test_the_count_reaches_the_client(monkeypatch):
    """Declared on CheckpointOutcome or pydantic drops it, and the queue would sit
    there until something happened to refetch."""
    db = _DB()
    db["quizzes"] = _quizzes(_checkpoint())
    db["roadmaps"].find_one = AsyncMock(return_value=_roadmap_doc())
    db[repo.DIGESTS].update_many = AsyncMock(return_value=MagicMock(modified_count=3))
    client, _, route = _api(monkeypatch, db)
    monkeypatch.setattr(route, "refresh_misconceptions", AsyncMock())

    result = client.post(
        "/learning/checkpoint/submit",
        json={"quizId": _OID, "answers": [{"question": 0, "answer": 1}]},
    ).json()["result"]

    assert result["passed"] is True
    assert result["digests_closed"] == 3


# --------------------------------------------------------------------------- #
# the briefing: the agent speaking first
#
# The judgement is a model's and isn't tested here. What is tested is the part
# that must hold regardless of what the model says: a briefing may never offer an
# action the server would refuse, and it must still answer when the model doesn't.
# --------------------------------------------------------------------------- #

from app.agents.learning_tracker.context import situation_key, situation_text
from app.agents.learning_tracker.service import (
    _briefing_action_allowed,
    fallback_briefing,
)
from app.agents.learning_tracker.state import BriefingAction


def _ctx(**overrides):
    base = {
        "roadmaps": [
            {
                "roadmapId": "r1",
                "title": "Rust",
                "topic": "Ownership",
                "topicId": "t3",
                "topic_status": "needs_review",
                "completed": 2,
                "total": 21,
                "percent": 10,
                "unread": 0,
                "can_generate": True,
                "blocked_reason": None,
                "blocked_meaning": None,
            }
        ],
        "unread_digests": 0,
        "reviews_due": [],
        "reviews_due_total": 0,
        "misconceptions": [],
        "misconceptions_total": 0,
        "mastery": None,
        "streak_days": None,
        "profile": {},
    }
    base.update(overrides)
    return base


def test_briefing_never_offers_a_checkpoint_that_owes_revision():
    """The server refuses the attempt until the revision digest is read, so
    offering it would be the dead end the rest of the app avoids by hand."""
    ctx = _ctx()
    action = BriefingAction(
        kind="open_checkpoint", label="Take it", roadmapId="r1", topicId="t3"
    )
    assert _briefing_action_allowed(action, ctx) is True

    ctx["roadmaps"][0]["blocked_reason"] = "needs_revision"
    assert _briefing_action_allowed(action, ctx) is False


def test_briefing_never_offers_a_checkpoint_on_an_unfinished_topic():
    ctx = _ctx()
    ctx["roadmaps"][0]["topic_status"] = "in_progress"
    action = BriefingAction(
        kind="open_checkpoint", label="Take it", roadmapId="r1", topicId="t3"
    )
    assert _briefing_action_allowed(action, ctx) is False


def test_briefing_respects_can_generate():
    """The same flag the home screen's button reads, so the two can't disagree
    about whether anything can be pulled."""
    ctx = _ctx()
    action = BriefingAction(
        kind="generate_digest", label="Pull one", roadmapId="r1", topicId="t3"
    )
    assert _briefing_action_allowed(action, ctx) is True

    ctx["roadmaps"][0]["can_generate"] = False
    assert _briefing_action_allowed(action, ctx) is False


def test_briefing_rejects_an_invented_roadmap():
    """A model that hallucinates an id must not produce a button that goes
    somewhere wrong."""
    action = BriefingAction(kind="open_roadmap", label="Go", roadmapId="nope")
    assert _briefing_action_allowed(action, _ctx()) is False


def test_briefing_only_offers_reviews_when_one_is_due():
    ctx = _ctx()
    action = BriefingAction(kind="open_reviews", label="Review")
    assert _briefing_action_allowed(action, ctx) is False

    ctx["reviews_due"] = [{"topicId": "t1", "title": "Borrowing", "overdue_days": 2}]
    assert _briefing_action_allowed(action, ctx) is True


def test_briefing_only_offers_roadmap_creation_when_there_is_none():
    action = BriefingAction(kind="create_roadmap", label="Build one")
    assert _briefing_action_allowed(action, _ctx()) is False
    assert _briefing_action_allowed(action, _ctx(roadmaps=[])) is True


def test_fallback_briefing_leads_with_an_overdue_review():
    """Model unavailable is not the same as nothing to say. The fallback follows
    the same priority order the prompt asks for."""
    ctx = _ctx(
        reviews_due=[{"topicId": "t1", "title": "Borrowing", "overdue_days": 3}],
        reviews_due_total=1,
    )
    out = fallback_briefing(ctx)
    assert "Borrowing" in out["headline"]
    assert out["actions"][0]["kind"] == "open_reviews"


def test_fallback_briefing_sends_a_new_learner_to_build_a_roadmap():
    out = fallback_briefing(_ctx(roadmaps=[]))
    assert out["actions"][0]["kind"] == "create_roadmap"


def test_fallback_briefing_always_answers():
    """Every branch produces a headline and a detail — an empty coach is worse
    than a plain one."""
    for ctx in (_ctx(), _ctx(roadmaps=[]), _ctx(unread_digests=2)):
        out = fallback_briefing(ctx)
        assert out["headline"] and out["detail"]


def test_situation_key_ignores_the_clock():
    """A briefing holds while the learner's position holds. Keying it on anything
    that ticks would expire advice that is still correct."""
    a = _ctx()
    b = _ctx()
    b["next_digest_at"] = "2030-01-01T09:00:00Z"
    assert situation_key(a) == situation_key(b)


def test_situation_key_moves_when_the_position_does():
    a = _ctx()
    b = _ctx()
    b["roadmaps"][0]["blocked_reason"] = "needs_revision"
    assert situation_key(a) != situation_key(b)


def test_situation_text_never_leaks_the_probe():
    """`probe` describes how a misunderstanding will be re-tested. Every consumer
    of this text writes something the learner reads."""
    ctx = _ctx(
        misconceptions=[
            {
                "topic": "Ownership",
                "roadmap": "Rust",
                "label": "Move equals copy",
                "detail": "Believes assignment copies.",
            }
        ]
    )
    text = situation_text(ctx)
    assert "Move equals copy" in text
    assert "probe" not in text.lower()


# --------------------------------------------------------------------------- #
# the action tools: the tutor doing things rather than describing buttons
#
# The safety property is the one that has to hold no matter what the model asks
# for: no tool may open a gate. A digest is acknowledged by passing its recall
# check, a topic is completed by passing its checkpoint, and neither is reachable
# from chat — otherwise every mastery score in the app is a claim about nothing.
# --------------------------------------------------------------------------- #

import app.agents.learning_tracker.actions as lt_actions


def _tools(user_id="userA"):
    return {t.name: t for t in lt_actions.build_action_tools(user_id)}


def test_no_tool_can_open_a_gate():
    """The whole product rests on these three being unreachable from chat."""
    names = set(_tools())
    forbidden = {
        "mark_digest_read",
        "mark_digest",
        "complete_topic",
        "set_topic_complete",
        "submit_checkpoint",
        "answer_check",
        "submit_quiz",
        "delete_roadmap",
    }
    assert names & forbidden == set()


def test_no_tool_accepts_a_user_id():
    """Tools close over the learner instead of taking one as an argument, so a
    model has no parameter through which to reach another learner's data."""
    for name, tool in _tools().items():
        fields = set(tool.args_schema.model_fields)
        assert "user_id" not in fields, name
        assert "userId" not in fields, name


async def test_start_topic_says_what_it_displaced(monkeypatch):
    """Exactly one topic is underway, so starting one gives another up. A learner
    told only "started X" has not been told the whole truth."""
    roadmap = {
        "_id": ObjectId(),
        "title": "Rust",
        "topics": [
            {"id": "t1", "order": 1, "title": "Ownership", "progress_status": "in_progress"},
            {"id": "t2", "order": 2, "title": "Borrowing", "progress_status": "not_started"},
        ],
    }
    monkeypatch.setattr(lt_actions, "fetch_roadmap", AsyncMock(return_value=roadmap))
    monkeypatch.setattr(lt_actions, "start_topic", AsyncMock(return_value=True))

    out = await _tools()["start_topic"].ainvoke({"roadmapId": "r1", "topicId": "t2"})
    assert "Borrowing" in out
    assert "Ownership" in out


async def test_start_topic_refuses_an_unknown_topic(monkeypatch):
    roadmap = {"_id": ObjectId(), "title": "Rust", "topics": []}
    monkeypatch.setattr(lt_actions, "fetch_roadmap", AsyncMock(return_value=roadmap))
    started = AsyncMock(return_value=True)
    monkeypatch.setattr(lt_actions, "start_topic", started)

    out = await _tools()["start_topic"].ainvoke({"roadmapId": "r1", "topicId": "nope"})
    assert "No topic" in out
    started.assert_not_awaited()


async def test_pull_next_lesson_relays_a_refusal_verbatim(monkeypatch):
    """A gate that stops the drip-feed has to reach the learner as the reason it
    is, not as a generic failure — and the tool must not retry around it."""
    monkeypatch.setattr(
        lt_actions,
        "pull_next_digest",
        AsyncMock(
            return_value={
                "refused": "Pass the recall check on digest #2 first — it's on the digest itself.",
                "reason": "awaiting_quiz",
            }
        ),
    )
    out = await _tools()["pull_next_lesson"].ainvoke({})
    assert "recall check" in out


async def test_pull_next_lesson_reports_a_delivered_digest(monkeypatch):
    monkeypatch.setattr(
        lt_actions,
        "pull_next_digest",
        AsyncMock(
            return_value={
                "digest": {"topicTitle": "Ownership", "bullets": ["a", "b", "c"], "quiz": None}
            }
        ),
    )
    out = await _tools()["pull_next_lesson"].ainvoke({})
    assert "Ownership" in out
    assert "3 points" in out


async def test_resume_explains_the_active_cap(monkeypatch):
    """Being over the cap is a choice to make, not an error to report."""
    monkeypatch.setattr(
        lt_actions, "fetch_roadmap", AsyncMock(return_value={"title": "Go"})
    )
    monkeypatch.setattr(
        lt_actions,
        "set_roadmap_status",
        AsyncMock(side_effect=repo.ActiveRoadmapLimit(2, [{"title": "Rust"}, {"title": "SQL"}])),
    )
    out = await _tools()["resume_roadmap"].ainvoke({"roadmapId": "r9"})
    assert "Rust" in out and "SQL" in out
    assert "paused first" in out


async def test_set_digest_time_rejects_a_bad_hour(monkeypatch):
    """The validation lives in one place and both callers report it their own way;
    the tool's way is a sentence."""
    out = await _tools()["set_digest_time"].ainvoke({"hour": 99})
    assert "0-23" in out


async def test_list_topics_reports_ids_so_actions_need_no_guessing(monkeypatch):
    roadmap = {
        "_id": ObjectId(),
        "title": "Rust",
        "topics": [{"id": "t7", "order": 1, "title": "Ownership", "progress_status": "in_progress"}],
    }
    monkeypatch.setattr(lt_actions, "fetch_roadmap", AsyncMock(return_value=roadmap))
    out = await _tools()["list_topics"].ainvoke({"roadmapId": "r1"})
    assert "id=t7" in out
    assert "in_progress" in out


def test_take_action_routes_to_a_node_that_exists():
    assert lt.INTENT_ROUTES["take_action"] in set(lt.build_graph().nodes)
