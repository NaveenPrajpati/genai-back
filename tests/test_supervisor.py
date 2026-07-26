"""Tests for the supervisor graph and its HTTP surface.

Offline: the LLM, the MCP client, the approval store and the compiled graph are
all replaced with fakes. What is worth pinning down here is the orchestration —
queue dispatch, replay safety, the approval gate, and identity checks — not the
model's wording.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app.agents.supervisor.workflow as sup
import app.routers.supervisor as sup_router
from app.agents.supervisor.state import merge_results
from app.dependencies import get_current_user


def _make_client(agent, user=None):
    app = FastAPI()
    app.include_router(sup_router.router)
    app.state.supervisor_agent = agent
    app.dependency_overrides[get_current_user] = lambda: user or {"uid": "u1"}
    return TestClient(app)


def _tool_message(name, payload_text):
    """A tool result shaped the way MCP actually returns one: a list of typed
    content blocks, not a bare string."""
    return ToolMessage(
        name=name,
        tool_call_id=f"call_{name}",
        content=[{"type": "text", "text": payload_text}],
    )


# --------------------------------------------------------------------------- #
# dispatch: walking the skill queue
# --------------------------------------------------------------------------- #
def test_dispatch_runs_queue_in_order_then_finalizes():
    state = {"route": ["meal", "assistant"], "completed": []}
    assert sup.dispatch(state) == "meal_agent"

    state["completed"] = ["meal"]
    assert sup.dispatch(state) == "assistant_agent"

    state["completed"] = ["meal", "assistant"]
    assert sup.dispatch(state) == "finalize"


def test_dispatch_finalizes_when_no_skill_is_needed():
    assert sup.dispatch({"route": [], "completed": []}) == "finalize"
    assert sup.dispatch({}) == "finalize"


def test_dispatch_never_reruns_a_completed_skill():
    # The guarantee that makes resume-after-approval cheap: a skill already in
    # `completed` is skipped even if it is still listed in `route`.
    state = {"route": ["learning", "meal"], "completed": ["learning"]}
    assert sup.dispatch(state) == "meal_agent"


def test_merge_results_accumulates_across_skills():
    assert merge_results({"meal": 1}, {"assistant": 2}) == {"meal": 1, "assistant": 2}
    assert merge_results(None, {"meal": 1}) == {"meal": 1}
    assert merge_results({"meal": 1}, None) == {"meal": 1}


# --------------------------------------------------------------------------- #
# decoding MCP results
# --------------------------------------------------------------------------- #
def test_tool_result_decodes_mcp_content_blocks():
    # MCP wraps results in content blocks; the JSON lives in the text block.
    message = _tool_message("propose_meal_plan", '{"slots": [1, 2], "saved": false}')
    assert sup._tool_result(message) == {"slots": [1, 2], "saved": False}


def test_tool_result_handles_plain_strings_and_non_json():
    assert sup._tool_result(AIMessage(content='{"a": 1}')) == {"a": 1}
    assert sup._tool_result(AIMessage(content="just text")) == "just text"


def test_tool_result_accepts_a_direct_invoke_return():
    # save_meal_plan is called directly, not through a ToolMessage.
    assert sup._tool_result([{"type": "text", "text": '{"plan_id": "p1"}'}]) == {
        "plan_id": "p1"
    }


def test_last_proposal_picks_the_most_recent_and_ignores_other_tools():
    messages = [
        _tool_message("list_meal_plans", "[]"),
        _tool_message("propose_meal_plan", '{"week_start": "2026-01-05", "slots": [1]}'),
        _tool_message("get_food_preferences", '{"diet": "vegetarian"}'),
        _tool_message("propose_meal_plan", '{"week_start": "2026-01-12", "slots": [2]}'),
    ]
    assert sup._last_proposal(messages)["week_start"] == "2026-01-12"


def test_last_proposal_ignores_a_proposal_with_no_slots():
    messages = [_tool_message("propose_meal_plan", '{"slots": []}')]
    assert sup._last_proposal(messages) is None


def test_last_proposal_returns_none_without_any_proposal():
    assert sup._last_proposal([_tool_message("list_meal_plans", "[]")]) is None


# --------------------------------------------------------------------------- #
# meal skill: MCP + the approval gate
# --------------------------------------------------------------------------- #
def _patch_meal(monkeypatch, *, tools, pending=None, loop_messages=None):
    monkeypatch.setattr(sup, "load_meal_tools", AsyncMock(return_value=tools))
    monkeypatch.setattr(sup, "get_pending", AsyncMock(return_value=pending))
    monkeypatch.setattr(sup, "create_pending", AsyncMock(return_value="appr1"))
    monkeypatch.setattr(sup, "resolve", AsyncMock())
    monkeypatch.setattr(sup, "autonomous_tools", lambda t: t)
    monkeypatch.setattr(sup, "ToolNode", lambda t: MagicMock())
    monkeypatch.setattr(sup, "llm", MagicMock())
    if loop_messages is not None:
        monkeypatch.setattr(sup, "run_tool_loop", AsyncMock(return_value=loop_messages))


def _save_tool():
    tool = MagicMock()
    tool.name = "save_meal_plan"
    tool.ainvoke = AsyncMock(
        return_value=[{"type": "text", "text": '{"plan_id": "plan-9", "saved": true}'}]
    )
    return tool


_STATE = {"user_id": "u1", "thread_id": "t1", "query": "plan my week"}


async def test_meal_skill_degrades_when_the_mcp_server_is_down(monkeypatch):
    # A dead MCP dependency must cost the meal skill, not the whole turn.
    _patch_meal(monkeypatch, tools=[])
    out = await sup.meal_agent(dict(_STATE))
    assert out["completed"] == ["meal"]
    assert "error" in out["results"]["meal"]


async def test_meal_skill_without_a_proposal_does_not_ask_for_approval(monkeypatch):
    # Read-only turns (e.g. "what's my grocery list?") must not gate.
    called = AsyncMock()
    monkeypatch.setattr(sup, "interrupt", called)
    _patch_meal(
        monkeypatch,
        tools=[_save_tool()],
        loop_messages=[AIMessage(content="Here is your list.")],
    )
    out = await sup.meal_agent(dict(_STATE))
    called.assert_not_called()
    assert out["results"]["meal"]["summary"] == "Here is your list."


async def test_meal_skill_saves_over_mcp_only_after_approval(monkeypatch):
    save = _save_tool()
    monkeypatch.setattr(sup, "interrupt", lambda _payload: "approved")
    _patch_meal(
        monkeypatch,
        tools=[save],
        loop_messages=[
            _tool_message(
                "propose_meal_plan",
                '{"week_start": "2026-01-05", "slots": [{"recipe_name": "dal"}]}',
            ),
            AIMessage(content="Proposed a week of meals."),
        ],
    )
    out = await sup.meal_agent(dict(_STATE))

    save.ainvoke.assert_awaited_once()
    assert save.ainvoke.await_args.args[0]["week_start"] == "2026-01-05"
    sup.resolve.assert_awaited_once_with("appr1", "approved")
    assert out["results"]["meal"]["plan_status"] == "approved and saved"
    assert out["plan_id"] == "plan-9"


async def test_meal_skill_writes_nothing_when_the_user_rejects(monkeypatch):
    save = _save_tool()
    monkeypatch.setattr(sup, "interrupt", lambda _payload: "rejected")
    _patch_meal(
        monkeypatch,
        tools=[save],
        loop_messages=[
            _tool_message("propose_meal_plan", '{"week_start": "w", "slots": [1]}'),
            AIMessage(content="Proposed."),
        ],
    )
    out = await sup.meal_agent(dict(_STATE))

    save.ainvoke.assert_not_awaited()
    sup.resolve.assert_awaited_once_with("appr1", "rejected")
    assert out["results"]["meal"]["plan_status"] == "rejected by user"


async def test_meal_skill_reuses_the_stored_proposal_on_replay(monkeypatch):
    """Resuming replays this node from the top. It must reuse the proposal the
    user actually approved instead of paying for a second tool loop and saving a
    different plan."""
    save = _save_tool()
    monkeypatch.setattr(sup, "interrupt", lambda _payload: "approved")
    _patch_meal(
        monkeypatch,
        tools=[save],
        pending={
            "_id": "appr-existing",
            "action_type": sup.MEAL_APPROVAL,
            "payload": {
                "proposal": {"week_start": "2026-01-05", "slots": [{"r": 1}]},
                "transcript": "earlier summary",
            },
        },
    )
    monkeypatch.setattr(
        sup, "run_tool_loop", AsyncMock(side_effect=AssertionError("re-ran the loop"))
    )

    out = await sup.meal_agent(dict(_STATE))

    sup.create_pending.assert_not_awaited()
    sup.resolve.assert_awaited_once_with("appr-existing", "approved")
    assert out["results"]["meal"]["summary"] == "earlier summary"


async def test_meal_skill_ignores_a_pending_approval_from_another_skill(monkeypatch):
    """A roadmap approval sitting on this thread must not be mistaken for a meal
    proposal — the node should run its own loop instead."""
    monkeypatch.setattr(sup, "interrupt", lambda _payload: "approved")
    _patch_meal(
        monkeypatch,
        tools=[_save_tool()],
        pending={"_id": "a2", "action_type": "save_roadmap", "payload": {}},
        loop_messages=[AIMessage(content="ran the loop")],
    )
    out = await sup.meal_agent(dict(_STATE))
    assert out["results"]["meal"]["summary"] == "ran the loop"


# --------------------------------------------------------------------------- #
# subgraph skills
# --------------------------------------------------------------------------- #
async def test_learning_skill_summarizes_the_subgraph_result(monkeypatch):
    subgraph = MagicMock()
    subgraph.ainvoke = AsyncMock(
        return_value={
            "intent": "create_roadmap",
            "roadmap_status": "approved",
            "roadmapId": "r1",
            "roadmap": {"title": "Rust", "topics": [{"id": "t1"}, {"id": "t2"}]},
            "pa_tasks_created": 2,
        }
    )
    monkeypatch.setattr(sup, "learning_subgraph", subgraph)

    out = await sup.learning_agent({**_STATE, "roadmapId": None})
    result = out["results"]["learning"]
    assert out["completed"] == ["learning"]
    assert result["roadmap_title"] == "Rust"
    assert result["topic_count"] == 2
    assert out["roadmapId"] == "r1"


async def test_rejected_roadmap_reports_no_topic_count(monkeypatch):
    # A literal 0 here reads to the finalizer as "produced an empty roadmap".
    subgraph = MagicMock()
    subgraph.ainvoke = AsyncMock(
        return_value={"intent": "create_roadmap", "roadmap_status": "rejected"}
    )
    monkeypatch.setattr(sup, "learning_subgraph", subgraph)

    result = (await sup.learning_agent(dict(_STATE)))["results"]["learning"]
    assert "topic_count" not in result
    assert result["roadmap_status"] == "rejected"


async def test_assistant_skill_passes_through_the_pa_prose(monkeypatch):
    subgraph = MagicMock()
    subgraph.ainvoke = AsyncMock(
        return_value={
            "intent": "add",
            "task_status": "added",
            "response": "Added buy milk.",
            "todos": [{"title": "buy milk", "priority": "medium", "due_at": None}],
        }
    )
    monkeypatch.setattr(sup, "assistant_subgraph", subgraph)

    result = (await sup.assistant_agent(dict(_STATE)))["results"]["assistant"]
    assert result["summary"] == "Added buy milk."
    assert result["tasks"] == [
        {"title": "buy milk", "priority": "medium", "due": None}
    ]


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #
def test_query_returns_the_approval_proposal_when_a_skill_pauses():
    agent = MagicMock()
    agent.ainvoke = AsyncMock(
        return_value={"__interrupt__": [MagicMock(value={"type": "save_roadmap"})]}
    )
    response = _make_client(agent).post("/assistant/query", json={"text": "learn rust"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_approval"
    assert body["proposal"] == {"type": "save_roadmap"}
    assert body["thread_id"]


def test_query_strips_graph_internals_and_the_message_log():
    agent = MagicMock()
    agent.ainvoke = AsyncMock(
        return_value={
            "response": "done",
            "messages": [HumanMessage(content="hi")],
            "__root__": "internal",
        }
    )
    result = _make_client(agent).post("/assistant/query", json={"text": "hi"}).json()

    assert result["result"] == {"response": "done"}


def test_approve_rejects_another_users_approval(monkeypatch):
    monkeypatch.setattr(
        sup_router, "get_pending", AsyncMock(return_value={"user_id": "someone-else"})
    )
    agent = MagicMock()
    agent.ainvoke = AsyncMock()

    response = _make_client(agent).post(
        "/assistant/approve", json={"thread_id": "t1", "decision": "approved"}
    )

    assert response.status_code == 403
    agent.ainvoke.assert_not_awaited()


def test_approve_404s_when_nothing_is_pending(monkeypatch):
    monkeypatch.setattr(sup_router, "get_pending", AsyncMock(return_value=None))
    response = _make_client(MagicMock()).post(
        "/assistant/approve", json={"thread_id": "t1", "decision": "approved"}
    )
    assert response.status_code == 404


def test_approve_404s_when_the_thread_is_no_longer_paused(monkeypatch):
    # e.g. the server restarted and the checkpoint is gone.
    monkeypatch.setattr(
        sup_router, "get_pending", AsyncMock(return_value={"user_id": "u1"})
    )
    agent = MagicMock()
    agent.aget_state = AsyncMock(return_value=MagicMock(next=()))
    response = _make_client(agent).post(
        "/assistant/approve", json={"thread_id": "t1", "decision": "approved"}
    )
    assert response.status_code == 404


def test_approve_surfaces_a_second_approval_in_a_multi_skill_turn(monkeypatch):
    monkeypatch.setattr(
        sup_router, "get_pending", AsyncMock(return_value={"user_id": "u1"})
    )
    agent = MagicMock()
    agent.aget_state = AsyncMock(return_value=MagicMock(next=("meal_agent",), values={}))
    agent.ainvoke = AsyncMock(
        return_value={"__interrupt__": [MagicMock(value={"type": "meal"})]}
    )
    body = (
        _make_client(agent)
        .post("/assistant/approve", json={"thread_id": "t1", "decision": "approved"})
        .json()
    )
    assert body["status"] == "needs_approval"
    assert body["proposal"] == {"type": "meal"}


# --------------------------------------------------------------------------- #
# memory extraction follows the skills that actually ran
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "route, expected",
    [
        (["meal"], 1),
        (["meal", "assistant"], 2),
        ([], 0),  # a pure-chat turn learns nothing and pays for nothing
    ],
)
def test_memory_extraction_scales_with_the_skills_used(route, expected):
    tasks = MagicMock()
    sup_router._schedule_memory(tasks, "u1", "text", {"route": route, "memory": {}})
    assert tasks.add_task.call_count == expected
