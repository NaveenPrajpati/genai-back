"""Tests for the personal-assistant multi-agent system and its endpoints.

Offline: the LLM, the Supabase client, MongoDB helpers, the search API, and the
compiled LangGraph agent are all replaced with fakes.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END

# Agent logic now lives in the app.agents.personal_assistant package; the router
# is a thin HTTP layer. `pa` points at the graph module (which imports every
# node/repository/state symbol the node tests patch); endpoint and trigger tests
# patch their own modules so monkeypatch hits the namespace the code looks up.
import app.agents.personal_assistant.workflow as pa
import app.agents.personal_assistant.triggers as pa_triggers
import app.routers.personal_assistant as pa_router
from app.dependencies import get_current_user


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_CHAIN_METHODS = (
    "select", "eq", "in_", "ilike", "order", "limit",
    "maybe_single", "insert", "update", "upsert",
)


def _chainable(execute_data):
    q = MagicMock()
    for meth in _CHAIN_METHODS:
        getattr(q, meth).return_value = q
    q.execute.return_value = SimpleNamespace(data=execute_data)
    supa = MagicMock()
    supa.table.return_value = q
    return supa


def _supabase_seq(*results):
    q = MagicMock()
    for meth in _CHAIN_METHODS:
        getattr(q, meth).return_value = q
    q.execute.side_effect = [SimpleNamespace(data=d) for d in results]
    supa = MagicMock()
    supa.table.return_value = q
    return supa


def _llm_returning(model_obj):
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(
        lambda _: model_obj
    )
    return mock_llm


def _make_client(agent, user=None):
    app = FastAPI()
    app.include_router(pa_router.router)
    app.state.pa_agent = agent
    app.dependency_overrides[get_current_user] = lambda: user or {"uid": "u1"}
    return TestClient(app)


# --------------------------------------------------------------------------- #
# intent classifier + routing
# --------------------------------------------------------------------------- #
async def test_classify_intent(monkeypatch):
    monkeypatch.setattr(pa, "llm", _llm_returning(pa.IntentOutput(intent="research")))
    out = await pa.classify_intent({"query": "what is langgraph"})
    assert out == {"intent": "research"}


def test_decide_agent_routes():
    for i in ("add", "list", "complete", "delete", "update"):
        assert pa.decide_agent({"intent": i}) == "todo_agent"
    assert pa.decide_agent({"intent": "research"}) == "research_agent"
    for i in ("note", "recall"):
        assert pa.decide_agent({"intent": i}) == "notes_agent"
    assert pa.decide_agent({"intent": "agenda"}) == "agenda_agent"
    assert pa.decide_agent({"intent": "breakdown"}) == "breakdown_agent"
    assert pa.decide_agent({"intent": "huh"}) == END
    assert pa.decide_agent({}) == END


# --------------------------------------------------------------------------- #
# todo_agent (MongoDB to-do management)
# --------------------------------------------------------------------------- #
async def test_todo_agent_add(monkeypatch):
    monkeypatch.setattr(
        pa, "llm", _llm_returning(pa.TaskInput(title="Buy milk", priority="high"))
    )
    monkeypatch.setattr(
        pa, "insert_todo", AsyncMock(return_value={"id": "t1", "title": "Buy milk"})
    )
    out = await pa.todo_agent(
        {"intent": "add", "user_id": "u1", "query": "remind me to buy milk"}
    )
    assert out["task_status"] == "added"
    assert out["todos"][0]["title"] == "Buy milk"
    pa.insert_todo.assert_awaited_once()


async def test_todo_agent_list(monkeypatch):
    monkeypatch.setattr(
        pa, "fetch_todos", AsyncMock(return_value=[{"id": "t1", "title": "x"}])
    )
    out = await pa.todo_agent(
        {"intent": "list", "user_id": "u1", "query": "what's on my list"}
    )
    assert out["task_status"] == "listed"
    pa.fetch_todos.assert_awaited_once_with("u1", status="pending")


async def test_todo_agent_complete(monkeypatch):
    monkeypatch.setattr(
        pa, "_extract_selector", AsyncMock(return_value=pa.TaskSelector(title="report"))
    )
    monkeypatch.setattr(
        pa, "complete_todo", AsyncMock(return_value={"id": "t1", "title": "report"})
    )
    monkeypatch.setattr(pa, "append_memory_list", AsyncMock())
    monkeypatch.setattr(pa, "fetch_todos", AsyncMock(return_value=[]))
    out = await pa.todo_agent(
        {"intent": "complete", "user_id": "u1", "query": "finished the report"}
    )
    assert out["task_status"] == "completed"
    pa.append_memory_list.assert_awaited_once_with(
        "u1", pa.MEMORY_COMPLETED_KEY, "report"
    )


async def test_todo_agent_complete_not_found(monkeypatch):
    monkeypatch.setattr(
        pa, "_extract_selector", AsyncMock(return_value=pa.TaskSelector(title="nope"))
    )
    monkeypatch.setattr(pa, "complete_todo", AsyncMock(return_value=None))
    out = await pa.todo_agent(
        {"intent": "complete", "user_id": "u1", "query": "done nope"}
    )
    assert out["task_status"] == "not_found"


# --------------------------------------------------------------------------- #
# Human-in-the-Loop deletion
# --------------------------------------------------------------------------- #
async def test_delete_not_found_skips_interrupt(monkeypatch):
    monkeypatch.setattr(pa, "supabase", _supabase_seq(None))  # no existing approval
    monkeypatch.setattr(
        pa, "_extract_selector", AsyncMock(return_value=pa.TaskSelector(match_all=True))
    )
    monkeypatch.setattr(pa, "find_pending_todos", AsyncMock(return_value=[]))
    # interrupt must never be called when there's nothing to delete
    monkeypatch.setattr(
        pa, "interrupt", MagicMock(side_effect=AssertionError("should not interrupt"))
    )
    out = await pa._delete_with_approval(
        {"user_id": "u1", "thread_id": "t1", "query": "delete everything"}
    )
    assert out["task_status"] == "not_found"


async def test_delete_approved_path(monkeypatch):
    # execute() sequence: existing-lookup, approval-insert, approval-update
    monkeypatch.setattr(pa, "supabase", _supabase_seq(None, [{"id": "ap1"}], None))
    monkeypatch.setattr(
        pa, "_extract_selector", AsyncMock(return_value=pa.TaskSelector(match_all=True))
    )
    monkeypatch.setattr(
        pa,
        "find_pending_todos",
        AsyncMock(return_value=[{"id": "t1", "title": "a"}, {"id": "t2", "title": "b"}]),
    )
    monkeypatch.setattr(pa, "interrupt", lambda payload: "approved")
    monkeypatch.setattr(pa, "delete_todos_by_ids", AsyncMock(return_value=2))
    monkeypatch.setattr(pa, "fetch_todos", AsyncMock(return_value=[]))

    out = await pa._delete_with_approval(
        {"user_id": "u1", "thread_id": "t1", "query": "delete all"}
    )

    assert out["task_status"] == "deleted:2"
    pa.delete_todos_by_ids.assert_awaited_once_with("u1", ["t1", "t2"])


async def test_delete_rejected_path(monkeypatch):
    monkeypatch.setattr(pa, "supabase", _supabase_seq(None, [{"id": "ap1"}], None))
    monkeypatch.setattr(
        pa, "_extract_selector", AsyncMock(return_value=pa.TaskSelector(match_all=True))
    )
    monkeypatch.setattr(
        pa, "find_pending_todos", AsyncMock(return_value=[{"id": "t1", "title": "a"}])
    )
    monkeypatch.setattr(pa, "interrupt", lambda payload: "rejected")
    monkeypatch.setattr(pa, "delete_todos_by_ids", AsyncMock())

    out = await pa._delete_with_approval(
        {"user_id": "u1", "thread_id": "t1", "query": "delete all"}
    )

    assert out["task_status"] == "delete_rejected"
    pa.delete_todos_by_ids.assert_not_awaited()


# --------------------------------------------------------------------------- #
# research_agent + search API
# --------------------------------------------------------------------------- #
async def test_research_agent(monkeypatch):
    monkeypatch.setattr(
        pa,
        "web_search",
        AsyncMock(
            return_value={
                "answer": "ans",
                "results": [{"title": "t", "url": "u", "content": "c"}],
            }
        ),
    )
    monkeypatch.setattr(
        pa,
        "llm",
        _llm_returning(
            pa.ResearchOutput(summary="s", key_points=["k1", "k2"], sources=["u"])
        ),
    )
    monkeypatch.setattr(pa, "append_memory_list", AsyncMock())

    out = await pa.research_agent({"user_id": "u1", "query": "langgraph"})

    assert out["research"]["summary"] == "s"
    assert out["suggestions"] == ["k1", "k2"]
    pa.append_memory_list.assert_awaited_once_with(
        "u1", pa.MEMORY_RESEARCH_KEY, "langgraph"
    )


async def test_web_search_without_key_is_empty(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    out = await pa.web_search("anything")
    assert out == {"answer": "", "results": []}


# --------------------------------------------------------------------------- #
# notes_agent (personal facts)
# --------------------------------------------------------------------------- #
async def test_notes_agent_note(monkeypatch):
    monkeypatch.setattr(
        pa, "llm", _llm_returning(pa.NoteInput(content="wife's birthday is June 2"))
    )
    monkeypatch.setattr(pa, "add_note", AsyncMock())
    monkeypatch.setattr(
        pa,
        "fetch_notes",
        AsyncMock(return_value=[{"content": "wife's birthday is June 2"}]),
    )
    out = await pa.notes_agent(
        {"intent": "note", "user_id": "u1", "query": "remember my wife's birthday"}
    )
    assert out["task_status"] == "noted"
    assert out["notes"][0]["content"] == "wife's birthday is June 2"
    pa.add_note.assert_awaited_once()


async def test_notes_agent_recall(monkeypatch):
    monkeypatch.setattr(
        pa, "fetch_notes", AsyncMock(return_value=[{"content": "likes tea"}])
    )
    out = await pa.notes_agent(
        {"intent": "recall", "user_id": "u1", "query": "what do you know about me"}
    )
    assert out["task_status"] == "recalled"
    assert out["notes"][0]["content"] == "likes tea"


# --------------------------------------------------------------------------- #
# agenda_agent (due-date awareness)
# --------------------------------------------------------------------------- #
async def test_agenda_agent(monkeypatch):
    monkeypatch.setattr(
        pa,
        "fetch_todos",
        AsyncMock(
            return_value=[
                {"title": "old", "due_at": "2000-01-01"},
                {"title": "later", "due_at": "2999-01-01"},
                {"title": "someday"},
            ]
        ),
    )
    out = await pa.agenda_agent({"intent": "agenda", "user_id": "u1"})
    assert out["task_status"] == "agenda"
    assert out["agenda"]["overdue"][0]["title"] == "old"
    assert out["agenda"]["upcoming"][0]["title"] == "later"
    assert out["agenda"]["no_date"][0]["title"] == "someday"


# --------------------------------------------------------------------------- #
# breakdown_agent (subtask generation)
# --------------------------------------------------------------------------- #
async def test_breakdown_agent(monkeypatch):
    monkeypatch.setattr(
        pa,
        "llm",
        _llm_returning(
            pa.BreakdownOutput(parent_title="Plan trip", subtasks=["book flight", "pack"])
        ),
    )
    monkeypatch.setattr(
        pa, "insert_todo", AsyncMock(return_value={"id": "p1", "title": "Plan trip"})
    )
    monkeypatch.setattr(
        pa,
        "insert_subtasks",
        AsyncMock(return_value=[{"id": "s1"}, {"id": "s2"}]),
    )
    out = await pa.breakdown_agent(
        {"intent": "breakdown", "user_id": "u1", "query": "help me plan a trip"}
    )
    assert out["task_status"] == "broke_down:2"
    assert out["todos"][0]["title"] == "Plan trip"
    assert len(out["subtasks"]) == 2
    pa.insert_subtasks.assert_awaited_once_with("u1", "p1", ["book flight", "pack"])


# --------------------------------------------------------------------------- #
# endpoints: approve / resume cycle, tasks
# --------------------------------------------------------------------------- #
def test_pa_approve_happy_path(monkeypatch):
    monkeypatch.setattr(pa_router, "supabase", _chainable({"id": "ap1", "user_id": "u1"}))
    agent = MagicMock()
    agent.aget_state = AsyncMock(return_value=SimpleNamespace(next=("todo_agent",)))
    agent.ainvoke = AsyncMock(return_value={"task_status": "deleted:1"})
    client = _make_client(agent)

    resp = client.post(
        "/personal-assistant/approve",
        json={"thread_id": "t1", "decision": "approved"},
    )

    assert resp.status_code == 200
    assert resp.json()["result"]["task_status"] == "deleted:1"
    agent.ainvoke.assert_awaited_once()


def test_pa_approve_foreign_thread_forbidden(monkeypatch):
    monkeypatch.setattr(
        pa_router, "supabase", _chainable({"id": "ap1", "user_id": "someone_else"})
    )
    agent = MagicMock()
    agent.aget_state = AsyncMock()
    agent.ainvoke = AsyncMock()
    client = _make_client(agent)

    resp = client.post(
        "/personal-assistant/approve",
        json={"thread_id": "t1", "decision": "approved"},
    )

    assert resp.status_code == 403
    agent.ainvoke.assert_not_awaited()


def test_pa_approve_no_pending_is_404(monkeypatch):
    monkeypatch.setattr(pa_router, "supabase", _chainable(None))
    agent = MagicMock()
    agent.aget_state = AsyncMock()
    agent.ainvoke = AsyncMock()
    client = _make_client(agent)

    resp = client.post(
        "/personal-assistant/approve",
        json={"thread_id": "missing", "decision": "approved"},
    )

    assert resp.status_code == 404
    agent.ainvoke.assert_not_awaited()


def test_pa_tasks_endpoint(monkeypatch):
    monkeypatch.setattr(
        pa_router, "fetch_todos", AsyncMock(return_value=[{"id": "t1", "title": "x"}])
    )
    client = _make_client(MagicMock())

    resp = client.get("/personal-assistant/tasks")

    assert resp.status_code == 200
    assert resp.json()["result"][0]["title"] == "x"


def test_pa_agenda_endpoint(monkeypatch):
    monkeypatch.setattr(
        pa_router,
        "fetch_todos",
        AsyncMock(return_value=[{"title": "old", "due_at": "2000-01-01"}]),
    )
    client = _make_client(MagicMock())

    resp = client.get("/personal-assistant/tasks/agenda")

    assert resp.status_code == 200
    body = resp.json()["result"]
    assert body["counts"]["overdue"] == 1
    assert body["buckets"]["overdue"][0]["title"] == "old"


def test_pa_notes_endpoints(monkeypatch):
    monkeypatch.setattr(
        pa_router, "fetch_notes", AsyncMock(return_value=[{"content": "likes tea"}])
    )
    monkeypatch.setattr(
        pa_router, "add_note", AsyncMock(return_value={"content": "new fact"})
    )
    client = _make_client(MagicMock())

    get_resp = client.get("/personal-assistant/notes")
    assert get_resp.status_code == 200
    assert get_resp.json()["result"][0]["content"] == "likes tea"

    post_resp = client.post(
        "/personal-assistant/notes", json={"content": "new fact"}
    )
    assert post_resp.status_code == 200
    assert post_resp.json()["result"]["content"] == "new fact"


def test_pa_subtasks_endpoint(monkeypatch):
    monkeypatch.setattr(
        pa_router, "fetch_subtasks", AsyncMock(return_value=[{"id": "s1", "title": "step"}])
    )
    client = _make_client(MagicMock())

    resp = client.get("/personal-assistant/tasks/p1/subtasks")

    assert resp.status_code == 200
    assert resp.json()["result"][0]["title"] == "step"


def test_categorize_agenda_buckets():
    from app.agents.personal_assistant.repository import categorize_agenda

    buckets = categorize_agenda(
        [
            {"title": "overdue", "due_at": "2000-01-01"},
            {"title": "future", "due_at": "2999-12-31"},
            {"title": "undated"},
        ]
    )
    assert [t["title"] for t in buckets["overdue"]] == ["overdue"]
    assert [t["title"] for t in buckets["upcoming"]] == ["future"]
    assert [t["title"] for t in buckets["no_date"]] == ["undated"]


# --------------------------------------------------------------------------- #
# automation engine
# --------------------------------------------------------------------------- #
async def test_run_pa_triggers_creates_digest(monkeypatch):
    # execute() sequence: triggers-select, approvals-insert, triggers-update
    monkeypatch.setattr(
        pa_triggers,
        "supabase",
        _supabase_seq([{"id": "tr1", "user_id": "u1"}], None, None),
    )
    monkeypatch.setattr(
        pa_triggers, "fetch_todos", AsyncMock(return_value=[{"id": "t1", "title": "x"}])
    )

    await pa_triggers.run_pa_triggers()

    pa_triggers.fetch_todos.assert_awaited_once_with("u1", status="pending")
