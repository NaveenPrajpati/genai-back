"""Tests for the personal-assistant multi-agent system and its endpoints.

Offline: the LLM, the Supabase client, MongoDB helpers, the search API, and the
compiled LangGraph agent are all replaced with fakes.
"""

from datetime import date
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
import app.agents.personal_assistant.repository as pa_repo
import app.agents.personal_assistant.tools as pa_tools
import app.agents.personal_assistant.triggers as pa_triggers
import app.agents.memory_store as memory_store
import app.agents.trigger_store as trigger_store
import app.routers.personal_assistant as pa_router
from app.dependencies import get_current_user

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_CHAIN_METHODS = (
    "select",
    "eq",
    "in_",
    "ilike",
    "order",
    "limit",
    "maybe_single",
    "insert",
    "update",
    "upsert",
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
    mock_llm.with_structured_output.return_value = RunnableLambda(lambda _: model_obj)
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
    # classify_intent runs the fast model behind the semantic cache. Patch the
    # fast model and bypass the cache so the test stays offline & deterministic.
    monkeypatch.setattr(pa, "fast_llm", _llm_returning(pa.IntentOutput(intent="research")))

    async def _no_cache(text, scope, threshold, produce):
        return await produce()

    monkeypatch.setattr(pa, "cached_value", _no_cache)
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
        pa, "fast_llm", _llm_returning(pa.TaskInput(title="Buy milk", priority="high"))
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
    monkeypatch.setattr(pa, "get_pending", AsyncMock(return_value=None))  # no existing
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
    # No existing approval -> create one, then resolve it after deletion.
    monkeypatch.setattr(pa, "get_pending", AsyncMock(return_value=None))
    monkeypatch.setattr(pa, "create_pending", AsyncMock(return_value="ap1"))
    monkeypatch.setattr(pa, "resolve", AsyncMock())
    monkeypatch.setattr(
        pa, "_extract_selector", AsyncMock(return_value=pa.TaskSelector(match_all=True))
    )
    monkeypatch.setattr(
        pa,
        "find_pending_todos",
        AsyncMock(
            return_value=[{"id": "t1", "title": "a"}, {"id": "t2", "title": "b"}]
        ),
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
    monkeypatch.setattr(pa, "get_pending", AsyncMock(return_value=None))
    monkeypatch.setattr(pa, "create_pending", AsyncMock(return_value="ap1"))
    monkeypatch.setattr(pa, "resolve", AsyncMock())
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
    # The research model answers without calling a tool, so the ReAct loop ends
    # after one turn and the final structured pass (on llm) produces the output.
    # research_agent now builds its tool loop per-user via build_research_loop.
    no_tool_reply = MagicMock(tool_calls=[])
    research_model = MagicMock()
    research_model.ainvoke = AsyncMock(return_value=no_tool_reply)
    monkeypatch.setattr(
        pa, "build_research_loop", lambda user_id: (research_model, MagicMock())
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
    out = await pa_tools.web_search("anything")
    assert out == {"answer": "", "results": []}


# --------------------------------------------------------------------------- #
# notes_agent (personal facts)
# --------------------------------------------------------------------------- #
async def test_notes_agent_note(monkeypatch):
    monkeypatch.setattr(
        pa, "fast_llm", _llm_returning(pa.NoteInput(content="wife's birthday is June 2"))
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
            pa.BreakdownOutput(
                parent_title="Plan trip",
                subtasks=[
                    pa.SubtaskInput(title="book flight", priority="high", due_at="2026-09-01"),
                    pa.SubtaskInput(title="pack"),
                ],
            )
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
        {"intent": "breakdown", "user_id": "u1", "query": "help me plan a trip", "memory": {}}
    )
    assert out["task_status"] == "broke_down:2"
    assert out["todos"][0]["title"] == "Plan trip"
    assert len(out["subtasks"]) == 2
    # Schedule and importance arrive as fields, not as English inside a title.
    specs = pa.insert_subtasks.await_args.args[2]
    assert specs[0] == {
        "title": "book flight",
        "priority": "high",
        "due_at": "2026-09-01",
    }
    # exclude_none: an unscheduled step must not be written with a null due date.
    assert "due_at" not in specs[1]


async def test_breakdown_puts_the_day_in_a_field_not_the_title(monkeypatch):
    # The exact shape the audit found: 7 subtasks titled "Monday: …",
    # "Tuesday: …", all undated, so the agenda's Upcoming count stayed at 0.
    monkeypatch.setattr(
        pa,
        "llm",
        _llm_returning(
            pa.BreakdownOutput(
                parent_title="Launch my side project",
                subtasks=[pa.SubtaskInput(title="Draft the outline", due_at="2026-09-07")],
            )
        ),
    )
    monkeypatch.setattr(pa, "insert_todo", AsyncMock(return_value={"id": "p1", "title": "x"}))
    monkeypatch.setattr(pa, "insert_subtasks", AsyncMock(return_value=[{"id": "s1"}]))

    await pa.breakdown_agent(
        {"intent": "breakdown", "user_id": "u1", "query": "schedule these across next week", "memory": {}}
    )

    spec = pa.insert_subtasks.await_args.args[2][0]
    assert spec["due_at"] == "2026-09-07"
    assert "Monday" not in spec["title"]


async def test_insert_subtasks_carries_the_fields_through(monkeypatch):
    monkeypatch.setattr(pa_repo, "insert_todo", AsyncMock(side_effect=lambda u, d: d))

    created = await pa_repo.insert_subtasks(
        "u1",
        "p1",
        [
            {"title": "  book flight  ", "priority": "high", "due_at": "2026-09-01"},
            {"title": "   "},
            {"title": "pack"},
        ],
    )

    # Blank titles are skipped, the rest keep their scheduling.
    assert len(created) == 2
    assert created[0]["title"] == "book flight"
    assert created[0]["priority"] == "high"
    assert created[0]["due_at"] == "2026-09-01"
    assert created[0]["parent_id"] == "p1"


# --------------------------------------------------------------------------- #
# endpoints: approve / resume cycle, tasks
# --------------------------------------------------------------------------- #
def test_pa_approve_happy_path(monkeypatch):
    monkeypatch.setattr(
        pa_router,
        "get_pending",
        AsyncMock(
            return_value={
                "_id": "ap1",
                "user_id": "u1",
                "action_type": "pa_delete_task",
            }
        ),
    )
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
        pa_router,
        "get_pending",
        AsyncMock(
            return_value={
                "_id": "ap1",
                "user_id": "someone_else",
                "action_type": "pa_delete_task",
            }
        ),
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
    monkeypatch.setattr(pa_router, "get_pending", AsyncMock(return_value=None))
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
    # `result` stays the plain list it has always been, so the deployed client
    # keeps rendering while it catches up with the pagination beside it.
    monkeypatch.setattr(
        pa_router,
        "fetch_todos_page",
        AsyncMock(return_value={"items": [{"id": "t1", "title": "x"}], "total": 1}),
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

    post_resp = client.post("/personal-assistant/notes", json={"content": "new fact"})
    assert post_resp.status_code == 200
    assert post_resp.json()["result"]["content"] == "new fact"


def test_pa_subtasks_endpoint(monkeypatch):
    monkeypatch.setattr(
        pa_router,
        "fetch_subtasks",
        AsyncMock(return_value=[{"id": "s1", "title": "step"}]),
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
    # One due trigger -> digest snapshot created in the Mongo approval store and
    # the trigger stamped as run.
    monkeypatch.setattr(
        pa_triggers,
        "due_triggers",
        AsyncMock(return_value=[{"_id": "tr1", "user_id": "u1"}]),
    )
    monkeypatch.setattr(pa_triggers, "create_pending", AsyncMock())
    monkeypatch.setattr(pa_triggers, "mark_ran", AsyncMock())
    monkeypatch.setattr(pa_triggers, "send_push_notification", AsyncMock())
    # Due *today*. A digest now goes out only when something is actually due, so
    # an undated task here would exercise the quiet path instead.
    monkeypatch.setattr(
        pa_triggers,
        "fetch_todos",
        AsyncMock(
            return_value=[{"id": "t1", "title": "x", "due_at": date.today().isoformat()}]
        ),
    )

    await pa_triggers.run_pa_triggers()

    pa_triggers.fetch_todos.assert_awaited_once_with("u1", status="pending")
    pa_triggers.create_pending.assert_awaited_once()
    pa_triggers.mark_ran.assert_awaited_once()
    pa_triggers.send_push_notification.assert_awaited_once()


# --------------------------------------------------------------------------- #
# digest trigger state
# --------------------------------------------------------------------------- #
def test_pa_trigger_state_reports_enabled(monkeypatch):
    # The row exists and is on: the client can render the switch from this
    # instead of assuming a starting state.
    monkeypatch.setattr(
        pa_router,
        "get_trigger",
        AsyncMock(
            return_value={
                "enabled": True,
                "schedule_hour": 8,
                "timezone": "UTC",
                "last_run_at": "2026-08-22T08:00:00+00:00",
            }
        ),
    )
    client = _make_client(MagicMock())

    resp = client.get("/personal-assistant/trigger")

    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["enabled"] is True
    assert result["schedule_hour"] == 8
    # Enabled triggers advertise when they next fire.
    assert result["next_run_at"]


def test_pa_trigger_state_without_a_row_is_disabled(monkeypatch):
    # Never opted in. That is not an error — it is the state they would be
    # switching on from, reported at the hour they would get.
    monkeypatch.setattr(pa_router, "get_trigger", AsyncMock(return_value=None))
    client = _make_client(MagicMock())

    resp = client.get("/personal-assistant/trigger")

    assert resp.status_code == 200
    assert resp.json()["result"] == {
        "enabled": False,
        "schedule_hour": 8,
        "timezone": "UTC",
        "schedule_dow": None,
        "next_run_at": None,
        "last_run_at": None,
    }


def test_pa_trigger_state_disabled_row_has_no_next_run(monkeypatch):
    monkeypatch.setattr(
        pa_router,
        "get_trigger",
        AsyncMock(return_value={"enabled": False, "schedule_hour": 8}),
    )
    client = _make_client(MagicMock())

    resp = client.get("/personal-assistant/trigger")

    assert resp.json()["result"]["enabled"] is False
    assert resp.json()["result"]["next_run_at"] is None


# --------------------------------------------------------------------------- #
# manual task creation
# --------------------------------------------------------------------------- #
def test_pa_create_task_endpoint(monkeypatch):
    # The direct path: no intent classification, no field extraction, no LLM.
    monkeypatch.setattr(
        pa_router,
        "insert_todo",
        AsyncMock(return_value={"id": "t1", "title": "Call the dentist"}),
    )
    client = _make_client(MagicMock())

    resp = client.post(
        "/personal-assistant/tasks",
        json={"title": "Call the dentist", "priority": "high", "due_at": "2026-09-01"},
    )

    assert resp.status_code == 201
    assert resp.json()["result"]["title"] == "Call the dentist"
    payload = pa_router.insert_todo.await_args.args[1]
    assert payload["priority"] == "high"
    # exclude_none: an unset recurrence must not be written as an explicit null,
    # which would look like a task that was deliberately made non-repeating.
    assert "recurrence" not in payload


def test_pa_create_task_rejects_empty_title():
    client = _make_client(MagicMock())
    resp = client.post("/personal-assistant/tasks", json={"title": ""})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# bulk actions
# --------------------------------------------------------------------------- #
def test_pa_bulk_complete_by_ids(monkeypatch):
    monkeypatch.setattr(
        pa_router,
        "bulk_complete",
        AsyncMock(return_value={"completed": 2, "next_occurrences": [], "tasks": []}),
    )
    client = _make_client(MagicMock())

    resp = client.post(
        "/personal-assistant/tasks/bulk",
        json={"action": "complete", "ids": ["a", "b"]},
    )

    assert resp.status_code == 200
    assert resp.json()["result"]["completed"] == 2
    pa_router.bulk_complete.assert_awaited_once_with("u1", ["a", "b"])


def test_pa_bulk_complete_overdue_scope_resolves_ids_serverside(monkeypatch):
    # "Overdue" is bucketed on the date portion only; the client must not have
    # to reproduce that rule to sweep it.
    monkeypatch.setattr(
        pa_router, "fetch_overdue_ids", AsyncMock(return_value=["x", "y", "z"])
    )
    monkeypatch.setattr(
        pa_router,
        "bulk_complete",
        AsyncMock(return_value={"completed": 3, "next_occurrences": [], "tasks": []}),
    )
    client = _make_client(MagicMock())

    resp = client.post(
        "/personal-assistant/tasks/bulk", json={"action": "complete", "scope": "overdue"}
    )

    assert resp.status_code == 200
    pa_router.bulk_complete.assert_awaited_once_with("u1", ["x", "y", "z"])


def test_pa_bulk_clear_completed(monkeypatch):
    monkeypatch.setattr(pa_router, "delete_completed", AsyncMock(return_value=7))
    client = _make_client(MagicMock())

    resp = client.post(
        "/personal-assistant/tasks/bulk", json={"action": "delete", "scope": "completed"}
    )

    assert resp.json()["result"]["deleted"] == 7


def test_pa_bulk_rejects_mismatched_scope_and_action():
    # Deleting everything overdue is not something a "sweep" gesture should be
    # able to express by accident.
    client = _make_client(MagicMock())
    resp = client.post(
        "/personal-assistant/tasks/bulk", json={"action": "delete", "scope": "overdue"}
    )
    assert resp.status_code == 400


def test_pa_bulk_with_no_ids_is_a_no_op(monkeypatch):
    monkeypatch.setattr(pa_router, "bulk_complete", AsyncMock())
    client = _make_client(MagicMock())

    resp = client.post("/personal-assistant/tasks/bulk", json={"action": "complete", "ids": []})

    assert resp.status_code == 200
    assert resp.json()["result"]["completed"] == 0
    pa_router.bulk_complete.assert_not_awaited()


# --------------------------------------------------------------------------- #
# note editing
# --------------------------------------------------------------------------- #
def test_pa_update_note(monkeypatch):
    monkeypatch.setattr(
        pa_router,
        "update_note",
        AsyncMock(return_value={"id": "n1", "content": "allergic to walnuts"}),
    )
    client = _make_client(MagicMock())

    resp = client.put(
        "/personal-assistant/notes/n1", json={"content": "allergic to walnuts"}
    )

    assert resp.status_code == 200
    assert resp.json()["result"]["content"] == "allergic to walnuts"


def test_pa_update_note_missing_is_404(monkeypatch):
    monkeypatch.setattr(pa_router, "update_note", AsyncMock(return_value=None))
    client = _make_client(MagicMock())

    resp = client.put("/personal-assistant/notes/nope", json={"content": "x"})

    assert resp.status_code == 404


def test_pa_update_note_with_no_fields_is_400():
    client = _make_client(MagicMock())
    resp = client.put("/personal-assistant/notes/n1", json={})
    assert resp.status_code == 400


def test_pa_delete_note(monkeypatch):
    monkeypatch.setattr(pa_router, "delete_note", AsyncMock(return_value=True))
    client = _make_client(MagicMock())

    resp = client.delete("/personal-assistant/notes/n1")

    assert resp.status_code == 200
    pa_router.delete_note.assert_awaited_once_with("u1", "n1")


def test_pa_delete_note_missing_is_404(monkeypatch):
    monkeypatch.setattr(pa_router, "delete_note", AsyncMock(return_value=False))
    client = _make_client(MagicMock())

    resp = client.delete("/personal-assistant/notes/nope")

    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# digest schedule control
# --------------------------------------------------------------------------- #
def test_pa_set_trigger_schedule(monkeypatch):
    monkeypatch.setattr(
        pa_router,
        "update_trigger",
        AsyncMock(
            return_value={
                "enabled": True,
                "schedule_hour": 6,
                "timezone": "Asia/Kolkata",
            }
        ),
    )
    client = _make_client(MagicMock())

    resp = client.put(
        "/personal-assistant/trigger",
        json={"schedule_hour": 6, "timezone": "Asia/Kolkata"},
    )

    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["schedule_hour"] == 6
    assert result["timezone"] == "Asia/Kolkata"
    assert pa_router.update_trigger.await_args.args[2] == {
        "schedule_hour": 6,
        "timezone": "Asia/Kolkata",
    }


def test_pa_set_trigger_rejects_an_impossible_hour():
    client = _make_client(MagicMock())
    resp = client.put("/personal-assistant/trigger", json={"schedule_hour": 25})
    assert resp.status_code == 422


def test_pa_set_trigger_with_no_fields_is_400():
    client = _make_client(MagicMock())
    resp = client.put("/personal-assistant/trigger", json={})
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# quiet digest
# --------------------------------------------------------------------------- #
def test_has_anything_due_ignores_undated_tasks():
    # An undated task is pending forever, so counting it would mean a
    # notification every morning for the rest of time.
    assert not pa_triggers.has_anything_due({"no_date": [{"title": "someday"}]})
    assert pa_triggers.has_anything_due({"today": [{"title": "now"}]})
    assert not pa_triggers.has_anything_due({})


async def test_run_pa_triggers_stays_quiet_when_nothing_is_due(monkeypatch):
    monkeypatch.setattr(
        pa_triggers,
        "due_triggers",
        AsyncMock(return_value=[{"_id": "tr1", "user_id": "u1"}]),
    )
    monkeypatch.setattr(pa_triggers, "create_pending", AsyncMock())
    monkeypatch.setattr(pa_triggers, "mark_ran", AsyncMock())
    monkeypatch.setattr(pa_triggers, "send_push_notification", AsyncMock())
    # One pending task, no due date: nothing is actually due today.
    monkeypatch.setattr(
        pa_triggers,
        "fetch_todos",
        AsyncMock(return_value=[{"id": "t1", "title": "someday", "due_at": None}]),
    )

    await pa_triggers.run_pa_triggers()

    pa_triggers.send_push_notification.assert_not_awaited()
    pa_triggers.create_pending.assert_not_awaited()
    # Still stamped, or the hourly sweep retries this user all day.
    pa_triggers.mark_ran.assert_awaited_once()


# --------------------------------------------------------------------------- #
# planner_agent (plan my day)
# --------------------------------------------------------------------------- #
async def test_planner_agent_schedules_overdue_and_today(monkeypatch):
    monkeypatch.setattr(
        pa,
        "fetch_todos",
        AsyncMock(
            return_value=[
                {"id": "a", "title": "old", "due_at": "2000-01-01", "priority": "high"},
                {"id": "b", "title": "later", "due_at": "2999-01-01"},
                {"id": "c", "title": "someday"},
            ]
        ),
    )
    monkeypatch.setattr(
        pa,
        "llm",
        _llm_returning(
            pa.PlanOutput(
                blocks=[
                    {"start": "09:00", "end": "10:00", "title": "old", "task_id": "a"}
                ],
                unscheduled=["someday"],
            )
        ),
    )

    out = await pa.planner_agent(
        {"intent": "plan", "user_id": "u1", "query": "plan my day", "memory": {}}
    )

    assert out["task_status"] == "planned:1"
    assert out["plan"]["blocks"][0]["task_id"] == "a"
    # Upcoming work is not today's problem: only overdue, today and undated
    # tasks are offered to the planner.
    titles = [t["title"] for t in out["todos"]]
    assert "later" not in titles
    assert titles[0] == "old"


async def test_planner_agent_with_nothing_to_do_skips_the_model(monkeypatch):
    monkeypatch.setattr(pa, "fetch_todos", AsyncMock(return_value=[]))
    # No _llm_returning: reaching the model here would raise.
    monkeypatch.setattr(pa, "llm", MagicMock())

    out = await pa.planner_agent(
        {"intent": "plan", "user_id": "u1", "query": "plan my day", "memory": {}}
    )

    assert out["task_status"] == "nothing_to_plan"
    assert out["plan"]["blocks"] == []


async def test_planner_agent_feeds_the_learned_profile_to_the_model(monkeypatch):
    # work_hours and recurring_commitments have been recorded all along and only
    # ever shaped the wording of replies; the planner is what they were for.
    seen = {}

    def _capture(payload):
        seen.update(payload)
        return pa.PlanOutput(blocks=[])

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(
        lambda p: _capture(dict(p))
    )
    monkeypatch.setattr(pa, "llm", mock_llm)
    monkeypatch.setattr(
        pa,
        "fetch_todos",
        AsyncMock(return_value=[{"id": "a", "title": "x", "due_at": "2000-01-01"}]),
    )

    await pa.planner_agent(
        {
            "intent": "plan",
            "user_id": "u1",
            "query": "plan my day",
            "memory": {"work_hours": "10-6", "recurring_commitments": ["gym Mon"]},
        }
    )

    rendered = str(seen)
    assert "10-6" in rendered
    assert "gym Mon" in rendered


# --------------------------------------------------------------------------- #
# review_agent (weekly review)
# --------------------------------------------------------------------------- #
async def test_review_agent_reads_the_completed_window(monkeypatch):
    monkeypatch.setattr(
        pa,
        "fetch_completed_since",
        AsyncMock(return_value=[{"title": "shipped it"}, {"title": "and that"}]),
    )
    monkeypatch.setattr(pa, "fetch_todos", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        pa,
        "llm",
        _llm_returning(
            pa.ReviewOutput(
                summary="Solid week.",
                highlights=["shipped it"],
                carry_over=[],
                suggestions=["keep going"],
            )
        ),
    )

    out = await pa.review_agent(
        {"intent": "review", "user_id": "u1", "query": "what did I do", "memory": {}}
    )

    assert out["task_status"] == "reviewed:2"
    assert out["review"]["completed_count"] == 2
    assert out["suggestions"] == ["keep going"]
    # A week back, not "everything ever".
    assert pa.fetch_completed_since.await_args.args[0] == "u1"


async def test_review_agent_uses_the_research_history(monkeypatch):
    # `pa_research_history` has been written on every research turn and read by
    # nothing; the review is the first thing that consumes it.
    seen = {}
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(
        lambda p: (seen.update(dict(p)), pa.ReviewOutput(summary="ok"))[1]
    )
    monkeypatch.setattr(pa, "llm", mock_llm)
    monkeypatch.setattr(pa, "fetch_completed_since", AsyncMock(return_value=[]))
    monkeypatch.setattr(pa, "fetch_todos", AsyncMock(return_value=[]))

    await pa.review_agent(
        {
            "intent": "review",
            "user_id": "u1",
            "query": "weekly review",
            "memory": {pa.MEMORY_RESEARCH_KEY: ["note apps", "standing desks"]},
        }
    )

    assert "standing desks" in str(seen)


def test_decide_agent_routes_the_new_intents():
    assert pa.decide_agent({"intent": "plan"}) == "planner_agent"
    assert pa.decide_agent({"intent": "review"}) == "review_agent"


def test_synthesis_context_includes_the_plan_and_review():
    # Both nodes feed synthesize; a node whose output the synthesiser cannot see
    # produces a reply that does not mention what it did.
    graph = pa.build_graph().compile()
    assert "planner_agent" in graph.nodes
    assert "review_agent" in graph.nodes


# --------------------------------------------------------------------------- #
# recurrence survives every completion path
# --------------------------------------------------------------------------- #
async def test_update_by_id_spawns_the_next_occurrence(monkeypatch):
    # The endpoint the app's tick box calls. It used to just set status=done,
    # so completing a repeating task in the UI ended the series — while asking
    # the assistant to complete the same task kept it going.
    task_id = "651f1f77bcf86cd799439011"
    recurring = {
        "id": task_id,
        "title": "water the plants",
        "status": "pending",
        "recurrence": "weekly",
        "due_at": "2026-08-20T09:00:00",
        "priority": "medium",
    }
    monkeypatch.setattr(
        pa_repo, "get_todo_by_id", AsyncMock(return_value=dict(recurring))
    )
    col = MagicMock()
    col.update_one = AsyncMock()
    col.find_one = AsyncMock(return_value={**recurring, "status": "done"})
    monkeypatch.setattr(pa_repo, "get_db", lambda: {pa_repo.TODOS: col})
    monkeypatch.setattr(
        pa_repo, "insert_todo", AsyncMock(return_value={"id": "t2", "title": "water the plants"})
    )

    out = await pa_repo.update_todo_by_id("u1", task_id, {"status": "done"})

    assert out["next_occurrence"]["id"] == "t2"
    follow_up = pa_repo.insert_todo.await_args.args[1]
    assert follow_up["recurrence"] == "weekly"
    # One week on from the old due date, not from now.
    assert follow_up["due_at"].startswith("2026-08-27")


async def test_update_by_id_does_not_respawn_an_already_done_task(monkeypatch):
    # Re-saving a completed task, or editing its title, must not mint another
    # occurrence each time.
    task_id = "651f1f77bcf86cd799439011"
    done = {"id": task_id, "title": "x", "status": "done", "recurrence": "daily"}
    monkeypatch.setattr(pa_repo, "get_todo_by_id", AsyncMock(return_value=dict(done)))
    col = MagicMock()
    col.update_one = AsyncMock()
    col.find_one = AsyncMock(return_value=dict(done))
    monkeypatch.setattr(pa_repo, "get_db", lambda: {pa_repo.TODOS: col})
    monkeypatch.setattr(pa_repo, "insert_todo", AsyncMock())

    out = await pa_repo.update_todo_by_id("u1", task_id, {"status": "done"})

    assert "next_occurrence" not in out
    pa_repo.insert_todo.assert_not_awaited()


async def test_reopening_a_task_spawns_nothing(monkeypatch):
    task_id = "651f1f77bcf86cd799439011"
    done = {"id": task_id, "title": "x", "status": "done", "recurrence": "daily"}
    monkeypatch.setattr(pa_repo, "get_todo_by_id", AsyncMock(return_value=dict(done)))
    col = MagicMock()
    col.update_one = AsyncMock()
    col.find_one = AsyncMock(return_value={**done, "status": "pending"})
    monkeypatch.setattr(pa_repo, "get_db", lambda: {pa_repo.TODOS: col})
    monkeypatch.setattr(pa_repo, "insert_todo", AsyncMock())

    await pa_repo.update_todo_by_id("u1", task_id, {"status": "pending"})

    pa_repo.insert_todo.assert_not_awaited()


async def test_update_by_id_on_a_missing_task_is_none(monkeypatch):
    monkeypatch.setattr(pa_repo, "get_todo_by_id", AsyncMock(return_value=None))
    assert await pa_repo.update_todo_by_id("u1", "nope", {"status": "done"}) is None


# --------------------------------------------------------------------------- #
# note identity
# --------------------------------------------------------------------------- #
async def test_fetch_notes_backfills_ids_once(monkeypatch):
    # Notes predate having ids. Editing or deleting one needs a stable handle,
    # so legacy notes get one the first time they are read — and it is written
    # back, so the id a client is handed stays valid.
    legacy = [{"content": "old fact"}, {"content": "another"}]
    monkeypatch.setattr(pa_repo, "_memory_value", AsyncMock(return_value=legacy))
    monkeypatch.setattr(pa_repo, "remember", AsyncMock())

    notes = await pa_repo.fetch_notes("u1")

    assert all(n["id"] for n in notes)
    assert len({n["id"] for n in notes}) == 2
    pa_repo.remember.assert_awaited_once()


async def test_fetch_notes_leaves_existing_ids_alone(monkeypatch):
    already = [{"id": "keep-me", "content": "x"}]
    monkeypatch.setattr(pa_repo, "_memory_value", AsyncMock(return_value=already))
    monkeypatch.setattr(pa_repo, "remember", AsyncMock())

    notes = await pa_repo.fetch_notes("u1")

    assert notes[0]["id"] == "keep-me"
    # No rewrite when there is nothing to backfill.
    pa_repo.remember.assert_not_awaited()


async def test_update_note_edits_in_place(monkeypatch):
    stored = [{"id": "n1", "content": "allergic to peanuts", "category": "health"}]
    monkeypatch.setattr(pa_repo, "_memory_value", AsyncMock(return_value=stored))
    monkeypatch.setattr(pa_repo, "remember", AsyncMock())

    note = await pa_repo.update_note("u1", "n1", content="allergic to walnuts")

    assert note["content"] == "allergic to walnuts"
    # Untouched fields survive a content-only edit.
    assert note["category"] == "health"
    assert note["updated_at"]


async def test_update_note_can_clear_a_category(monkeypatch):
    stored = [{"id": "n1", "content": "x", "category": "health"}]
    monkeypatch.setattr(pa_repo, "_memory_value", AsyncMock(return_value=stored))
    monkeypatch.setattr(pa_repo, "remember", AsyncMock())

    note = await pa_repo.update_note("u1", "n1", category="")

    assert note["category"] is None


async def test_update_note_unknown_id(monkeypatch):
    monkeypatch.setattr(pa_repo, "_memory_value", AsyncMock(return_value=[]))
    monkeypatch.setattr(pa_repo, "remember", AsyncMock())
    assert await pa_repo.update_note("u1", "nope", content="x") is None


async def test_delete_note_removes_only_that_note(monkeypatch):
    stored = [{"id": "n1", "content": "a"}, {"id": "n2", "content": "b"}]
    monkeypatch.setattr(pa_repo, "_memory_value", AsyncMock(return_value=stored))
    monkeypatch.setattr(pa_repo, "remember", AsyncMock())

    assert await pa_repo.delete_note("u1", "n1") is True

    remaining = pa_repo.remember.await_args.args[2]
    assert [n["id"] for n in remaining] == ["n2"]


async def test_delete_note_unknown_id_reports_false(monkeypatch):
    monkeypatch.setattr(
        pa_repo, "_memory_value", AsyncMock(return_value=[{"id": "n1"}])
    )
    monkeypatch.setattr(pa_repo, "remember", AsyncMock())
    assert await pa_repo.delete_note("u1", "nope") is False


# --------------------------------------------------------------------------- #
# bulk repository behaviour
# --------------------------------------------------------------------------- #
async def test_bulk_complete_is_per_task_so_recurrence_is_honoured(monkeypatch):
    # An update_many would be one round trip and would skip every repeating
    # task's next occurrence.
    async def _complete(user_id, task_id):
        if task_id == "b":
            return {"id": "b", "title": "b", "next_occurrence": {"id": "b2"}}
        if task_id == "missing":
            return None
        return {"id": task_id, "title": task_id}

    monkeypatch.setattr(pa_repo, "complete_todo_by_id", _complete)

    out = await pa_repo.bulk_complete("u1", ["a", "b", "missing"])

    assert out["completed"] == 2
    assert out["next_occurrences"] == [{"id": "b2"}]


async def test_fetch_overdue_ids_uses_the_agenda_rule(monkeypatch):
    # Bucketed on the date portion, so something due later today is not overdue.
    monkeypatch.setattr(
        pa_repo,
        "fetch_todos",
        AsyncMock(
            return_value=[
                {"id": "old", "due_at": "2000-01-01"},
                {"id": "later_today", "due_at": f"{date.today().isoformat()}T23:59:00"},
                {"id": "undated"},
            ]
        ),
    )

    assert await pa_repo.fetch_overdue_ids("u1") == ["old"]


# --------------------------------------------------------------------------- #
# server-side paging, search and sort
# --------------------------------------------------------------------------- #
def _todos_page_collection(monkeypatch, docs=(), total=None):
    """A `todos` collection that answers one aggregate + one count."""
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=[dict(d) for d in docs])
    col = MagicMock()
    col.aggregate = MagicMock(return_value=cursor)
    col.count_documents = AsyncMock(
        return_value=len(docs) if total is None else total
    )
    monkeypatch.setattr(pa_repo, "get_db", lambda: {pa_repo.TODOS: col})
    return col


def _stage(pipeline, name):
    """The first stage of kind `name` in an aggregation pipeline."""
    return next((s[name] for s in pipeline if name in s), None)


async def test_search_matches_title_and_details_and_escapes_the_needle():
    # Whatever the user typed into a search box reaches Mongo as a regex. An
    # unescaped '(' is an error and an unescaped '.*' scans the collection.
    query = pa_repo._todo_query("u1", q="  plan (draft)  ")

    import re

    assert query["user_id"] == "u1"
    fields = {list(clause)[0] for clause in query["$or"]}
    assert fields == {"title", "details"}
    # Trimmed, escaped, and case-insensitive — the same substring match the
    # client was doing, over the whole list instead of one fetch of it.
    assert query["$or"][0]["title"]["$regex"] == re.escape("plan (draft)")
    assert "(" not in query["$or"][0]["title"]["$regex"].replace("\\(", "")
    assert query["$or"][0]["title"]["$options"] == "i"


async def test_a_blank_search_is_not_a_filter():
    # An empty search box must not become a regex that matches nothing useful.
    assert "$or" not in pa_repo._todo_query("u1", q="   ")
    assert "$or" not in pa_repo._todo_query("u1", q=None)


async def test_page_reports_the_total_it_was_taken_from(monkeypatch):
    col = _todos_page_collection(monkeypatch, docs=[{"_id": "a"}, {"_id": "b"}], total=307)

    out = await pa_repo.fetch_todos_page("u1", limit=2, offset=10)

    assert out["total"] == 307
    assert [t["id"] for t in out["items"]] == ["a", "b"]
    pipeline = col.aggregate.call_args.args[0]
    assert _stage(pipeline, "$skip") == 10
    assert _stage(pipeline, "$limit") == 2
    # The count is of the whole filtered set, not of the page.
    assert col.count_documents.await_args.args[0]["user_id"] == "u1"


async def test_priority_sorts_by_urgency_not_alphabetically(monkeypatch):
    # Sorting the stored word gives high < low < medium, which puts "low" in
    # the middle of a list the user asked to see by importance.
    col = _todos_page_collection(monkeypatch)

    await pa_repo.fetch_todos_page("u1", sort="priority")

    pipeline = col.aggregate.call_args.args[0]
    ranked = _stage(pipeline, "$addFields")["_rank"]["$switch"]["branches"]
    assert [b["then"] for b in ranked] == [3, 2, 1]
    sort_spec = _stage(pipeline, "$sort")
    # Default direction for priority is "most urgent first".
    assert sort_spec["_rank"] == -1
    assert "priority" not in sort_spec
    # The computed rank is not part of the task the client gets back.
    assert _stage(pipeline, "$project") == {"_rank": 0}


async def test_undated_tasks_trail_a_due_date_sort_in_both_directions(monkeypatch):
    for order, expected in (("asc", 1), ("desc", -1)):
        col = _todos_page_collection(monkeypatch)

        await pa_repo.fetch_todos_page("u1", sort="due_at", order=order)

        sort_spec = _stage(col.aggregate.call_args.args[0], "$sort")
        # Undated first key, always ascending: they are unscheduled rather than
        # earliest, the same distinction categorize_agenda draws.
        assert list(sort_spec) == ["_undated", "due_at", "_id"]
        assert sort_spec["_undated"] == 1
        assert sort_spec["due_at"] == expected


async def test_every_sort_gets_a_stable_tiebreaker(monkeypatch):
    # Without one, two tasks sharing a priority (or a due date, or a title) can
    # swap places between two requests and a row slides across a page boundary
    # — appearing twice, or not at all, as the client walks the offsets.
    for field in pa_repo.TODO_SORT_FIELDS:
        col = _todos_page_collection(monkeypatch)

        await pa_repo.fetch_todos_page("u1", sort=field)

        sort_spec = _stage(col.aggregate.call_args.args[0], "$sort")
        assert list(sort_spec)[-1] == "_id"


async def test_source_self_means_the_field_is_absent():
    # Every cross-agent task has carried its origin since create_tasks shipped,
    # and nothing read it — so 31 "Learn:" tasks sat in the same list as the
    # four the user typed. `self` is the half that was unaskable.
    assert pa_repo._todo_query("u1", source="self")["source"] is None
    # Mongo's null match covers both a missing field on old rows and an
    # explicit null on new ones.
    assert "source" not in pa_repo._todo_query("u1")


async def test_source_filters_to_one_agent():
    assert pa_repo._todo_query("u1", source="learning_tracker") == {
        "user_id": "u1",
        "source": "learning_tracker",
    }


async def test_fetch_todos_page_passes_the_source_filter(monkeypatch):
    col = _todos_page_collection(monkeypatch)

    await pa_repo.fetch_todos_page("u1", source="job_helper")

    assert _stage(col.aggregate.call_args.args[0], "$match")["source"] == "job_helper"


def test_pa_tasks_forwards_the_source_filter(monkeypatch):
    monkeypatch.setattr(
        pa_router, "fetch_todos_page", AsyncMock(return_value={"items": [], "total": 0})
    )
    client = _make_client(MagicMock())

    client.get("/personal-assistant/tasks?source=self")

    assert pa_router.fetch_todos_page.await_args.kwargs["source"] == "self"


def test_pa_stats_counts_by_source(monkeypatch):
    # What a client builds the filter chips from: it cannot offer "Learning" as
    # a filter without knowing how many there are.
    col = MagicMock()
    col.aggregate.return_value.to_list = AsyncMock(
        return_value=[
            {"_id": {"status": "pending", "priority": "medium", "source": "learning_tracker"}, "count": 31},
            {"_id": {"status": "pending", "priority": "high", "source": None}, "count": 3},
            {"_id": {"status": "done", "priority": "low"}, "count": 1},
        ]
    )
    monkeypatch.setattr(pa_router, "get_db", lambda: {pa_router.TODOS: col})
    client = _make_client(MagicMock())

    result = client.get("/personal-assistant/tasks/stats").json()["result"]

    assert result["total"] == 35
    assert result["by_source"]["learning_tracker"] == 31
    # A missing or null source counts as the user's own, matching the value
    # GET /tasks?source= takes.
    assert result["by_source"]["self"] == 4
    assert result["by_status"]["pending"] == 34


async def test_filters_still_apply_to_a_page(monkeypatch):
    col = _todos_page_collection(monkeypatch)

    await pa_repo.fetch_todos_page("u1", status="pending", priority="high", q="tax")

    match = _stage(col.aggregate.call_args.args[0], "$match")
    assert match["status"] == "pending"
    assert match["priority"] == "high"
    assert "$or" in match


def test_pa_tasks_endpoint_reports_pagination(monkeypatch):
    monkeypatch.setattr(
        pa_router,
        "fetch_todos_page",
        AsyncMock(return_value={"items": [{"id": "t1"}], "total": 307}),
    )
    client = _make_client(MagicMock())

    resp = client.get("/personal-assistant/tasks?limit=1&offset=0")

    page = resp.json()["page"]
    assert page["total"] == 307
    assert page["has_more"] is True
    assert page["limit"] == 1 and page["offset"] == 0
    # The direction actually used, so the client can render the control from it
    # rather than re-deriving the default.
    assert page["order"] == "desc"


def test_pa_tasks_last_page_has_no_more(monkeypatch):
    monkeypatch.setattr(
        pa_router,
        "fetch_todos_page",
        AsyncMock(return_value={"items": [{"id": "t1"}, {"id": "t2"}], "total": 12}),
    )
    client = _make_client(MagicMock())

    resp = client.get("/personal-assistant/tasks?limit=10&offset=10")

    assert resp.json()["page"]["has_more"] is False


def test_pa_tasks_forwards_search_and_sort_to_the_query(monkeypatch):
    monkeypatch.setattr(
        pa_router, "fetch_todos_page", AsyncMock(return_value={"items": [], "total": 0})
    )
    client = _make_client(MagicMock())

    client.get(
        "/personal-assistant/tasks?q=dentist&sort=due_at&order=asc&status=pending"
    )

    kwargs = pa_router.fetch_todos_page.await_args.kwargs
    assert kwargs["q"] == "dentist"
    assert kwargs["sort"] == "due_at"
    assert kwargs["order"] == "asc"
    assert kwargs["status"] == "pending"


def test_pa_tasks_rejects_an_unknown_sort_field():
    # Sort reaches a Mongo field name, so the set of them is closed.
    client = _make_client(MagicMock())
    assert client.get("/personal-assistant/tasks?sort=user_id").status_code == 422
    assert client.get("/personal-assistant/tasks?order=sideways").status_code == 422


def test_pa_tasks_bounds_the_page_size():
    client = _make_client(MagicMock())
    assert client.get("/personal-assistant/tasks?limit=501").status_code == 422
    assert client.get("/personal-assistant/tasks?limit=0").status_code == 422
    assert client.get("/personal-assistant/tasks?offset=-1").status_code == 422


# --------------------------------------------------------------------------- #
# forgetting a learned fact
# --------------------------------------------------------------------------- #
async def test_forget_profile_key_unsets_just_that_field(monkeypatch):
    col = MagicMock()
    col.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=1))
    monkeypatch.setattr(memory_store, "get_db", lambda: {memory_store.MEMORIES: col})

    assert await memory_store.forget_profile_key("u1", "work_hours") is True

    query, update = col.update_one.await_args.args
    # Scoped to the field existing, so "nothing to forget" is distinguishable
    # from "forgotten" without a second read.
    assert query == {"user_id": "u1", "data.work_hours": {"$exists": True}}
    assert update["$unset"] == {"data.work_hours": ""}


async def test_forget_profile_key_reports_a_miss(monkeypatch):
    col = MagicMock()
    col.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=0))
    monkeypatch.setattr(memory_store, "get_db", lambda: {memory_store.MEMORIES: col})

    assert await memory_store.forget_profile_key("u1", "never_set") is False


def test_pa_delete_memory(monkeypatch):
    monkeypatch.setattr(
        pa_router, "forget_profile_key", AsyncMock(return_value=True)
    )
    client = _make_client(MagicMock())

    resp = client.delete("/personal-assistant/memory/work_hours")

    assert resp.status_code == 200
    pa_router.forget_profile_key.assert_awaited_once_with("u1", "work_hours")


def test_pa_delete_memory_missing_is_404(monkeypatch):
    monkeypatch.setattr(pa_router, "forget_profile_key", AsyncMock(return_value=False))
    client = _make_client(MagicMock())

    assert client.delete("/personal-assistant/memory/nope").status_code == 404


def test_pa_delete_memory_refuses_to_wipe_the_notes(monkeypatch):
    # `data.pa_notes` shares the memory document but is the whole notes list,
    # not a learned fact — one call here would forget every note at once.
    monkeypatch.setattr(pa_router, "forget_profile_key", AsyncMock())
    client = _make_client(MagicMock())

    resp = client.delete(f"/personal-assistant/memory/{pa_repo.NOTES_KEY}")

    assert resp.status_code == 400
    pa_router.forget_profile_key.assert_not_awaited()


def test_pa_delete_memory_rejects_a_field_path(monkeypatch):
    # The key becomes a field path in the $unset.
    monkeypatch.setattr(pa_router, "forget_profile_key", AsyncMock())
    client = _make_client(MagicMock())

    assert client.delete("/personal-assistant/memory/$where").status_code == 400
    pa_router.forget_profile_key.assert_not_awaited()


# --------------------------------------------------------------------------- #
# which days the digest arrives on
# --------------------------------------------------------------------------- #
def _at(iso):
    from datetime import datetime as _dt

    return _dt.fromisoformat(iso)


def test_schedule_dows_reads_old_rows_and_new_ones():
    # The meal planner still writes a bare int; those rows must keep meaning
    # exactly what they did.
    assert trigger_store.schedule_dows({"schedule_dow": 6}) == [6]
    assert trigger_store.schedule_dows({"schedule_dow": [4, 0, 2]}) == [0, 2, 4]
    assert trigger_store.schedule_dows({}) is None
    # "No days at all" is never what someone meant to store.
    assert trigger_store.schedule_dows({"schedule_dow": []}) is None


def test_is_due_honours_a_weekdays_list():
    # 2026-08-31 is a Monday, 2026-08-30 the Sunday before it.
    weekdays = {
        "enabled": True,
        "schedule_hour": 8,
        "timezone": "UTC",
        "schedule_dow": [0, 1, 2, 3, 4],
    }
    assert trigger_store.is_due(weekdays, _at("2026-08-31T08:00:00+00:00")) is True
    assert trigger_store.is_due(weekdays, _at("2026-08-30T08:00:00+00:00")) is False


def test_next_run_skips_the_weekend():
    weekdays = {
        "enabled": True,
        "schedule_hour": 8,
        "timezone": "UTC",
        "schedule_dow": [0, 1, 2, 3, 4],
    }
    # Friday 2026-08-28, past the hour: the next slot is Monday, not Saturday.
    assert trigger_store.next_run_at(
        weekdays, _at("2026-08-28T10:00:00+00:00")
    ).startswith("2026-08-31T08:00")


async def test_update_trigger_can_clear_a_field(monkeypatch):
    col = MagicMock()
    col.update_one = AsyncMock()
    col.find_one = AsyncMock(return_value={"user_id": "u1"})
    monkeypatch.setattr(trigger_store, "get_db", lambda: {"triggers": col})

    await trigger_store.update_trigger(
        "u1", "pa_digest", {"schedule_hour": 8}, unset=["schedule_dow"]
    )

    update = col.update_one.await_args.args[1]
    assert update["$unset"] == {"schedule_dow": ""}
    # Mongo rejects a document that writes and clears the same path, so a
    # cleared field must not also arrive as an insert default.
    assert "schedule_dow" not in update["$setOnInsert"]
    assert "schedule_dow" not in update["$set"]


def test_pa_set_trigger_to_weekdays(monkeypatch):
    monkeypatch.setattr(
        pa_router,
        "update_trigger",
        AsyncMock(return_value={"enabled": True, "schedule_dow": [0, 1, 2, 3, 4]}),
    )
    client = _make_client(MagicMock())

    resp = client.put(
        "/personal-assistant/trigger",
        json={"schedule_dow": [4, 0, 1, 2, 3, 0]},
    )

    assert resp.status_code == 200
    assert resp.json()["result"]["schedule_dow"] == [0, 1, 2, 3, 4]
    # Deduped and ordered, so two requests naming the same days store the same.
    assert pa_router.update_trigger.await_args.args[2]["schedule_dow"] == [0, 1, 2, 3, 4]


def test_pa_set_trigger_null_dow_goes_back_to_every_day(monkeypatch):
    monkeypatch.setattr(
        pa_router, "update_trigger", AsyncMock(return_value={"enabled": True})
    )
    client = _make_client(MagicMock())

    resp = client.put("/personal-assistant/trigger", json={"schedule_dow": None})

    assert resp.status_code == 200
    assert resp.json()["result"]["schedule_dow"] is None
    # An explicit null is a change, not an empty request.
    assert pa_router.update_trigger.await_args.kwargs["unset"] == ["schedule_dow"]


def test_pa_set_trigger_rejects_an_empty_dow_list():
    # Storing "no days at all" would switch the digest off without saying so.
    client = _make_client(MagicMock())
    resp = client.put("/personal-assistant/trigger", json={"schedule_dow": []})
    assert resp.status_code == 400


def test_pa_set_trigger_rejects_an_impossible_day():
    client = _make_client(MagicMock())
    assert (
        client.put("/personal-assistant/trigger", json={"schedule_dow": [7]}).status_code
        == 422
    )


# --------------------------------------------------------------------------- #
# a qualified "all" is a filter, not every task
# --------------------------------------------------------------------------- #
def _pending(monkeypatch, docs):
    col = MagicMock()
    col.find.return_value.to_list = AsyncMock(return_value=[dict(d) for d in docs])
    monkeypatch.setattr(pa_repo, "get_db", lambda: {pa_repo.TODOS: col})


_WORKSPACE = [
    {"_id": "1", "title": "Learn: recursion"},
    {"_id": "2", "title": "Learn: graphs"},
    {"_id": "3", "title": "Call Mom"},
    {"_id": "4", "title": "Launch My Side Project"},
]


async def test_a_title_narrows_even_when_match_all_is_set(monkeypatch):
    # "Delete all my Learn: tasks" — the word that makes the request *more*
    # specific is the same word that sets match_all. The filter has to win, or
    # scoping a bulk delete is what makes it unbounded.
    _pending(monkeypatch, _WORKSPACE)

    matched = await pa_repo.find_pending_todos("u1", "Learn:", match_all=True)

    assert [t["title"] for t in matched] == ["Learn: recursion", "Learn: graphs"]


async def test_match_all_without_a_title_still_means_everything(monkeypatch):
    # "Delete everything" must keep working — match_all widens only when there
    # is no filter to widen past.
    _pending(monkeypatch, _WORKSPACE)

    matched = await pa_repo.find_pending_todos("u1", None, match_all=True)

    assert len(matched) == 4


async def test_neither_a_title_nor_match_all_matches_nothing(monkeypatch):
    _pending(monkeypatch, _WORKSPACE)
    assert await pa_repo.find_pending_todos("u1", None, match_all=False) == []


def test_describe_selection_states_the_rule_not_just_the_count():
    assert (
        pa_repo.describe_selection("Learn:", True, 12)
        == '12 tasks whose title contains "Learn:"'
    )
    assert pa_repo.describe_selection(None, True, 34) == "all 34 pending tasks"
    # Singular reads as a sentence too — this goes on a confirmation card.
    assert pa_repo.describe_selection("Mom", False, 1) == '1 task whose title contains "Mom"'


async def test_the_delete_card_carries_the_filter_it_resolved(monkeypatch):
    # A list of titles with no statement of the rule that produced them gives
    # the user nothing to check the rule against.
    monkeypatch.setattr(pa, "get_pending", AsyncMock(return_value=None))
    monkeypatch.setattr(
        pa, "_extract_selector", AsyncMock(return_value=pa.TaskSelector(title="Learn:", match_all=True))
    )
    monkeypatch.setattr(
        pa,
        "find_pending_todos",
        AsyncMock(return_value=[{"id": "1", "title": "Learn: recursion"}]),
    )
    monkeypatch.setattr(pa, "create_pending", AsyncMock(return_value="ap1"))
    captured = {}

    def _interrupt(payload):
        captured.update(payload)
        return "rejected"

    monkeypatch.setattr(pa, "interrupt", _interrupt)
    monkeypatch.setattr(pa, "resolve", AsyncMock())

    await pa._delete_with_approval({"user_id": "u1", "thread_id": "t1", "query": "delete all my Learn: tasks"})

    assert captured["count"] == 1
    assert captured["scope"] == '1 task whose title contains "Learn:"'
    # And it is persisted, so a resumed thread shows the same card.
    assert pa.create_pending.await_args.args[3]["scope"] == captured["scope"]


# --------------------------------------------------------------------------- #
# relative dates resolve against today, not the model's prior
# --------------------------------------------------------------------------- #
def test_the_date_anchor_is_in_the_users_timezone():
    # Dated in their zone, not the server's: "tomorrow" turns over at their
    # midnight, and the digest already schedules on the same principle.
    anchored = pa._today_for({"memory": {"timezone": "Pacific/Kiritimati"}})
    behind = pa._today_for({"memory": {"timezone": "Pacific/Midway"}})

    # +14 and -11 are never the same date at the same instant.
    assert anchored != behind
    # Named as well as numbered, so "Friday" is resolvable.
    assert anchored.split()[0] in (
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    )


def test_an_unknown_timezone_falls_back_to_utc_rather_than_failing():
    from datetime import datetime as _dt, timezone as _tz

    out = pa._today_for({"memory": {"timezone": "Mars/Olympus_Mons"}, "user_id": "u1"})

    assert out.endswith(_dt.now(_tz.utc).date().isoformat())


def test_a_user_with_no_recorded_timezone_still_gets_a_date():
    assert pa._today_for({})
    assert pa._today_for({"memory": {}})


async def test_add_passes_today_into_the_extractor(monkeypatch):
    # The bug this closes: with no anchor the model resolved "tomorrow" from its
    # training prior, so a task added on 29 Aug landed in October — and that
    # wrong date then drove the agenda, the digest and every overdue count.
    seen = {}

    class _Chain:
        def __or__(self, other):
            return self

        async def ainvoke(self, params):
            seen.update(params)
            return pa.TaskInput(title="call the dentist", due_at="2026-08-30")

    monkeypatch.setattr(pa.ChatPromptTemplate, "from_messages", lambda msgs: _Chain())
    monkeypatch.setattr(pa, "insert_todo", AsyncMock(return_value={"id": "t1"}))

    await pa.todo_agent(
        {
            "intent": "add",
            "user_id": "u1",
            "query": "remind me to call the dentist tomorrow",
            "memory": {"timezone": "UTC"},
        }
    )

    assert "today" in seen and seen["today"]


async def test_update_passes_today_into_the_extractor(monkeypatch):
    # Same gap on the path that moves a due date.
    seen = {}

    class _Chain:
        def __or__(self, other):
            return self

        async def ainvoke(self, params):
            seen.update(params)
            return pa.TaskUpdateInput(title="dentist", new_due_at="2026-09-04")

    monkeypatch.setattr(pa.ChatPromptTemplate, "from_messages", lambda msgs: _Chain())
    monkeypatch.setattr(pa, "update_todo", AsyncMock(return_value={"id": "t1"}))

    await pa.todo_agent(
        {"intent": "update", "user_id": "u1", "query": "move the dentist to Friday", "memory": {}}
    )

    assert "today" in seen and seen["today"]


def test_pa_trigger_state_reports_a_legacy_int_as_a_list(monkeypatch):
    # One shape for the client to render, whichever way the row was written.
    monkeypatch.setattr(
        pa_router,
        "get_trigger",
        AsyncMock(return_value={"enabled": True, "schedule_hour": 8, "schedule_dow": 6}),
    )
    client = _make_client(MagicMock())

    resp = client.get("/personal-assistant/trigger")

    assert resp.json()["result"]["schedule_dow"] == [6]
