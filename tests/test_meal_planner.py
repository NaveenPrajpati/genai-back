"""Tests for the meal-planner agent and HTTP endpoints.

No real LLM calls or Supabase access — the LLM chain, the compiled LangGraph
agent, and the Supabase client are all replaced with fakes so the tests are
deterministic and offline.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END

import app.routers.meal_planner as meal_planner
from app.dependencies import get_current_user


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _chainable(execute_data):
    """A Supabase-client stand-in whose query builder returns itself for every
    chained call and yields `execute_data` from .execute()."""
    q = MagicMock()
    for meth in (
        "select", "eq", "in_", "ilike", "order", "limit",
        "maybe_single", "insert", "update", "upsert",
    ):
        getattr(q, meth).return_value = q
    q.execute.return_value = SimpleNamespace(data=execute_data)
    supa = MagicMock()
    supa.table.return_value = q
    return supa


def _supabase_seq(*results):
    """Like _chainable but returns a different .data for each .execute() call,
    in order — for code paths that issue several queries."""
    q = MagicMock()
    for meth in (
        "select", "eq", "in_", "ilike", "order", "limit",
        "maybe_single", "insert", "update", "upsert",
    ):
        getattr(q, meth).return_value = q
    q.execute.side_effect = [SimpleNamespace(data=d) for d in results]
    supa = MagicMock()
    supa.table.return_value = q
    return supa


def _make_client(agent, user=None):
    app = FastAPI()
    app.include_router(meal_planner.mealRouter)
    app.state.agent = agent
    app.dependency_overrides[get_current_user] = lambda: user or {"uid": "u1"}
    return TestClient(app)


# --------------------------------------------------------------------------- #
# intent classifier + routing
# --------------------------------------------------------------------------- #
async def test_classify_intent_returns_llm_intent(monkeypatch):
    fake_chain = RunnableLambda(
        lambda _: meal_planner.IntentOutput(intent="plan")
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = fake_chain
    monkeypatch.setattr(meal_planner, "llm", mock_llm)

    out = await meal_planner.classify_intent({"query": "plan my whole week"})

    assert out == {"intent": "plan"}
    mock_llm.with_structured_output.assert_called_once_with(
        meal_planner.IntentOutput
    )


def test_decide_agent_routes_each_intent():
    assert meal_planner.decide_agent({"intent": "log"}) == "log_agent"
    assert meal_planner.decide_agent({"intent": "query"}) == "query_agent"
    assert meal_planner.decide_agent({"intent": "research"}) == "research_agent"
    assert meal_planner.decide_agent({"intent": "plan"}) == "plan_agent"


def test_decide_agent_unknown_intent_ends():
    assert meal_planner.decide_agent({"intent": "nonsense"}) == END
    assert meal_planner.decide_agent({}) == END


# --------------------------------------------------------------------------- #
# approve / resume cycle
# --------------------------------------------------------------------------- #
def test_approve_happy_path_resumes_agent(monkeypatch):
    monkeypatch.setattr(
        meal_planner, "supabase", _chainable({"id": "a1", "user_id": "u1"})
    )
    agent = MagicMock()
    agent.aget_state = AsyncMock(
        return_value=SimpleNamespace(next=("plan_agent",))
    )
    agent.ainvoke = AsyncMock(
        return_value={"plan_status": "approved", "plan_id": "p1"}
    )
    client = _make_client(agent)

    resp = client.post(
        "/meal-planner/approve",
        json={"thread_id": "t1", "decision": "approved"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["result"]["plan_status"] == "approved"
    agent.ainvoke.assert_awaited_once()


def test_approve_other_users_thread_is_forbidden(monkeypatch):
    monkeypatch.setattr(
        meal_planner,
        "supabase",
        _chainable({"id": "a1", "user_id": "someone_else"}),
    )
    agent = MagicMock()
    agent.aget_state = AsyncMock()
    agent.ainvoke = AsyncMock()
    client = _make_client(agent)

    resp = client.post(
        "/meal-planner/approve",
        json={"thread_id": "t1", "decision": "approved"},
    )

    assert resp.status_code == 403
    agent.ainvoke.assert_not_awaited()  # never resumes someone else's plan


def test_approve_no_pending_approval_is_404(monkeypatch):
    monkeypatch.setattr(meal_planner, "supabase", _chainable(None))
    agent = MagicMock()
    agent.aget_state = AsyncMock()
    agent.ainvoke = AsyncMock()
    client = _make_client(agent)

    resp = client.post(
        "/meal-planner/approve",
        json={"thread_id": "missing", "decision": "approved"},
    )

    assert resp.status_code == 404
    agent.ainvoke.assert_not_awaited()


def test_approve_lost_thread_after_restart_is_404(monkeypatch):
    # Approval row exists and is owned, but the checkpointer has no paused thread
    # (e.g. server restarted with an in-memory saver).
    monkeypatch.setattr(
        meal_planner, "supabase", _chainable({"id": "a1", "user_id": "u1"})
    )
    agent = MagicMock()
    agent.aget_state = AsyncMock(return_value=SimpleNamespace(next=()))
    agent.ainvoke = AsyncMock()
    client = _make_client(agent)

    resp = client.post(
        "/meal-planner/approve",
        json={"thread_id": "t1", "decision": "approved"},
    )

    assert resp.status_code == 404
    agent.ainvoke.assert_not_awaited()


# --------------------------------------------------------------------------- #
# conflict resolution
# --------------------------------------------------------------------------- #
def test_resolve_conflict_accept_logs_recipe(monkeypatch):
    monkeypatch.setattr(
        meal_planner, "verify_plan_ownership", AsyncMock(return_value=True)
    )
    logged = AsyncMock(return_value=[{"id": "s1"}])
    monkeypatch.setattr(meal_planner, "log_recipe_to_slot", logged)
    client = _make_client(MagicMock())

    resp = client.post(
        "/meal-planner/resolve-conflict",
        json={
            "plan_id": "p1",
            "recipe": "paneer curry",
            "day_of_week": 0,
            "meal_type": "dinner",
            "decision": "accept",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["log_status"] == "logged"
    logged.assert_awaited_once_with("p1", "paneer curry", 0, "dinner")


def test_resolve_conflict_reject_records_dislike(monkeypatch):
    monkeypatch.setattr(
        meal_planner, "verify_plan_ownership", AsyncMock(return_value=True)
    )
    disliked = AsyncMock()
    logged = AsyncMock()
    monkeypatch.setattr(meal_planner, "add_disliked_dish", disliked)
    monkeypatch.setattr(meal_planner, "log_recipe_to_slot", logged)
    client = _make_client(MagicMock())

    resp = client.post(
        "/meal-planner/resolve-conflict",
        json={
            "plan_id": "p1",
            "recipe": "paneer curry",
            "day_of_week": 0,
            "meal_type": "dinner",
            "decision": "reject",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["log_status"] == "rejected"
    disliked.assert_awaited_once_with("u1", "paneer curry")
    logged.assert_not_awaited()


def test_resolve_conflict_foreign_plan_is_forbidden(monkeypatch):
    monkeypatch.setattr(
        meal_planner, "verify_plan_ownership", AsyncMock(return_value=False)
    )
    client = _make_client(MagicMock())

    resp = client.post(
        "/meal-planner/resolve-conflict",
        json={
            "plan_id": "not-mine",
            "recipe": "x",
            "day_of_week": 0,
            "meal_type": "dinner",
            "decision": "accept",
        },
    )

    assert resp.status_code == 403


# --------------------------------------------------------------------------- #
# disliked_dishes write path
# --------------------------------------------------------------------------- #
async def test_add_disliked_dish_merges_and_persists(monkeypatch):
    captured = {}

    async def fake_remember(uid, key, value):
        captured["uid"], captured["key"], captured["value"] = uid, key, value

    monkeypatch.setattr(
        meal_planner, "get_disliked_dishes", AsyncMock(return_value=["okra"])
    )
    monkeypatch.setattr(meal_planner, "remember", fake_remember)

    out = await meal_planner.add_disliked_dish("u1", "paneer curry")

    assert out == ["okra", "paneer curry"]
    assert captured == {
        "uid": "u1",
        "key": "disliked_dishes",
        "value": ["okra", "paneer curry"],
    }


async def test_add_disliked_dish_dedupes(monkeypatch):
    monkeypatch.setattr(
        meal_planner, "get_disliked_dishes", AsyncMock(return_value=["okra"])
    )
    monkeypatch.setattr(meal_planner, "remember", AsyncMock())

    out = await meal_planner.add_disliked_dish("u1", "okra")

    assert out == ["okra"]  # no duplicate


async def test_remove_disliked_dish(monkeypatch):
    monkeypatch.setattr(
        meal_planner,
        "get_disliked_dishes",
        AsyncMock(return_value=["okra", "paneer curry"]),
    )
    monkeypatch.setattr(meal_planner, "remember", AsyncMock())

    out = await meal_planner.remove_disliked_dish("u1", "okra")

    assert out == ["paneer curry"]


def test_disliked_endpoints(monkeypatch):
    monkeypatch.setattr(
        meal_planner, "get_disliked_dishes", AsyncMock(return_value=["okra"])
    )
    monkeypatch.setattr(
        meal_planner, "add_disliked_dish", AsyncMock(return_value=["okra", "tofu"])
    )
    monkeypatch.setattr(
        meal_planner, "remove_disliked_dish", AsyncMock(return_value=[])
    )
    client = _make_client(MagicMock())

    assert client.get("/meal-planner/disliked").json()["result"] == ["okra"]

    added = client.post("/meal-planner/disliked", json={"dish": "tofu"})
    assert added.json()["result"] == ["okra", "tofu"]

    removed = client.request(
        "DELETE", "/meal-planner/disliked", json={"dish": "okra"}
    )
    assert removed.json()["result"] == []


# --------------------------------------------------------------------------- #
# grocery list generation
# --------------------------------------------------------------------------- #
async def test_build_grocery_list_aggregates_by_slot(monkeypatch):
    # Oatmeal appears in two slots → its ingredients count twice.
    slots = [
        {"recipe_id": "r1", "recipe_name": "oatmeal"},
        {"recipe_id": "r1", "recipe_name": "oatmeal"},
    ]
    by_id = [
        {
            "id": "r1",
            "name": "oatmeal",
            "ingredients": [
                {"name": "Oats", "qty": 50, "unit": "g"},
                {"name": "Milk", "qty": 200, "unit": "ml"},
            ],
        }
    ]
    monkeypatch.setattr(
        meal_planner, "supabase", _supabase_seq(slots, by_id, [])
    )

    items = await meal_planner.build_grocery_list("p1")
    by_name = {i["name"]: i for i in items}

    assert by_name["Oats"]["qty"] == 100
    assert by_name["Oats"]["unit"] == "g"
    assert by_name["Milk"]["qty"] == 400
    assert all(i["checked"] is False for i in items)


async def test_build_grocery_list_resolves_by_name_when_no_recipe_id(monkeypatch):
    slots = [{"recipe_id": None, "recipe_name": "salad"}]
    by_name = [
        {
            "id": "r9",
            "name": "salad",
            "ingredients": [{"name": "Lettuce", "qty": 1, "unit": "head"}],
        }
    ]
    # ids is empty → only the slots query and the by-name query run.
    monkeypatch.setattr(meal_planner, "supabase", _supabase_seq(slots, by_name))

    items = await meal_planner.build_grocery_list("p1")

    assert items == [
        {"name": "Lettuce", "qty": 1, "unit": "head", "checked": False}
    ]


def test_grocery_list_endpoint_checks_ownership(monkeypatch):
    monkeypatch.setattr(
        meal_planner, "verify_plan_ownership", AsyncMock(return_value=False)
    )
    client = _make_client(MagicMock())

    resp = client.get("/meal-planner/grocery-list/p1")

    assert resp.status_code == 403


def test_grocery_list_endpoint_returns_items(monkeypatch):
    monkeypatch.setattr(
        meal_planner, "verify_plan_ownership", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        meal_planner,
        "build_grocery_list",
        AsyncMock(
            return_value=[{"name": "oats", "qty": 100, "unit": "g", "checked": False}]
        ),
    )
    client = _make_client(MagicMock())

    resp = client.get("/meal-planner/grocery-list/p1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["plan_id"] == "p1"
    assert body["result"][0]["name"] == "oats"
