"""Tests for the meal-planner agent and HTTP endpoints.

No real LLM calls or Supabase access — the LLM chain, the compiled LangGraph
agent, and the Supabase client are all replaced with fakes so the tests are
deterministic and offline.

Agent logic lives in the app.agents.meal_planner package; the router is a thin
HTTP layer. Tests patch the module where each symbol is looked up at call time:
node logic in `workflow`, persistence in `repository`, endpoints in the router.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END

import app.agents.meal_planner.workflow as mp
import app.agents.meal_planner.repository as mp_repo
import app.routers.meal_planner as mp_router
from app.dependencies import get_current_user


def _chainable(execute_data):
    """A Supabase-client stand-in whose query builder returns itself for every
    chained call and yields `execute_data` from .execute()."""
    q = MagicMock()
    for meth in (
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
    ):
        getattr(q, meth).return_value = q
    q.execute.side_effect = [SimpleNamespace(data=d) for d in results]
    supa = MagicMock()
    supa.table.return_value = q
    return supa


def _make_client(agent, user=None):
    app = FastAPI()
    app.include_router(mp_router.router)
    app.state.meal_agent = agent
    app.dependency_overrides[get_current_user] = lambda: user or {"uid": "u1"}
    return TestClient(app)


# --------------------------------------------------------------------------- #
# intent classifier + routing
# --------------------------------------------------------------------------- #
async def test_classify_intent_returns_llm_intent(monkeypatch):
    # classify_intent runs the fast model behind the semantic cache. Patch the
    # fast model and bypass the cache so the test stays offline & deterministic.
    fake_chain = RunnableLambda(lambda _: mp.IntentOutput(intent="plan"))
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = fake_chain
    monkeypatch.setattr(mp, "fast_llm", mock_llm)

    async def _no_cache(text, scope, threshold, produce):
        return await produce()

    monkeypatch.setattr(mp, "cached_value", _no_cache)

    out = await mp.classify_intent({"query": "plan my whole week"})

    assert out == {"intent": "plan"}
    mock_llm.with_structured_output.assert_called_once_with(mp.IntentOutput)


def test_decide_agent_routes_each_intent():
    assert mp.decide_agent({"intent": "log"}) == "log_agent"
    assert mp.decide_agent({"intent": "query"}) == "query_agent"
    assert mp.decide_agent({"intent": "research"}) == "research_agent"
    assert mp.decide_agent({"intent": "plan"}) == "plan_agent"


def test_decide_agent_unknown_intent_ends():
    assert mp.decide_agent({"intent": "nonsense"}) == END
    assert mp.decide_agent({}) == END


# --------------------------------------------------------------------------- #
# approve / resume cycle
# --------------------------------------------------------------------------- #
def test_approve_happy_path_resumes_agent(monkeypatch):
    monkeypatch.setattr(
        mp_router, "get_pending", AsyncMock(return_value={"_id": "a1", "user_id": "u1"})
    )
    agent = MagicMock()
    agent.aget_state = AsyncMock(return_value=SimpleNamespace(next=("plan_agent",)))
    agent.ainvoke = AsyncMock(return_value={"plan_status": "approved", "plan_id": "p1"})
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
        mp_router,
        "get_pending",
        AsyncMock(return_value={"_id": "a1", "user_id": "someone_else"}),
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
    monkeypatch.setattr(mp_router, "get_pending", AsyncMock(return_value=None))
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
        mp_router, "get_pending", AsyncMock(return_value={"_id": "a1", "user_id": "u1"})
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
        mp_router, "verify_plan_ownership", AsyncMock(return_value=True)
    )
    logged = AsyncMock(return_value=[{"id": "s1"}])
    monkeypatch.setattr(mp_router, "log_recipe_to_slot", logged)
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
        mp_router, "verify_plan_ownership", AsyncMock(return_value=True)
    )
    disliked = AsyncMock()
    logged = AsyncMock()
    monkeypatch.setattr(mp_router, "add_disliked_dish", disliked)
    monkeypatch.setattr(mp_router, "log_recipe_to_slot", logged)
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
        mp_router, "verify_plan_ownership", AsyncMock(return_value=False)
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
        mp_repo, "get_disliked_dishes", AsyncMock(return_value=["okra"])
    )
    monkeypatch.setattr(mp_repo, "remember", fake_remember)

    out = await mp_repo.add_disliked_dish("u1", "paneer curry")

    assert out == ["okra", "paneer curry"]
    assert captured == {
        "uid": "u1",
        "key": "disliked_dishes",
        "value": ["okra", "paneer curry"],
    }


async def test_add_disliked_dish_dedupes(monkeypatch):
    monkeypatch.setattr(
        mp_repo, "get_disliked_dishes", AsyncMock(return_value=["okra"])
    )
    monkeypatch.setattr(mp_repo, "remember", AsyncMock())

    out = await mp_repo.add_disliked_dish("u1", "okra")

    assert out == ["okra"]  # no duplicate


async def test_remove_disliked_dish(monkeypatch):
    monkeypatch.setattr(
        mp_repo,
        "get_disliked_dishes",
        AsyncMock(return_value=["okra", "paneer curry"]),
    )
    monkeypatch.setattr(mp_repo, "remember", AsyncMock())

    out = await mp_repo.remove_disliked_dish("u1", "okra")

    assert out == ["paneer curry"]


def test_disliked_endpoints(monkeypatch):
    monkeypatch.setattr(
        mp_router, "get_disliked_dishes", AsyncMock(return_value=["okra"])
    )
    monkeypatch.setattr(
        mp_router, "add_disliked_dish", AsyncMock(return_value=["okra", "tofu"])
    )
    monkeypatch.setattr(mp_router, "remove_disliked_dish", AsyncMock(return_value=[]))
    client = _make_client(MagicMock())

    assert client.get("/meal-planner/disliked").json()["result"] == ["okra"]

    added = client.post("/meal-planner/disliked", json={"dish": "tofu"})
    assert added.json()["result"] == ["okra", "tofu"]

    removed = client.request("DELETE", "/meal-planner/disliked", json={"dish": "okra"})
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
    monkeypatch.setattr(mp_repo, "supabase", _supabase_seq(slots, by_id, []))

    items = await mp_repo.build_grocery_list("p1")
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
    monkeypatch.setattr(mp_repo, "supabase", _supabase_seq(slots, by_name))

    items = await mp_repo.build_grocery_list("p1")

    assert items == [{"name": "Lettuce", "qty": 1, "unit": "head", "checked": False}]


def test_grocery_list_endpoint_checks_ownership(monkeypatch):
    monkeypatch.setattr(
        mp_router, "verify_plan_ownership", AsyncMock(return_value=False)
    )
    client = _make_client(MagicMock())

    resp = client.get("/meal-planner/grocery-list/p1")

    assert resp.status_code == 403


def test_grocery_list_endpoint_returns_items(monkeypatch):
    monkeypatch.setattr(
        mp_router, "verify_plan_ownership", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        mp_router,
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
