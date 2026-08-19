"""Proof that per-user rate limiting is ENFORCED on the cost-bearing learning
endpoints — no network, no Redis, no keys.

Mirrors test_rag_rate_limit.py. The limiter is faked at the module level: one
variant records the call and raises 429, so we can assert the endpoint enforces
the limit — and does it BEFORE any spend — and one is a no-op, so we can assert a
request under the limit still proceeds.

Why these four routes and not the others: a chat turn, a digest, a checkpoint and
a Feynman judgement each cost model calls (a digest also costs a web search). The
reads cost a Mongo query. Grading costs arithmetic. The domain limits that exist
— CHECKPOINT_MAX_ATTEMPTS_PER_DAY, DIGEST_MAX_UNREAD — are per topic, so they
bound one topic rather than one account.

Routes are not the only door, which is the thing these tests exist to keep true.
The tutor's `pull_next_lesson` tool writes a digest straight from a chat turn, so
the digest cap is asserted on both paths and against one shared key — a cap on
the route alone left the same spend reachable under `learning_query`, which is
30 a minute rather than 20 an hour.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.agents.learning_tracker import triggers
from app.agents.learning_tracker.actions import build_action_tools
from app.core import config
from app.dependencies import get_current_user
from app.routers import learning_tracker as route
from app.services import rate_limit

_OID = "507f1f77bcf86cd799439011"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(route.router)
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={})
    agent.aget_state = AsyncMock(return_value=None)
    app.state.learning_agent = agent
    app.dependency_overrides[get_current_user] = lambda: {"uid": "u-rl"}
    return TestClient(app)


@pytest.fixture
def blocked(monkeypatch):
    """Make the per-user limiter record its args, then reject with 429 — as if the
    user is over their limit. Fires before any downstream work, so no model call,
    no web search and no Mongo read is reached."""
    recorded: list = []

    async def _capture_and_block(name, user_id, limit, window_seconds):
        recorded.append((name, user_id, limit, window_seconds))
        raise HTTPException(status_code=429, detail="Too many attempts.")

    monkeypatch.setattr(rate_limit, "limit_user", _capture_and_block)
    return recorded


@pytest.fixture
def digest_blocked(monkeypatch):
    """The digest cap, over its limit.

    Patches the non-raising variant, because the digest cap does not sit on the
    route: it lives in `pull_next_digest`, which the chat tool reaches directly.
    Returns the seconds to wait, as the real one does.
    """
    recorded: list = []

    async def _capture_and_block(name, user_id, limit, window_seconds):
        recorded.append((name, user_id, limit, window_seconds))
        return 900

    monkeypatch.setattr(rate_limit, "check_user", _capture_and_block)
    return recorded


@pytest.fixture
def allowed(monkeypatch):
    """The limiter as it behaves under the cap — and as it behaves during a Redis
    outage, which fails open."""
    async def _noop(*a, **kw):
        return None

    monkeypatch.setattr(rate_limit, "limit_user", _noop)
    monkeypatch.setattr(rate_limit, "check_user", _noop)


def test_a_chat_turn_is_capped(client, blocked):
    r = client.post("/learning/query", json={"text": "teach me rust"})
    assert r.status_code == 429
    assert blocked == [
        (
            "learning_query",
            "u-rl",
            config.LEARNING_QUERY_RATE_LIMIT,
            config.LEARNING_QUERY_RATE_WINDOW,
        )
    ]


def test_the_streaming_turn_shares_the_same_bucket(client, blocked):
    """Two doors onto one turn. Separate buckets would double the allowance for a
    client that alternates."""
    r = client.post("/learning/query/stream", json={"text": "teach me rust"})
    assert r.status_code == 429
    assert blocked[0][0] == "learning_query"


def test_pulling_a_digest_is_capped(client, digest_blocked):
    """The most expensive request in the feature: a web search plus two or three
    model calls."""
    r = client.post("/learning/digests/generate")
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "900"
    assert digest_blocked == [
        (
            "learning_digest",
            "u-rl",
            config.LEARNING_DIGEST_RATE_LIMIT,
            config.LEARNING_DIGEST_RATE_WINDOW,
        )
    ]


async def test_the_chat_tool_shares_the_digest_budget(digest_blocked, monkeypatch):
    """The second door onto the same spend, and the one that used to be open.

    `pull_next_lesson` calls `pull_next_digest` directly, so while the cap sat on
    the HTTP route the tool was bounded only by `learning_query` — 30 a minute,
    1800 an hour, ninety times the digest budget on the single most expensive
    operation in the product.
    """
    generated = AsyncMock()
    monkeypatch.setattr(triggers, "build_digest", generated)
    monkeypatch.setattr(triggers, "build_revision_digest", generated)
    monkeypatch.setattr(triggers, "fetch_roadmap", generated)

    tool = {t.name: t for t in build_action_tools("u-rl")}["pull_next_lesson"]
    out = await tool.ainvoke({})

    # A sentence the tutor can relay, not an exception that breaks the turn.
    assert "900s" in out
    # Nothing downstream was reached — not the search, not the model, not Mongo.
    generated.assert_not_awaited()
    assert digest_blocked == [
        (
            "learning_digest",
            "u-rl",
            config.LEARNING_DIGEST_RATE_LIMIT,
            config.LEARNING_DIGEST_RATE_WINDOW,
        )
    ]


def test_both_doors_spend_one_allowance(client, monkeypatch):
    """Route and tool must not get a bucket each — the key is what couples them,
    so a learner cannot double their budget by alternating."""
    keys: list = []

    async def _record(key, limit, window_seconds):
        keys.append(key)
        return None

    monkeypatch.setattr(rate_limit, "check", _record)
    monkeypatch.setattr(triggers, "fetch_roadmap", AsyncMock(return_value=None))

    client.post("/learning/digests/generate")
    tool = {t.name: t for t in build_action_tools("u-rl")}["pull_next_lesson"]
    asyncio.run(tool.ainvoke({}))

    assert keys == ["learning_digest:user:u-rl"] * 2


def test_issuing_a_checkpoint_is_capped(client, blocked):
    r = client.post(f"/learning/topics/t1/checkpoint", json={"roadmapId": _OID})
    assert r.status_code == 429
    assert blocked == [
        (
            "learning_checkpoint",
            "u-rl",
            config.LEARNING_CHECKPOINT_RATE_LIMIT,
            config.LEARNING_CHECKPOINT_RATE_WINDOW,
        )
    ]


def test_judging_an_explanation_is_capped(client, blocked):
    """The only cost-bearing route with no domain limit of its own — it is
    optional and never a gate, so nothing else bounds it."""
    r = client.post(
        f"/learning/topics/t1/explain",
        json={"roadmapId": _OID, "text": " ".join(["word"] * 40)},
    )
    assert r.status_code == 429
    assert blocked == [
        (
            "learning_explain",
            "u-rl",
            config.LEARNING_EXPLAIN_RATE_LIMIT,
            config.LEARNING_EXPLAIN_RATE_WINDOW,
        )
    ]


def test_the_limit_fires_before_the_graph_is_touched(client, blocked):
    """A 429 that arrives after the model call has already been paid for is not a
    spend limit."""
    client.post("/learning/query", json={"text": "teach me rust"})
    client.app.state.learning_agent.ainvoke.assert_not_awaited()


def test_reads_are_not_capped(client, blocked, monkeypatch):
    """A read costs a Mongo query. Rate-limiting the roadmap list would throttle
    scrolling, not spending."""
    monkeypatch.setattr(route, "list_roadmaps", AsyncMock(return_value=[]))
    assert client.get("/learning/roadmaps").status_code == 200
    assert blocked == []


def test_under_the_cap_the_request_proceeds(client, allowed, monkeypatch):
    """Also the Redis-outage path: the limiter fails open, and a learner must not
    be locked out of their own roadmap by a cache being down."""
    monkeypatch.setattr(route, "resolve_roadmap_id", AsyncMock(return_value=None))
    monkeypatch.setattr(route, "fetch_roadmap", AsyncMock(return_value=None))

    r = client.post("/learning/digests/generate")

    # Past the limiter and into the handler, which then declines on its own terms.
    assert r.status_code == 404
