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
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

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
def allowed(monkeypatch):
    """The limiter as it behaves under the cap — and as it behaves during a Redis
    outage, which fails open."""
    async def _noop(*a, **kw):
        return None

    monkeypatch.setattr(rate_limit, "limit_user", _noop)


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


def test_pulling_a_digest_is_capped(client, blocked):
    """The most expensive request in the feature: a web search plus two or three
    model calls."""
    r = client.post("/learning/digests/generate")
    assert r.status_code == 429
    assert blocked == [
        (
            "learning_digest",
            "u-rl",
            config.LEARNING_DIGEST_RATE_LIMIT,
            config.LEARNING_DIGEST_RATE_WINDOW,
        )
    ]


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
