"""Proof that per-user rate limiting is ENFORCED on the RAG endpoints — no
network, no Redis, no keys.

The limiter (services/rate_limit.limit_user) is faked at the router-module level:
one variant records the call and raises 429 (so we can assert the endpoint both
enforces the limit and uses the right key/limits, before doing any work), and one
is a no-op (so we can assert a request under the limit still proceeds).
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.core import config
from app.dependencies import get_current_user
from app.routers import rag
from app.services import rate_limit


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(rag.router)
    app.dependency_overrides[get_current_user] = lambda: {"uid": "u-rl"}
    return TestClient(app)


@pytest.fixture
def blocked(monkeypatch):
    """Make the per-user limiter record its args, then reject with 429 — as if the
    user is over their limit. Fires before any downstream work, so no LLM/Pinecone/
    storage is touched."""
    recorded: list = []

    async def _capture_and_block(name, user_id, limit, window_seconds):
        recorded.append((name, user_id, limit, window_seconds))
        raise HTTPException(status_code=429, detail="Too many attempts.")

    monkeypatch.setattr(rate_limit, "limit_user", _capture_and_block)
    return recorded


def test_query_stream_is_rate_limited(client, blocked):
    r = client.post("/rag/query/stream", json={"question": "hi"})
    assert r.status_code == 429
    # the answer route is capped per user under the "rag_query" bucket, before
    # the stream starts
    assert blocked == [
        ("rag_query", "u-rl", config.RAG_QUERY_RATE_LIMIT, config.RAG_QUERY_RATE_WINDOW)
    ]


def test_ingest_is_rate_limited(client, blocked):
    # the 429 fires before form validation/work, so a minimal body is enough
    r = client.post("/rag/ingest/url", data={"data": "{}"})
    assert r.status_code == 429
    assert blocked == [
        ("rag_ingest", "u-rl", config.RAG_INGEST_RATE_LIMIT, config.RAG_INGEST_RATE_WINDOW)
    ]


def test_delete_is_rate_limited(client, blocked):
    # limiter runs before the ownership lookup, so no storage is touched
    r = client.delete("/rag/ingest/some-doc")
    assert r.status_code == 429
    assert blocked == [
        ("rag_delete", "u-rl", config.RAG_DELETE_RATE_LIMIT, config.RAG_DELETE_RATE_WINDOW)
    ]


def test_query_stream_proceeds_when_under_limit(client, monkeypatch):
    """Under the limit, the request proceeds — the limiter is a gate, not a wall."""

    async def _allow(*_a, **_k):
        return None

    async def _amiss(*_a, **_k):  # cache miss
        return None

    async def _scope0(*_a, **_k):
        return 0

    class _Emb:
        def embed_query(self, _q):
            return [0.1, 0.2]

    monkeypatch.setattr(rate_limit, "limit_user", _allow)
    monkeypatch.setattr(rag.metrics, "budget_exceeded", lambda: False)
    monkeypatch.setattr(rag, "get_embeddings", lambda: _Emb())
    monkeypatch.setattr(rag.cache, "lookup", _amiss)
    monkeypatch.setattr(rag.cache, "scope_size", _scope0)
    monkeypatch.setattr(
        rag,
        "retrieve_and_rerank",
        lambda *_a, **_k: ([], {"retrieve_ms": 0.0, "rerank_ms": 0.0, "candidates": 0}),
    )

    # chat_id supplied so no chat row is created; no docs → the pipeline emits its
    # "no documents" frame, proving we got past the limiter into the route
    r = client.post("/rag/query/stream", json={"question": "hi", "chat_id": "c1"})
    assert r.status_code == 200
    assert "No relevant documents" in r.text
