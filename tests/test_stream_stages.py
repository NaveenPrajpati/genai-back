"""Offline test for /rag/query/stream stage telemetry — no network, no keys.

Every external dependency (embeddings, cache, retriever+reranker, gate, LLM,
storage) is faked at the router-module level, then the SSE frames are parsed to
assert the protocol: real per-stage `stage` events in pipeline order, and a
`done` event carrying the timings summary.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.dependencies import get_current_user
from app.routers import rag

PIPELINE = ("embed", "cache", "retrieve", "rerank", "gate", "stream", "persist")


class _FakeChain:
    def __init__(self, chunks):
        self._chunks = chunks

    async def astream(self, _inputs):
        for c in self._chunks:
            yield type("Chunk", (), {"content": c})()


class _FakePrompt:
    """Stands in for RAG_ANSWER: `prompt | llm` returns the fake chain."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __or__(self, _llm):
        return _FakeChain(self._chunks)


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(rag.router)
    app.dependency_overrides[get_current_user] = lambda: {"uid": "user-1"}

    class _Emb:
        def embed_query(self, _q):
            return [0.1, 0.2]

    monkeypatch.setattr(rag, "get_embeddings", lambda: _Emb())

    async def _cache_miss(*_a, **_k):
        return None

    async def _cache_save(*_a, **_k):
        return None

    async def _scope_size(*_a, **_k):  # avoid the real Redis call (would return -1)
        return 0

    monkeypatch.setattr(rag.cache, "lookup", _cache_miss)
    monkeypatch.setattr(rag.cache, "save", _cache_save)
    monkeypatch.setattr(rag.cache, "scope_size", _scope_size)
    monkeypatch.setattr(rag.storage, "save_messages", lambda **_k: None)
    # The stream opens a chat up front when the request carries no chat_id, so
    # this must be faked too — unfaked it inserts a real row into Supabase.
    monkeypatch.setattr(
        rag.storage,
        "create_chat",
        lambda title, user_id=None: {
            "id": "chat-test-1",
            "title": title,
            "user_id": user_id,
        },
    )

    docs = [
        Document(
            page_content="Alpha revenue grew 12% in 2023.",
            metadata={"source": "a.pdf", "doc_id": "d1", "relevance_score": 0.9},
        )
    ]
    monkeypatch.setattr(
        rag,
        "retrieve_and_rerank",
        lambda _uid, _ids, _q: (docs, {"retrieve_ms": 12.0, "rerank_ms": 8.0, "candidates": 5}),
    )

    async def _answerable(_q, _c):
        return True

    monkeypatch.setattr(rag, "is_answerable", _answerable)

    # Two chunks; the first alone exceeds the sentinel length so streaming opens.
    answer = "Alpha revenue grew 12% in 2023 [1]. This is supported by the context."
    monkeypatch.setattr(rag, "RAG_ANSWER", _FakePrompt([answer[:30], answer[30:]]))

    return TestClient(app)


def _events(text: str) -> list[dict]:
    return [json.loads(l[len("data: ") :]) for l in text.splitlines() if l.startswith("data: ")]


def test_stream_emits_real_stage_timings(client):
    r = client.post("/rag/query/stream", json={"question": "what grew?"})
    assert r.status_code == 200
    events = _events(r.text)

    stages = {e["name"]: e for e in events if e["type"] == "stage"}
    assert set(stages) == set(PIPELINE)

    # retrieve/rerank carry the timings measured in retrieval (faked here).
    assert stages["retrieve"]["ms"] == 12.0
    assert stages["retrieve"]["info"] == "5 candidates"
    assert stages["rerank"]["ms"] == 8.0
    # A miss reports the scope it looked in and how many entries that scope held,
    # so a cache that never hits can be told apart from a scope that never matched.
    assert stages["cache"]["info"].startswith("miss")
    assert "scope=user-1::__all__" in stages["cache"]["info"]
    assert "entries=0" in stages["cache"]["info"]
    assert stages["gate"]["info"] == "answerable"
    assert all("ms" in s for s in stages.values())

    # stage events arrive in pipeline order.
    ordered = [e["name"] for e in events if e["type"] == "stage"]
    assert ordered == list(PIPELINE)

    # done carries the summary and grounded flag (answer cites [1]).
    done = next(e for e in events if e["type"] == "done")
    assert done["grounded"] is True
    assert done["total_ms"] > 0
    assert set(done["timings"]) == set(PIPELINE)

    # the answer itself still streams, after sources.
    types = [e["type"] for e in events]
    assert "sources" in types and "token" in types and "citations" in types
    assert types.index("sources") < types.index("token")


def test_stream_refusal_skips_generation_stage(client, monkeypatch):
    async def _not_answerable(_q, _c):
        return False

    monkeypatch.setattr(rag, "is_answerable", _not_answerable)

    r = client.post("/rag/query/stream", json={"question": "unrelated?"})
    events = _events(r.text)

    stages = {e["name"]: e for e in events if e["type"] == "stage"}
    assert stages["gate"]["info"] == "refused"
    assert stages["stream"].get("skipped") is True  # no generation ran

    done = next(e for e in events if e["type"] == "done")
    assert done["grounded"] is False
    # the refusal message is still delivered as a token.
    assert any(e["type"] == "token" for e in events)
