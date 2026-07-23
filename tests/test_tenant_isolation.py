"""Multi-tenant isolation proof — no network, no keys.

The invariant under test: a query can NEVER retrieve another user's chunks. It
rests on three things, each asserted here so a future refactor can't silently
break isolation:
  1. `_scope_filter` ALWAYS pins `user_id`, and doc narrowing is AND'd under it
     (so asking for another user's doc_id returns nothing, never a leak).
  2. `_base_hybrid_retriever` binds that user-scoped filter onto every retrieval.
  3. The /query route scopes by the TOKEN identity, never anything the client
     can put in the request body.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import get_current_user
from app.routers import rag
from app.services.rag import step4_retrieval as s4


# ── 1. the filter itself always carries the caller's user_id ─────────────────
def test_scope_filter_always_includes_user():
    assert s4._scope_filter("userA", None) == {"user_id": "userA"}
    # empty ingestions means "all of MY docs", never a global/unscoped search
    assert s4._scope_filter("userA", []) == {"user_id": "userA"}


def test_scope_filter_keeps_user_when_narrowing_to_docs():
    f = s4._scope_filter("userA", ["d1", "d2"])
    # user_id is never dropped or overridden by doc scoping; doc_id is AND'd on
    assert f == {"user_id": "userA", "doc_id": {"$in": ["d1", "d2"]}}


# ── 2. the retriever is bound to that user-scoped filter ─────────────────────
class _CapturingRetriever:
    """Stands in for PineconeHybridSearchRetriever: records the bound filter
    instead of talking to Pinecone."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.bound_filter = None

    def bind(self, **kw):
        self.bound_filter = kw.get("filter")
        return self


def _patch_retriever(monkeypatch):
    monkeypatch.setattr(s4, "PineconeHybridSearchRetriever", _CapturingRetriever)
    monkeypatch.setattr(s4, "get_embeddings", lambda: "emb")
    monkeypatch.setattr(s4, "_get_bm25_encoder", lambda: "bm25")
    monkeypatch.setattr(s4, "get_pinecone_index", lambda: "index")


def test_base_retriever_binds_caller_user_filter(monkeypatch):
    _patch_retriever(monkeypatch)
    a = s4._base_hybrid_retriever("userA", None)
    b = s4._base_hybrid_retriever("userB", None)
    assert a.bound_filter == {"user_id": "userA"}
    assert b.bound_filter == {"user_id": "userB"}
    # userB's retrieval can never carry userA's scope
    assert "userA" not in str(b.bound_filter)


def test_requesting_another_users_doc_stays_scoped_to_caller(monkeypatch):
    _patch_retriever(monkeypatch)
    # adversarial: userB asks for a doc that belongs to userA
    r = s4._base_hybrid_retriever("userB", ["userA-doc"])
    # the filter is AND'd under user_id=userB, so Pinecone matches nothing —
    # cross-user access is impossible even when explicitly requested
    assert r.bound_filter == {"user_id": "userB", "doc_id": {"$in": ["userA-doc"]}}


# ── 3. the route scopes by token identity, not by request body ───────────────
def test_stream_route_scopes_to_token_user_not_request(monkeypatch):
    app = FastAPI()
    app.include_router(rag.router)
    app.dependency_overrides[get_current_user] = lambda: {"uid": "token-user"}

    async def _noop(*a, **k):
        return None

    async def _amiss(*a, **k):  # cache miss
        return None

    async def _scope0(*a, **k):
        return 0

    class _Emb:
        def embed_query(self, _q):
            return [0.1, 0.2]

    captured: dict = {}

    def _fake_retrieve(user_id, doc_ids, question):
        captured["user_id"] = user_id
        captured["doc_ids"] = doc_ids
        return [], {"retrieve_ms": 0.0, "rerank_ms": 0.0, "candidates": 0}

    monkeypatch.setattr(rag.rate_limit, "limit_user", _noop)
    monkeypatch.setattr(rag.metrics, "budget_exceeded", lambda: False)
    monkeypatch.setattr(rag, "get_embeddings", lambda: _Emb())
    monkeypatch.setattr(rag.cache, "lookup", _amiss)
    monkeypatch.setattr(rag.cache, "scope_size", _scope0)
    monkeypatch.setattr(rag, "retrieve_and_rerank", _fake_retrieve)

    client = TestClient(app)
    # the body carries another user's doc id — it must not change WHOSE vectors
    # are searched (chat_id supplied so no chat row is created)
    r = client.post(
        "/rag/query/stream",
        json={"question": "secrets?", "ingestions": ["victim-doc"], "chat_id": "c1"},
    )
    assert r.status_code == 200
    # retrieval is scoped to the authenticated token, never the body
    assert captured["user_id"] == "token-user"
    # the body can only narrow to doc_ids, which _scope_filter AND's under user_id
    assert captured["doc_ids"] == ["victim-doc"]
