"""Offline tests for external-call resilience — no network, no Redis, no keys.

Covers the two guards added for hung/failing upstreams:
  * Jina reranker: an HTTP timeout on every call (via _TimeoutSession) plus a
    fail-OPEN wrapper (_ResilientReranker) that degrades to the unreranked
    candidates instead of failing the query — rerank is quality, not correctness.
  * Redis cache: the client carries socket read/connect timeouts, so a stalled
    connection raises TimeoutError, which the cache's try/except degrades to a
    miss (a true fail-open, not a hang).
"""

import requests

from app.services import cache
from app.services.rag import step4_retrieval as s4


# ── Jina reranker: fail-open wrapper ─────────────────────────────────────────
class _OkJina(s4.JinaRerank):
    """A JinaRerank whose remote call is replaced by a deterministic local one.
    Returns just the first doc so a pass-through is distinguishable from the
    fallback (which returns top_n)."""

    def compress_documents(self, documents, query, callbacks=None):
        return list(documents)[:1]


class _BoomJina(s4.JinaRerank):
    """A JinaRerank whose call always fails — stands in for a down/timed-out API."""

    def compress_documents(self, documents, query, callbacks=None):
        raise RuntimeError("jina down / timeout")


def _docs(n: int):
    from langchain_core.documents import Document

    return [Document(page_content=f"chunk {i}") for i in range(n)]


def test_reranker_passes_through_when_jina_succeeds():
    inner = _OkJina(model="m", top_n=3, jina_api_key="dummy")
    wrapped = s4._ResilientReranker(reranker=inner, top_n=3)

    out = wrapped.compress_documents(_docs(5), "q")
    # got the reranker's own output (1 doc), NOT the 3-doc fallback
    assert len(out) == 1


def test_reranker_falls_back_to_unreranked_on_failure():
    inner = _BoomJina(model="m", top_n=3, jina_api_key="dummy")
    wrapped = s4._ResilientReranker(reranker=inner, top_n=3)

    docs = _docs(5)
    out = wrapped.compress_documents(docs, "q")  # must NOT raise
    # degraded to the retriever's own top-3, in original (hybrid) order
    assert [d.page_content for d in out] == [d.page_content for d in docs[:3]]


def test_reranker_fallback_truncates_to_available_docs():
    inner = _BoomJina(model="m", top_n=3, jina_api_key="dummy")
    wrapped = s4._ResilientReranker(reranker=inner, top_n=3)
    # fewer candidates than top_n → return them all, still no raise
    assert len(wrapped.compress_documents(_docs(2), "q")) == 2


# ── Jina reranker: timeout wiring ────────────────────────────────────────────
def test_get_reranker_wires_timeout_session_and_wrapper(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    s4._get_reranker.cache_clear()  # force a rebuild under the dummy key
    try:
        r = s4._get_reranker()
        assert isinstance(r, s4._ResilientReranker)
        assert isinstance(r.reranker.session, s4._TimeoutSession)
        assert r.reranker.session._timeout == s4.JINA_RERANK_TIMEOUT
        # the auth header the JinaRerank validator set survives the session swap
        assert "Authorization" in r.reranker.session.headers
    finally:
        s4._get_reranker.cache_clear()  # don't leak the dummy-key client


def test_timeout_session_injects_default_timeout(monkeypatch):
    captured = {}

    def spy(self, *args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        raise RuntimeError("stop before network")

    monkeypatch.setattr(requests.Session, "request", spy)

    # no explicit timeout → the session's default is stamped on
    try:
        s4._TimeoutSession(3.5).get("http://example.invalid")
    except RuntimeError:
        pass
    assert captured["timeout"] == 3.5

    # an explicit timeout from the caller is respected, not overridden (setdefault)
    try:
        s4._TimeoutSession(3.5).get("http://example.invalid", timeout=1.0)
    except RuntimeError:
        pass
    assert captured["timeout"] == 1.0


def test_timeout_session_sizes_connection_pool():
    # concurrent reranks would otherwise churn the requests default pool of 10
    session = s4._TimeoutSession(1.0)
    adapter = session.get_adapter("https://api.jina.ai/v1/rerank")
    assert adapter._pool_maxsize == s4.JINA_POOL_MAXSIZE


# ── Cold-start warmup ────────────────────────────────────────────────────────
def test_warmup_builds_all_retrieval_singletons(monkeypatch):
    called = []
    monkeypatch.setattr(s4, "get_embeddings", lambda: called.append("emb"))
    monkeypatch.setattr(s4, "_get_bm25_encoder", lambda: called.append("bm25"))
    monkeypatch.setattr(s4, "get_pinecone_index", lambda: called.append("pinecone"))
    monkeypatch.setattr(s4, "_get_reranker", lambda: called.append("reranker"))

    s4.warmup()
    assert called == ["emb", "bm25", "pinecone", "reranker"]


def test_warmup_survives_reranker_failure(monkeypatch):
    # a missing JINA_API_KEY must not abort warmup — the heavy parts still build
    called = []
    monkeypatch.setattr(s4, "get_embeddings", lambda: called.append("emb"))
    monkeypatch.setattr(s4, "_get_bm25_encoder", lambda: called.append("bm25"))
    monkeypatch.setattr(s4, "get_pinecone_index", lambda: called.append("pinecone"))

    def _boom():
        raise RuntimeError("no JINA_API_KEY")

    monkeypatch.setattr(s4, "_get_reranker", _boom)
    s4.warmup()  # must not raise
    assert called == ["emb", "bm25", "pinecone"]


# ── Redis cache: timeout wiring + fail-open on the resulting error ───────────
def test_redis_client_configured_with_socket_timeouts():
    from app.core.config import redis_client, REDIS_SOCKET_TIMEOUT

    ck = redis_client.connection_pool.connection_kwargs
    # without these a stalled connection hangs forever (try/except can't catch a hang)
    assert ck.get("socket_timeout") == REDIS_SOCKET_TIMEOUT
    assert ck.get("socket_connect_timeout") == REDIS_SOCKET_TIMEOUT


async def test_cache_lookup_degrades_to_miss_on_redis_timeout(monkeypatch):
    import redis.exceptions

    class _TimingOutRedis:
        async def smembers(self, key):
            raise redis.exceptions.TimeoutError("timed out")

    monkeypatch.setattr(cache, "redis_client", _TimingOutRedis())
    # the socket timeout turns a hang into this error; the cache must swallow it
    out = await cache.lookup_value([0.1, 0.2], "some-scope", 0.9)
    assert out is None  # miss, never raises → request proceeds without cache


async def test_scope_size_degrades_on_redis_timeout(monkeypatch):
    import redis.exceptions

    class _TimingOutRedis:
        async def scard(self, key):
            raise redis.exceptions.TimeoutError("timed out")

    monkeypatch.setattr(cache, "redis_client", _TimingOutRedis())
    assert await cache.scope_size("some-scope") == -1  # diagnostic sentinel, no raise
