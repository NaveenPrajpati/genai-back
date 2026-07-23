"""Offline tests for the Prometheus ops metrics — no network, no Redis, no keys.

Covers the parts that carry real risk of being silently wrong:
  * the pricing math + longest-prefix model matching (cost per query),
  * the cost callback reading BOTH usage shapes (streaming vs non-streaming),
  * the recording helpers moving the right counters,
  * the RAG stream route actually recording an outcome + a cache lookup.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from prometheus_client import REGISTRY

from app.core import metrics
from app.dependencies import get_current_user
from app.routers import rag


def _val(name: str, labels: dict) -> float:
    """Current value of a labelled sample, or 0.0 if it doesn't exist yet."""
    return REGISTRY.get_sample_value(name, labels) or 0.0


# ── pricing / cost math ──────────────────────────────────────────────────────
def test_cost_usd_pricing():
    # (1M in * $2.5) + (1M out * $10) = $12.5
    assert metrics.cost_usd("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.5)
    assert metrics.cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    # unknown / None model prices to zero and never raises.
    assert metrics.cost_usd("llama3:latest", 10, 10) == 0.0
    assert metrics.cost_usd(None, 10, 10) == 0.0


def test_price_key_longest_prefix():
    # A date-stamped snapshot must match its family — and mini must NOT be priced
    # as the (5x pricier) gpt-4o.
    assert metrics._price_key("gpt-4o-2024-08-06") == "gpt-4o"
    assert metrics._price_key("gpt-4o-mini-2024-07-18") == "gpt-4o-mini"
    assert metrics._price_key("text-embedding-3-small") == "text-embedding-3-small"
    assert metrics._price_key("some-other-model") is None


# ── cost callback: both usage shapes ─────────────────────────────────────────
def _resp(model, usage_meta=None, token_usage=None):
    msg = type("Msg", (), {
        "usage_metadata": usage_meta,
        "response_metadata": {"model_name": model},
    })()
    gen = type("Gen", (), {"message": msg})()
    return type("Resp", (), {
        "generations": [[gen]],
        "llm_output": {"model_name": model, "token_usage": token_usage or {}},
    })()


def test_cost_callback_streaming_usage():
    label = "gpt-4o"
    before_in = _val("llm_tokens_total", {"model": label, "type": "input"})
    before_cost = _val("llm_cost_usd_total", {"model": label})

    # streaming shape: usage rides on message.usage_metadata
    resp = _resp("gpt-4o", usage_meta={"input_tokens": 1000, "output_tokens": 500})
    metrics.CostCallback().on_llm_end(resp)

    assert _val("llm_tokens_total", {"model": label, "type": "input"}) == before_in + 1000
    # cost = (1000*2.5 + 500*10) / 1e6 = 0.0075
    assert _val("llm_cost_usd_total", {"model": label}) == pytest.approx(before_cost + 0.0075)


def test_cost_callback_nonstreaming_usage():
    label = "gpt-4o-mini"
    before_out = _val("llm_tokens_total", {"model": label, "type": "output"})
    # non-streaming shape: token_usage in llm_output, no usage_metadata
    resp = _resp("gpt-4o-mini", token_usage={"prompt_tokens": 200, "completion_tokens": 100})
    metrics.CostCallback().on_llm_end(resp)
    assert _val("llm_tokens_total", {"model": label, "type": "output"}) == before_out + 100


def test_cost_callback_never_raises_on_garbage():
    # A malformed response must not bubble an error into the LLM call.
    metrics.CostCallback().on_llm_end(object())
    metrics.CostCallback().on_llm_error(ValueError("boom"))


# ── recording helpers ────────────────────────────────────────────────────────
def test_record_helpers_move_counters():
    b_hit = _val("rag_cache_lookups_total", {"result": "hit"})
    metrics.record_cache(True)
    assert _val("rag_cache_lookups_total", {"result": "hit"}) == b_hit + 1

    b = _val("rag_requests_total", {"endpoint": "query_stream", "outcome": "refused"})
    metrics.record_request("query_stream", "refused", 1.23)
    assert _val("rag_requests_total", {"endpoint": "query_stream", "outcome": "refused"}) == b + 1


# ── integration: the stream route records outcome + cache lookup ─────────────
class _FakeChain:
    def __init__(self, chunks):
        self._chunks = chunks

    async def astream(self, _inputs):
        for c in self._chunks:
            yield type("Chunk", (), {"content": c})()


class _FakePrompt:
    def __init__(self, chunks):
        self._chunks = chunks

    def __or__(self, _llm):
        return _FakeChain(self._chunks)


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(rag.router)
    app.dependency_overrides[get_current_user] = lambda: {"uid": "u-metrics"}

    class _Emb:
        def embed_query(self, _q):
            return [0.1, 0.2]

    async def _miss(*_a, **_k):
        return None

    async def _noop(*_a, **_k):
        return None

    async def _scope_size(*_a, **_k):  # avoid the real Redis call
        return 0

    monkeypatch.setattr(rag, "get_embeddings", lambda: _Emb())
    monkeypatch.setattr(rag.cache, "lookup", _miss)
    monkeypatch.setattr(rag.cache, "save", _noop)
    monkeypatch.setattr(rag.cache, "scope_size", _scope_size)
    monkeypatch.setattr(rag.storage, "save_messages", lambda **_k: None)
    monkeypatch.setattr(
        rag.storage, "create_chat",
        lambda title, user_id=None: {"id": "c1", "title": title, "user_id": user_id},
    )
    docs = [Document(page_content="Alpha grew 12%.",
                     metadata={"source": "a.pdf", "doc_id": "d1", "relevance_score": 0.9})]
    monkeypatch.setattr(
        rag, "retrieve_and_rerank",
        lambda _u, _i, _q: (docs, {"retrieve_ms": 12.0, "rerank_ms": 8.0, "candidates": 5}),
    )

    async def _answerable(_q, _c):
        return True

    monkeypatch.setattr(rag, "is_answerable", _answerable)
    answer = "Alpha revenue grew 12% in 2023 [1]. Supported by the context."
    monkeypatch.setattr(rag, "RAG_ANSWER", _FakePrompt([answer[:30], answer[30:]]))
    return TestClient(app)


def test_stream_records_answered_and_cache_miss(client):
    ep = {"endpoint": "query_stream", "outcome": "answered"}
    before_answered = _val("rag_requests_total", ep)
    before_miss = _val("rag_cache_lookups_total", {"result": "miss"})
    before_embed = _val("rag_stage_latency_seconds_count", {"stage": "embed"})

    r = client.post("/rag/query/stream", json={"question": "what grew?"})
    assert r.status_code == 200
    # sanity: it actually reached the answered terminal
    assert any('"grounded": true' in line for line in r.text.splitlines())

    assert _val("rag_requests_total", ep) == before_answered + 1
    assert _val("rag_cache_lookups_total", {"result": "miss"}) == before_miss + 1
    # each real stage timing is also observed into the stage histogram
    assert _val("rag_stage_latency_seconds_count", {"stage": "embed"}) == before_embed + 1


# ── daily budget guard: accumulation, rollover, threshold ────────────────────
@pytest.fixture
def fresh_budget(monkeypatch):
    """Isolate the module-level daily accumulator for one test (other tests'
    priced calls also feed it, so start every budget test from zero)."""
    monkeypatch.setattr(metrics, "_budget_day", None)
    monkeypatch.setattr(metrics, "_budget_spend_usd", 0.0)


def test_add_daily_cost_accumulates_and_sets_gauge(fresh_budget):
    metrics.add_daily_cost(0.03)
    metrics.add_daily_cost(0.04)
    assert metrics.daily_spend_usd() == pytest.approx(0.07)
    # the gauge mirrors the running total (drives the "alert at 80% of cap" query)
    assert _val("llm_daily_spend_usd", {}) == pytest.approx(0.07)


def test_add_daily_cost_ignores_nonpositive(fresh_budget):
    metrics.add_daily_cost(0)
    metrics.add_daily_cost(-5)
    assert metrics.daily_spend_usd() == 0.0


def test_daily_spend_rolls_over_at_utc_midnight(fresh_budget, monkeypatch):
    day = {"v": "2026-07-23"}
    monkeypatch.setattr(metrics, "_utc_day", lambda: day["v"])
    metrics.add_daily_cost(0.05)
    assert metrics.daily_spend_usd() == pytest.approx(0.05)
    # cross UTC midnight: reads report 0 immediately, and the next add starts fresh
    day["v"] = "2026-07-24"
    assert metrics.daily_spend_usd() == 0.0
    metrics.add_daily_cost(0.02)
    assert metrics.daily_spend_usd() == pytest.approx(0.02)  # not 0.07


def test_budget_exceeded_trips_at_cap(fresh_budget, monkeypatch):
    monkeypatch.setattr(metrics, "LLM_DAILY_BUDGET_USD", 0.10)
    assert metrics.budget_exceeded() is False
    metrics.add_daily_cost(0.09)
    assert metrics.budget_exceeded() is False
    metrics.add_daily_cost(0.02)  # 0.11 >= 0.10 → tripped
    assert metrics.budget_exceeded() is True


def test_budget_disabled_never_trips(fresh_budget, monkeypatch):
    monkeypatch.setattr(metrics, "LLM_DAILY_BUDGET_USD", 0.0)  # cap off
    metrics.add_daily_cost(9999)
    assert metrics.budget_exceeded() is False


def test_cost_callback_feeds_daily_total(fresh_budget):
    # the on_llm_end hook must roll its priced cost into the daily accumulator
    resp = _resp("gpt-4o", usage_meta={"input_tokens": 1000, "output_tokens": 500})
    metrics.CostCallback().on_llm_end(resp)
    assert metrics.daily_spend_usd() == pytest.approx(0.0075)  # (1000*2.5+500*10)/1e6


# ── integration: the stream route refuses once the daily cap is hit ──────────
def test_stream_refuses_when_budget_exceeded(client, monkeypatch):
    # a cache miss (fixture) with the cap tripped must refuse before any LLM work
    monkeypatch.setattr(metrics, "budget_exceeded", lambda: True)
    ep = {"endpoint": "query_stream", "outcome": "budget_exceeded"}
    before = _val("rag_requests_total", ep)

    r = client.post("/rag/query/stream", json={"question": "what grew?"})
    assert r.status_code == 200
    assert any('"budget_exceeded": true' in line for line in r.text.splitlines())
    assert _val("rag_requests_total", ep) == before + 1
