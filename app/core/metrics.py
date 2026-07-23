"""
core/metrics.py
===============
OPERATIONAL metrics (Prometheus), distinct from LangSmith tracing.

LangSmith (core/observability.py) is opt-in per-request TRACING — great for
debugging one call, useless for "what's my error rate this hour". This module is
the ops layer: cheap, always-on counters/histograms exposed at GET /metrics and
scraped by Prometheus, so Grafana can chart the five things we actually alert on:

    1. latency per pipeline stage   -> rag_stage_latency_seconds{stage}
    2. error rate                   -> rag_requests_total{outcome="error"} / all
    3. cache hit rate               -> rag_cache_lookups_total{result}
    4. refusal rate                 -> rag_requests_total{outcome="refused"} / all
    5. cost per query               -> llm_cost_usd_total / rag_requests_total

WHY A CALLBACK FOR COST: token usage is only known to the model client, so cost
is captured the same way core/llm_capture.py captures prompts — a LangChain
callback attached to the shared `llm`/`fast_llm`. It propagates to every call
site (plain, with_structured_output, bind_tools) with zero agent changes. The
chat models dominate spend; embeddings ($0.02/1M) are left out as noise.

Everything here is wrapped so a metrics failure can NEVER break a real request.
The process runs ONE gunicorn worker on purpose (see Dockerfile), so a plain
in-process registry is correct — no multiprocess mode needed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from prometheus_client import Counter, Gauge, Histogram

from app.core.config import LLM_DAILY_BUDGET_USD

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metric definitions  (label sets kept low-cardinality on purpose — never label
# by user_id / question / chat_id, which would explode the series count)
# ─────────────────────────────────────────────────────────────────────────────

# Per-stage latency. Buckets span the ~1ms writes (cache/persist) to multi-second
# LLM stages (gate/stream), so p50/p95/p99 are all resolvable per stage.
_STAGE_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30)
RAG_STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds",
    "Latency of one RAG pipeline stage",
    ["stage"],  # embed | cache | retrieve | rerank | gate | stream | persist | evaluate
    buckets=_STAGE_BUCKETS,
)

# End-to-end request latency + count, labelled by outcome. `outcome` is the
# single source of truth for both error rate and refusal rate.
_REQUEST_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60)
RAG_REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "End-to-end RAG request latency",
    ["endpoint", "outcome"],
    buckets=_REQUEST_BUCKETS,
)
RAG_REQUESTS = Counter(
    "rag_requests_total",
    "RAG requests by terminal outcome",
    ["endpoint", "outcome"],  # answered | cache_hit | refused | no_docs | budget_exceeded | error
)

# Semantic-cache outcomes -> hit rate = hit / (hit + miss).
RAG_CACHE_LOOKUPS = Counter(
    "rag_cache_lookups_total",
    "Semantic cache lookups by result",
    ["result"],  # hit | miss
)

# Chunks flagged at ingestion as possible indirect prompt-injection, by matched
# pattern. Chunks are still ingested (flag-and-keep) — a rising rate is the alarm.
RAG_INJECTION_FLAGS = Counter(
    "rag_injection_flags_total",
    "Chunks flagged as possible prompt-injection at ingestion, by matched pattern",
    ["pattern"],
)

# LLM spend. Cost per query = rate(llm_cost_usd_total) / rate(rag_requests_total).
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "LLM tokens consumed",
    ["model", "type"],  # type: input | output
)
LLM_COST_USD = Counter(
    "llm_cost_usd_total",
    "Estimated LLM cost in USD (see pricing table)",
    ["model"],
)
LLM_CALLS = Counter(
    "llm_calls_total",
    "LLM calls by status",
    ["model", "status"],  # ok | error
)

# Running LLM spend for the current UTC day (drives the daily budget guard).
# A Gauge, not a Counter: it resets to 0 at UTC midnight rollover, so you can
# alert at e.g. 0.8 * LLM_DAILY_BUDGET_USD without rate() windows.
LLM_DAILY_SPEND_USD = Gauge(
    "llm_daily_spend_usd",
    "Estimated LLM spend so far today (UTC), USD",
)


# ─────────────────────────────────────────────────────────────────────────────
# Pricing  (USD per 1,000,000 tokens, as (input, output))
# ─────────────────────────────────────────────────────────────────────────────
# Keep this in sync with current OpenAI pricing. Override the whole table at
# deploy time without a code change via LLM_PRICING_JSON, e.g.
#   LLM_PRICING_JSON='{"gpt-4o": [2.5, 10.0], "gpt-4o-mini": [0.15, 0.6]}'
_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}


def _load_pricing() -> dict[str, tuple[float, float]]:
    raw = os.getenv("LLM_PRICING_JSON")
    if not raw:
        return dict(_DEFAULT_PRICING)
    try:
        parsed = json.loads(raw)
        return {k: (float(v[0]), float(v[1])) for k, v in parsed.items()}
    except Exception as e:  # pragma: no cover - defensive config parse
        logger.warning("LLM_PRICING_JSON ignored (parse error: %s)", e)
        return dict(_DEFAULT_PRICING)


_PRICING = _load_pricing()


def _price_key(model: Optional[str]) -> Optional[str]:
    """Match a concrete model id (often date-stamped, e.g. gpt-4o-2024-08-06) to a
    pricing entry by longest matching prefix, so new snapshots price correctly."""
    if not model:
        return None
    if model in _PRICING:
        return model
    best = None
    for key in _PRICING:
        if model.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return best


def cost_usd(model: Optional[str], input_tokens: int, output_tokens: int) -> float:
    """Estimated USD for one call. Returns 0.0 for models with no pricing entry."""
    key = _price_key(model)
    if key is None:
        return 0.0
    in_price, out_price = _PRICING[key]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Recording helpers  (never raise — a metrics bug must not break a request)
# ─────────────────────────────────────────────────────────────────────────────
def observe_stage(stage: str, seconds: float) -> None:
    try:
        RAG_STAGE_LATENCY.labels(stage=stage).observe(max(seconds, 0.0))
    except Exception:  # pragma: no cover - defensive
        logger.debug("observe_stage failed", exc_info=True)


def record_cache(hit: bool) -> None:
    try:
        RAG_CACHE_LOOKUPS.labels(result="hit" if hit else "miss").inc()
    except Exception:  # pragma: no cover - defensive
        logger.debug("record_cache failed", exc_info=True)


def record_request(endpoint: str, outcome: str, seconds: float) -> None:
    """Terminal record for one request: bumps the count and observes total latency
    under the same (endpoint, outcome) — the join key for error/refusal rates."""
    try:
        RAG_REQUESTS.labels(endpoint=endpoint, outcome=outcome).inc()
        RAG_REQUEST_LATENCY.labels(endpoint=endpoint, outcome=outcome).observe(
            max(seconds, 0.0)
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("record_request failed", exc_info=True)


def record_injection_flags(patterns) -> None:
    """Count one flagged chunk against each injection pattern it matched (see
    services/rag/content_safety.py). Never raises."""
    try:
        for name in patterns:
            RAG_INJECTION_FLAGS.labels(pattern=name).inc()
    except Exception:  # pragma: no cover - defensive
        logger.debug("record_injection_flags failed", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cost callback  (attached to the shared chat models in core/llm.py)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_usage(response: Any) -> tuple[Optional[str], int, int]:
    """(model, input_tokens, output_tokens) from an LLMResult, tolerant of both
    the non-streaming (`llm_output.token_usage`) and streaming
    (`message.usage_metadata`, requires stream_usage=True) shapes."""
    model: Optional[str] = None
    in_tok = out_tok = 0

    llm_output = getattr(response, "llm_output", None) or {}
    model = llm_output.get("model_name") or llm_output.get("model")
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if usage:
        in_tok = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        out_tok = usage.get("completion_tokens") or usage.get("output_tokens") or 0

    # Streaming path: usage rides on the aggregated message, not llm_output.
    try:
        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        if msg is not None:
            if not model:
                model = (getattr(msg, "response_metadata", {}) or {}).get("model_name")
            meta = getattr(msg, "usage_metadata", None)
            if meta and not (in_tok or out_tok):
                in_tok = meta.get("input_tokens", 0)
                out_tok = meta.get("output_tokens", 0)
    except Exception:  # pragma: no cover - defensive
        pass

    return model, int(in_tok or 0), int(out_tok or 0)


# ─────────────────────────────────────────────────────────────────────────────
# Daily budget guard  (in-process; global while the app runs one gunicorn worker
# — see LLM_DAILY_BUDGET_USD in core/config.py)
# ─────────────────────────────────────────────────────────────────────────────
_budget_lock = threading.Lock()
_budget_day: Optional[str] = None   # UTC "YYYY-MM-DD" the running total belongs to
_budget_spend_usd: float = 0.0      # accumulated estimated spend for _budget_day


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def add_daily_cost(usd: float) -> None:
    """Add one call's USD to today's running total, rolling over at UTC midnight.
    Called from the cost callback on every priced LLM call."""
    global _budget_day, _budget_spend_usd
    if usd <= 0:
        return
    with _budget_lock:
        today = _utc_day()
        if today != _budget_day:  # first call of a new UTC day → reset
            _budget_day = today
            _budget_spend_usd = 0.0
        _budget_spend_usd += usd
        total = _budget_spend_usd
    try:
        LLM_DAILY_SPEND_USD.set(total)
    except Exception:  # pragma: no cover - defensive
        logger.debug("LLM_DAILY_SPEND_USD.set failed", exc_info=True)


def daily_spend_usd() -> float:
    """Estimated LLM spend so far today (UTC). Returns 0.0 after a day rollover
    even before the next call records, so reads never report stale spend."""
    with _budget_lock:
        if _utc_day() != _budget_day:
            return 0.0
        return _budget_spend_usd


def budget_exceeded() -> bool:
    """True iff a positive daily cap is configured and today's spend has hit it.
    The guard the RAG routes check before doing paid LLM work."""
    return LLM_DAILY_BUDGET_USD > 0 and daily_spend_usd() >= LLM_DAILY_BUDGET_USD


class CostCallback(BaseCallbackHandler):
    """Records tokens + estimated cost for every LLM call. Errors are swallowed —
    this must never let a metrics problem surface into a real request."""

    raise_error = False

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            model, in_tok, out_tok = _extract_usage(response)
            label = model or "unknown"
            LLM_CALLS.labels(model=label, status="ok").inc()
            if in_tok:
                LLM_TOKENS.labels(model=label, type="input").inc(in_tok)
            if out_tok:
                LLM_TOKENS.labels(model=label, type="output").inc(out_tok)
            usd = cost_usd(model, in_tok, out_tok)
            if usd:
                LLM_COST_USD.labels(model=label).inc(usd)
                add_daily_cost(usd)
        except Exception:  # pragma: no cover - defensive
            logger.debug("CostCallback.on_llm_end failed", exc_info=True)

    def on_llm_error(self, error, **kwargs) -> None:
        try:
            LLM_CALLS.labels(model="unknown", status="error").inc()
        except Exception:  # pragma: no cover - defensive
            logger.debug("CostCallback.on_llm_error failed", exc_info=True)


def build_metrics_callbacks() -> list:
    """Callbacks to attach to the shared chat models. Always on (cheap, in-process)
    unless explicitly disabled with METRICS_ENABLED=0."""
    if os.getenv("METRICS_ENABLED", "1").lower() in ("0", "false", "no", "off"):
        logger.info("Prometheus cost capture disabled (METRICS_ENABLED=0)")
        return []
    return [CostCallback()]
