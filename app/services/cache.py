"""
services/cache.py
=================
SEMANTIC CACHE (a performance/cost optimisation that sits IN FRONT of retrieval).

Plain caches key on exact strings, so "What is RAG?" and "what's RAG" miss each
other. A SEMANTIC cache keys on the query EMBEDDING: if a new question is within
CACHE_SIMILARITY_THRESHOLD cosine of a cached one (in the same scope), we replay
the stored answer + sources instead of re-running retrieval and the LLM.

WHY SCOPED: a cached answer is only valid for the same set of documents. We key
each cache entry by a "scope" string derived from the requested doc_ids, so a
question scoped to doc A never returns an answer computed over docs A+B+C.

WHEN TO TUNE / REPLACE:
  • Threshold too low → wrong answers served (false hits). Too high → few hits.
    0.95 is conservative; measure hit-rate vs. user complaints and adjust.
  • This brute-forces cosine over all keys in a scope (O(n) per lookup). Fine for
    hundreds of entries; for scale, store cache embeddings in a vector index too
    (e.g. a dedicated Pinecone namespace or Redis Vector Search) and ANN-search them.
  • Consider GPTCache (a purpose-built semantic cache library) if this grows.
"""

import json
import uuid
import logging
from typing import Optional, List

import numpy as np

from app.core.config import (
    redis_client,
    CACHE_PREFIX,
    CACHE_INDEX_PREFIX,
    CACHE_TTL_SECONDS,
    CACHE_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


def scope_key(user_id: str, doc_ids: List[str]) -> str:
    """Stable identifier for (who is asking, over which sources) — order-independent.

    The `user_id` prefix keeps the cache multi-tenant: an empty `doc_ids` maps to
    this user's `__all__`, never a global one, so a cached answer computed over
    user A's documents can never be replayed to user B.
    """
    sources = "|".join(sorted(doc_ids)) if doc_ids else "__all__"
    return f"{user_id}::{sources}"


async def scope_size(scope: str) -> int:
    """How many entries are cached under `scope`. Diagnostic only: 0 means the
    scope key itself didn't match any earlier request (usually a changed doc
    selection), which is a different problem from entries that were too
    dissimilar. Never raises."""
    try:
        return await redis_client.scard(f"{CACHE_INDEX_PREFIX}{scope}")
    except Exception:
        return -1


async def invalidate_user(user_id: str) -> int:
    """Drop every cached answer belonging to `user_id`. Returns scopes cleared.

    Called after a user INGESTS or DELETES a document: their cached answers were
    computed over the OLD document set, so replaying them would hide freshly-added
    content (or resurrect deleted content) until the entries expire on their own
    (CACHE_TTL_SECONDS = 24h). We clear the whole user namespace rather than reason
    about exactly which scopes went stale — ingest/delete are rare, and a dropped
    entry simply gets recomputed and re-cached on the next question.

    `user_id` is a UUID/ObjectId hex (no Redis glob metacharacters), so it's safe
    to interpolate into the SCAN match. Never raises — a cache-clear failure must
    not break ingestion.
    """
    cleared = 0
    try:
        pattern = f"{CACHE_INDEX_PREFIX}{user_id}::*"
        async for index_key in redis_client.scan_iter(match=pattern):
            members = await redis_client.smembers(index_key)
            if members:
                await redis_client.delete(*members)  # the cached payloads
            await redis_client.delete(index_key)  # the scope's index set
            cleared += 1
        if cleared:
            logger.info("cache invalidated for user=%s (%d scopes)", user_id, cleared)
    except Exception:
        logger.warning("cache invalidation failed for user=%s", user_id)
    return cleared


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


# --------------------------------------------------------------------------- #
# Generic value store: cache ANY JSON-serializable payload keyed by query
# embedding within a scope. RAG and the agents share this one mechanism; the
# `threshold` is passed per call so callers tune strictness by task type.
# --------------------------------------------------------------------------- #
async def lookup_value(
    query_embedding: List[float], scope: str, threshold: float
) -> Optional[dict]:
    """Return the stored `value` of the most semantically-similar query in
    `scope` if it clears `threshold`, else None."""
    try:
        cache_keys = await redis_client.smembers(f"{CACHE_INDEX_PREFIX}{scope}")
        if not cache_keys:
            return None

        best_sim, best_value = 0.0, None
        for key in cache_keys:
            raw = await redis_client.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            sim = _cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim, best_value = sim, entry.get("value")

        # A miss has two very different causes that look identical from outside:
        # no entries in this scope (the scope string didn't match a previous
        # request), or entries that were simply too dissimilar. Log both numbers
        # so "always misses" is diagnosable without a debugger.
        if best_sim < threshold:
            logger.info(
                "cache miss scope=%s candidates=%d best_sim=%.3f threshold=%.2f",
                scope, len(cache_keys), best_sim, threshold,
            )
        return best_value if best_sim >= threshold else None
    except Exception:
        # Cache must NEVER break a request — degrade gracefully to a cache miss.
        logger.warning("Redis cache lookup failed — proceeding without cache")
        return None


async def save_value(query_embedding: List[float], scope: str, value: dict) -> None:
    """Persist an arbitrary JSON-serializable `value` under the given scope."""
    try:
        cache_key = f"{CACHE_PREFIX}{uuid.uuid4()}"
        index_key = f"{CACHE_INDEX_PREFIX}{scope}"
        payload = json.dumps({"embedding": query_embedding, "value": value})
        await redis_client.set(cache_key, payload, ex=CACHE_TTL_SECONDS)
        await redis_client.sadd(index_key, cache_key)
        await redis_client.expire(index_key, CACHE_TTL_SECONDS)
    except Exception:
        logger.warning("Redis cache save failed — result not cached")


async def cached_value(text: str, scope: str, threshold: float, produce) -> dict:
    """Embed `text`, replay a semantically-cached value for `scope` within
    `threshold`, else call `produce` (a zero-arg async fn returning a JSON-safe
    dict), cache it, and return it. Any cache failure degrades to a plain call.

    `text` should capture everything the result depends on. Intent classification
    depends only on the user's message, so callers pass the raw query and use a
    global per-node scope — making hits shared across all users.
    """
    embedding = None
    try:
        # Lazy import avoids a cache→retrieval import cycle at module load.
        from app.services.retrieval import get_embeddings

        embedding = await get_embeddings().aembed_query(text)
        hit = await lookup_value(embedding, scope, threshold)
        if hit is not None:
            logger.info("semantic cache hit scope=%s", scope)
            return hit
    except Exception:
        logger.warning("semantic cache lookup failed scope=%s — computing", scope)

    value = await produce()

    if embedding is not None and value is not None:
        await save_value(embedding, scope, value)
    return value


# --------------------------------------------------------------------------- #
# RAG-specific wrappers (Q&A payload), delegating to the generic store above.
# --------------------------------------------------------------------------- #
async def lookup(query_embedding: List[float], scope: str) -> Optional[dict]:
    """Return the cached {sources, answer} of a semantically-similar query."""
    return await lookup_value(query_embedding, scope, CACHE_SIMILARITY_THRESHOLD)


async def save(query_embedding: List[float], scope: str, sources: list, answer: str) -> None:
    """Persist a Q&A pair to the semantic cache under the given scope."""
    await save_value(query_embedding, scope, {"sources": sources, "answer": answer})
