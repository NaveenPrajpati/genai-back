"""
core/config.py
==============
Single source of truth for environment variables, tunable constants, and the
shared external clients (Pinecone, Supabase, Redis).

WHY THIS FILE EXISTS
--------------------
In the original code, Pinecone/Supabase/Redis setup, the cache constants, and
the model names were all scattered through one 600-line file. That makes it
hard to (a) see what knobs exist and (b) swap a provider later.

Centralizing config means: change a model name, a top_k, or a cache threshold
in ONE place and the whole pipeline picks it up.
"""

import os
import logging

import redis.asyncio as aioredis
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from supabase import create_client, Client

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tunable constants  (the "dials" of the whole RAG system)
# ─────────────────────────────────────────────────────────────────────────────

# --- Vector index ---
INDEX_NAME = "rag-hybrid"
EMBEDDING_DIM = 1536  # text-embedding-3-small / ada-002 dimension
PINECONE_METRIC = "dotproduct"  # REQUIRED for native sparse+dense hybrid search

# --- Retrieval ---
RETRIEVER_TOP_K = 10  # how many candidates the retriever pulls
RERANK_TOP_N = 5  # how many survive the rerank

# --- Grounding / citation enforcement ---
# When on, an answerability gate (a cheap fast-model check) runs before every RAG
# answer and refuses when the retrieved context can't support one — see
# services/grounding.py. Kill switch for debugging / latency-sensitive callers.
RAG_GROUNDING_GATE = os.getenv("RAG_GROUNDING_GATE", "1") == "1"

# --- Models ---
# Two-tier routing (see core/llm.py): the top model handles generation-heavy
# work (roadmap, plan, research, synthesis); the fast model handles the ~40-50%
# of calls that are trivial classification / extraction / selection, where a mini
# model is ~10-15x cheaper at no real quality loss. Override either via env.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "gpt-4o-mini")
# Hosted Jina reranker (free tier). Requires JINA_API_KEY in the environment.
RERANKER_MODEL = "jina-reranker-v2-base-multilingual"

GROQ_LLM_MODEL = "qwen/qwen3-32b"
GROQ_FAST_LLM_MODEL = "openai/gpt-oss-20b"

# --- Semantic cache ---
CACHE_PREFIX = "rag:cache:"
CACHE_INDEX_PREFIX = "rag:cache_idx:"  # one Redis Set per retrieval scope
CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours
# Cosine-sim required for a cache hit, tuned per task type:
#  • Generation (RAG answers): strict — a near-miss query can need a different
#    answer, and a wrong replay is user-visible. Keep high.
#  • Classification (intent routing): loose — intent is robust to phrasing
#    ("show my tasks" / "what do I have" → same label), so a looser match still
#    lands the right intent, and a hit here is essentially free.
CACHE_SIMILARITY_THRESHOLD = 0.95  # generation / RAG
CACHE_CLASSIFY_THRESHOLD = 0.90  # intent classification


def _require(name: str) -> str:
    """Fetch an env var or fail loudly at startup (better than failing mid-request)."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable not set")
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Pinecone  (vector database)
# ─────────────────────────────────────────────────────────────────────────────

_pinecone_index = None


def get_pinecone_index():
    """Return the Pinecone index, initializing it on first call."""
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=_require("PINECONE_KEY"))
        if not pc.has_index(INDEX_NAME):
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _pinecone_index = pc.Index(INDEX_NAME)
    return _pinecone_index


# ─────────────────────────────────────────────────────────────────────────────
# Supabase  (relational store: ingestion logs, chats, messages)
# ─────────────────────────────────────────────────────────────────────────────

supabase: Client = create_client(_require("SUPABASE_URL"), _require("SUPABASE_KEY"))


# ─────────────────────────────────────────────────────────────────────────────
# Redis  (semantic cache)
# ─────────────────────────────────────────────────────────────────────────────

redis_client: aioredis.Redis = aioredis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True,
)
