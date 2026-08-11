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
RETRIEVER_TOP_K = 25  # how many candidates the retriever pulls
RERANK_TOP_N = 5  # how many survive the rerank

# --- Grounding / citation enforcement ---
# When on, an answerability gate (a cheap fast-model check) runs before every RAG
# answer and refuses when the retrieved context can't support one — see
# services/grounding.py. Kill switch for debugging / latency-sensitive callers.
RAG_GROUNDING_GATE = os.getenv("RAG_GROUNDING_GATE", "1") == "1"

# --- Evaluation sampling ---
# LLM-as-judge eval (step7) fires N+2 model calls per query, so a caller that
# always sets evaluate=True multiplies per-query cost. This decouples ACTUAL eval
# execution from the raw client flag: even when eval is requested, run it only for
# this fraction of queries. 1.0 = run every requested eval (default, no behaviour
# change); 0.1 = sample 10% (recommended in production); 0.0 = disable. Clamped to
# [0, 1]; a bad value falls back to 1.0.
try:
    RAG_EVAL_SAMPLE_RATE = min(1.0, max(0.0, float(os.getenv("RAG_EVAL_SAMPLE_RATE", "1.0"))))
except ValueError:
    RAG_EVAL_SAMPLE_RATE = 1.0

# --- Daily spend cap ---
# Hard global cap on estimated LLM spend per UTC day. Once today's accumulated
# cost (see core/metrics.py) reaches this, new LLM-backed queries are refused
# until UTC midnight rollover; free paths (semantic-cache hits) still serve.
# 0 = disabled (no cap, default). Accounting is in-process, which is global only
# while the app runs a single worker (Dockerfile: gunicorn --workers 1); scaling
# to N workers makes the effective cap N x this value. A bad value disables it.
try:
    LLM_DAILY_BUDGET_USD = max(0.0, float(os.getenv("LLM_DAILY_BUDGET_USD", "0")))
except ValueError:
    LLM_DAILY_BUDGET_USD = 0.0

# --- RAG per-user rate limits ---
# Fixed-window request caps per authenticated user on the cost-bearing RAG
# endpoints (LLM / embeddings / Pinecone / Jina). Complements the global daily
# spend cap: the cap bounds total $/day, these bound one user's request rate.
# Enforced in routers/rag.py via services/rate_limit.py, which fails OPEN on a
# Redis outage. /query and /query/stream share one bucket ("rag_query"). Each
# limit and its window (seconds) is env-overridable.
def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default

RAG_QUERY_RATE_LIMIT = _int_env("RAG_QUERY_RATE_LIMIT", 30)
RAG_QUERY_RATE_WINDOW = _int_env("RAG_QUERY_RATE_WINDOW", 60)
RAG_INGEST_RATE_LIMIT = _int_env("RAG_INGEST_RATE_LIMIT", 10)
RAG_INGEST_RATE_WINDOW = _int_env("RAG_INGEST_RATE_WINDOW", 3600)
RAG_DELETE_RATE_LIMIT = _int_env("RAG_DELETE_RATE_LIMIT", 30)
RAG_DELETE_RATE_WINDOW = _int_env("RAG_DELETE_RATE_WINDOW", 3600)

# --- Learning per-user rate limits ---
# The same idea on the learning routes that spend money: a chat turn, a digest
# (a web search plus two or three model calls — the most expensive thing here),
# issuing a checkpoint, and judging a written explanation.
#
# The domain already limits some of this, but only within one topic:
# CHECKPOINT_MAX_ATTEMPTS_PER_DAY is per topic, and DIGEST_MAX_UNREAD is per
# topic. A learner with twenty topics is bounded by neither, and the Feynman
# judge has no domain limit at all. These are the account-level backstop, and
# they are what stops a guest account — 20 of which can be minted per IP per
# hour — from being an open tap on the model budget.
#
# /query and /query/stream share one bucket ("learning_query"), as RAG's do.
# Reads are not limited: they cost a Mongo query. Fails OPEN on a Redis outage.
LEARNING_QUERY_RATE_LIMIT = _int_env("LEARNING_QUERY_RATE_LIMIT", 30)
LEARNING_QUERY_RATE_WINDOW = _int_env("LEARNING_QUERY_RATE_WINDOW", 60)
LEARNING_DIGEST_RATE_LIMIT = _int_env("LEARNING_DIGEST_RATE_LIMIT", 20)
LEARNING_DIGEST_RATE_WINDOW = _int_env("LEARNING_DIGEST_RATE_WINDOW", 3600)
LEARNING_CHECKPOINT_RATE_LIMIT = _int_env("LEARNING_CHECKPOINT_RATE_LIMIT", 30)
LEARNING_CHECKPOINT_RATE_WINDOW = _int_env("LEARNING_CHECKPOINT_RATE_WINDOW", 3600)
LEARNING_EXPLAIN_RATE_LIMIT = _int_env("LEARNING_EXPLAIN_RATE_LIMIT", 20)
LEARNING_EXPLAIN_RATE_WINDOW = _int_env("LEARNING_EXPLAIN_RATE_WINDOW", 3600)

# --- Learning checkpoints (active recall) ---
# Percentage a learner must score on a topic's checkpoint before it counts as
# completed. Also the bar a spaced-repetition review has to clear to push the
# next review further out.
CHECKPOINT_PASS_SCORE = _int_env("CHECKPOINT_PASS_SCORE", 80)
CHECKPOINT_QUESTIONS = _int_env("CHECKPOINT_QUESTIONS", 4)

# --- Checkpoint retries ---
# A completion gate is only a gate if failing costs something. Two limits, doing
# different jobs: the cooldown stops reflexive re-submission in the seconds after
# a fail, and the daily cap stops a learner grinding the same topic until a
# regenerated question set happens to land somewhere they know.
CHECKPOINT_RETRY_COOLDOWN_MINUTES = _int_env("CHECKPOINT_RETRY_COOLDOWN_MINUTES", 15)
CHECKPOINT_MAX_ATTEMPTS_PER_DAY = _int_env("CHECKPOINT_MAX_ATTEMPTS_PER_DAY", 3)

# --- Misconception analysis ---
# Wrong answers pooled across digest checks, checkpoints and reviews, then read
# by an LLM for the belief underneath them. Below this much evidence there's
# nothing to generalise from and the model invents a pattern from one bad day.
MISCONCEPTION_MIN_EVIDENCE = _int_env("MISCONCEPTION_MIN_EVIDENCE", 3)
# Analysis window and output size. A learner who has moved on shouldn't keep
# being probed on a belief they corrected ten attempts ago.
MISCONCEPTION_EVIDENCE_WINDOW = _int_env("MISCONCEPTION_EVIDENCE_WINDOW", 40)
MISCONCEPTION_MAX_PATTERNS = _int_env("MISCONCEPTION_MAX_PATTERNS", 4)
# How long a misconception goes quiet after something has addressed it. Two
# clocks, one per surface: a digest that teaches against a confusion and a
# checkpoint that then probes it are a good pairing, so teaching must not
# suppress the test.
#
# Without a decay, every checkpoint chases the same belief forever and every
# digest re-teaches the same correction. With one, a freshly seen confusion is
# addressed at once, goes quiet, and comes back later — which is the only way to
# tell a correction that stuck from one that was papered over for a session.
MISCONCEPTION_REPROBE_DAYS = _int_env("MISCONCEPTION_REPROBE_DAYS", 7)
MISCONCEPTION_RETEACH_DAYS = _int_env("MISCONCEPTION_RETEACH_DAYS", 5)

# --- Learning roadmaps ---
# How many roadmaps a learner may have `active` at once. Active is what the
# daily sweep digests and what a bare "what should I study next?" resolves to, so
# this is really a cap on how many things are being drip-fed in parallel — past
# two, every stream gets less attention than it needs and the digest queue turns
# into noise. Parked and archived roadmaps don't count; they cost nothing.
MAX_ACTIVE_ROADMAPS = _int_env("MAX_ACTIVE_ROADMAPS", 2)

# --- Learning digests ---
# How many unacknowledged digests one topic may accumulate before the generator
# stops. Digests are nudges; a stack of unread ones is just noise, and each costs
# a web search and an LLM call.
DIGEST_MAX_UNREAD = _int_env("DIGEST_MAX_UNREAD", 3)
# From the second digest on a topic, acknowledging it requires recalling what the
# earlier ones said. Short and all-or-nothing on purpose: this is a "did you read
# it" check, not an exam — the topic checkpoint is the exam.
DIGEST_QUIZ_QUESTIONS = _int_env("DIGEST_QUIZ_QUESTIONS", 2)
# How often that check rides along: every 2nd digest (#2, #4, #6 …). A check on
# every digest past the first turns a nudge into homework — this is a "did you
# read it" tap on the shoulder, and the topic checkpoint is the real assessment.
DIGEST_QUIZ_EVERY = _int_env("DIGEST_QUIZ_EVERY", 2)
# The floor under the drip-feed: how many digests a topic gets before coverage is
# allowed to end it and hand over to the checkpoint.
#
# Without one, a narrow topic — two learning outcomes, five good bullets — is
# declared covered by the very first digest, and the topic goes straight to the
# checkpoint. Correct, and it quietly disables everything downstream of it: the
# recall check rides the SECOND digest, so a topic that never has one is a topic
# whose acknowledgement means "dismissed" rather than "read", and neither the
# written question nor the re-teach path can ever fire.
#
# Defaults to DIGEST_QUIZ_EVERY, which is what makes the guarantee exact: one
# check per topic, minimum. A floor below the cadence guarantees nothing, so it
# tracks that setting rather than being a second number to keep in step.
DIGEST_MIN_BEFORE_CHECKPOINT = _int_env("DIGEST_MIN_BEFORE_CHECKPOINT", DIGEST_QUIZ_EVERY)
# From this digest on, the recall check also asks for one typed sentence. Early
# checks stay tap-only so the habit forms with no friction; by the fourth digest
# there is enough material that "say it in your own words" is worth the keystroke,
# and a sentence shows how someone is thinking in a way four options cannot.
DIGEST_ONELINER_FROM = _int_env("DIGEST_ONELINER_FROM", 4)
# Failed attempts at one digest's recall check before the agent concludes the
# teaching is at fault rather than the reader, and re-explains the material a
# different way. Two, not one: everybody misreads a question occasionally, and
# escalating on a single slip would spend an LLM call on a typo.
DIGEST_RETEACH_AFTER = _int_env("DIGEST_RETEACH_AFTER", 2)

# --- Feynman checkpoint (explain it in your own words) ---
# An optional free-text explanation offered alongside the closed-form checkpoint.
# It never gates completion — the point is a low-friction moment at the highest-
# stakes step, not a second exam. It is paid for out of the review ladder: a
# strong explanation starts the topic further along it, so the learner who can
# explain the thing sees it again later than the learner who only recognised the
# right option.
FEYNMAN_PASS_SCORE = _int_env("FEYNMAN_PASS_SCORE", 70)
FEYNMAN_LADDER_BONUS = _int_env("FEYNMAN_LADDER_BONUS", 1)
# Below this, there is nothing to judge — a two-word answer is a shrug, and
# grading it produces a confident verdict about nothing.
FEYNMAN_MIN_WORDS = _int_env("FEYNMAN_MIN_WORDS", 20)
DIGEST_QUIZ_PASS_SCORE = _int_env("DIGEST_QUIZ_PASS_SCORE", 100)

# --- MCP (meal planner published over Model Context Protocol) ---
# app.main mounts the server at /mcp of this same process, so the default URL
# points back at ourselves. Both sides are async, so the supervisor awaiting its
# own MCP endpoint just yields to the event loop — no deadlock on one worker.
# Point MEAL_MCP_URL elsewhere to run the meal planner as its own service.
# The trailing slash matters: the server is mounted at /mcp and serves at its
# own root, so /mcp/ is the endpoint and /mcp would only redirect to it.
MEAL_MCP_URL = os.getenv("MEAL_MCP_URL", "http://127.0.0.1:8000/mcp/")
MEAL_MCP_TIMEOUT = max(1.0, float(os.getenv("MEAL_MCP_TIMEOUT", "60")))
# Shared secret for the mounted /mcp endpoint. Unset = no check, which is only
# acceptable when /mcp is unreachable from outside the host (EC2 security group).
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "")

# --- Models ---
# Two-tier routing (see core/llm.py): the top model handles generation-heavy
# work (roadmap, plan, research, synthesis); the fast model handles the ~40-50%
# of calls that are trivial classification / extraction / selection, where a mini
# model is ~10-15x cheaper at no real quality loss. Override either via env.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
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
# External-call timeouts  (resilience: a hung upstream must fail fast, not hang
# the whole request). Each is overridable via env; balanced defaults.
# ─────────────────────────────────────────────────────────────────────────────
# Redis socket read/connect timeout (s): converts a stalled cache connection into
# a TimeoutError that the cache's try/except degrades to a miss — a true fail-open
# rather than a hang (try/except alone catches errors, not hangs).
try:
    REDIS_SOCKET_TIMEOUT = max(0.1, float(os.getenv("REDIS_SOCKET_TIMEOUT", "2")))
except ValueError:
    REDIS_SOCKET_TIMEOUT = 2.0

# Jina reranker HTTP timeout (s): rerank is a quality pass, not correctness — on
# timeout we fall back to the unreranked candidates (see services/rag/step4).
try:
    JINA_RERANK_TIMEOUT = max(0.1, float(os.getenv("JINA_RERANK_TIMEOUT", "8")))
except ValueError:
    JINA_RERANK_TIMEOUT = 8.0

# Jina reranker HTTP connection-pool size. retrieve_and_rerank runs in the default
# asyncio thread pool (~32 workers), so size the pool to match — otherwise
# concurrent reranks churn the requests default of 10 connections.
try:
    JINA_POOL_MAXSIZE = max(1, int(os.getenv("JINA_POOL_MAXSIZE", "32")))
except ValueError:
    JINA_POOL_MAXSIZE = 32

# Max seconds to spend warming the retrieval stack at startup (BM25 download +
# Pinecone handle) before giving up and letting the first query lazy-load —
# keeps a slow/hung dependency from stalling the boot indefinitely.
try:
    RAG_WARMUP_TIMEOUT = max(1.0, float(os.getenv("RAG_WARMUP_TIMEOUT", "60")))
except ValueError:
    RAG_WARMUP_TIMEOUT = 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Redis  (semantic cache)
# ─────────────────────────────────────────────────────────────────────────────

redis_client: aioredis.Redis = aioredis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True,
    socket_timeout=REDIS_SOCKET_TIMEOUT,
    socket_connect_timeout=REDIS_SOCKET_TIMEOUT,
    health_check_interval=30,
)
