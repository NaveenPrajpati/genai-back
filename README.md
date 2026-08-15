# Grounded RAG + Multi-Agent Server

A production-shaped FastAPI backend with two subsystems that share one runtime:

1. **A grounded RAG pipeline** — chat over your own documents, with hybrid retrieval, reranking, a semantic cache, inline citations, and a refusal path for questions the documents don't support.
2. **A multi-agent platform** — LangGraph agents (learning tracker, personal assistant, meal planner) behind a supervisor that routes intent, with human-in-the-loop approvals and one skill exposed over the Model Context Protocol.

Both are instrumented end to end: Prometheus metrics, an offline eval harness that gates CI, per-user rate limits, and a daily LLM spend cap.

```
Python 3.13 · FastAPI · LangChain · LangGraph · Pinecone · Redis · Postgres · MongoDB · Supabase · Docker
```

---

## Why this exists

Most RAG demos answer confidently whether or not the answer is in the documents, and have no way to tell you if a prompt change made quality worse. This one is built around the opposite priorities:

- **It refuses.** A grounding gate runs *before* the first token streams, so an unsupported question yields a clean refusal rather than a fluent guess. See [step6_grounding.py](app/services/rag/step6_grounding.py).
- **It cites.** Only sources the answer actually referenced are returned to the client, extracted from the generated text rather than assumed from retrieval.
- **Quality is measured, not asserted.** An offline harness grades correctness, faithfulness, refusal accuracy, and over-refusal against a hand-verified golden set, and blocks PRs on catastrophic regressions.
- **Cost and latency are bounded.** A semantic cache scoped per user *and* per document set, plus a hard daily spend cap that degrades gracefully instead of erroring.

---

## Architecture

```
INGEST   POST /api/rag/ingest/{action}
           └─▶ [background] load ─▶ chunk ─▶ embed (dense + sparse) ─▶ upsert Pinecone
                                                                   └─▶ log status (Supabase)

QUERY    POST /api/rag/query/stream                                        (Server-Sent Events)
           └─▶ embed question
                 └─▶ ◇ semantic cache? ──hit──▶ replay answer + sources ─────────────▶ done
                          │ miss
                          ▼
                    hybrid retrieve (Pinecone, filtered to user_id)
                          ▼
                    rerank (Jina) ─▶ dedupe, number [n], reorder for long context
                          ▼
                    ◇ grounding gate ──not answerable──▶ refusal ────────────────────▶ done
                          │ answerable
                          ▼
                    stream tokens ─▶ extract citations ─▶ cache ─▶ persist ─────────▶ done

AGENTS   POST /api/supervisor/...
           └─▶ supervisor graph ─▶ routes to: learning · assistant (subgraphs)
                                              meal planner (over MCP)
```

Every stage emits a real server-side timing on the SSE stream, so the client can render a live pipeline breakdown and the same numbers land in Prometheus histograms.

---

## Quickstart

```bash
cp .env.example .env      # fill in the keys you need — see Configuration below
docker compose up --build
```

The API is then on `http://localhost:8000`, with interactive docs at `/docs`.

Running locally without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

> Image ingestion uses Tesseract OCR, which is a system binary rather than a Python package: install it with `brew install tesseract` (macOS) or `apt install tesseract-ocr` (Debian/Ubuntu). It is **not** currently installed in the Docker image, so image ingestion fails in a container until it is added to the runtime stage — every other source type works.

### Minimum keys to see RAG work

`OPENAI_API_KEY`, `PINECONE_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`, `REDIS_URL`. Everything else degrades gracefully — the server boots and logs a warning rather than failing. See [.env.example](.env.example) for the annotated full list.

---

## The RAG pipeline

Files are ordered to match the flow. Full detail in **[RAG_SYSTEM.md](RAG_SYSTEM.md)**.

| Stage | What it does | Code |
|---|---|---|
| 1. Ingestion | Loads PDF, DOCX, TXT, URLs, and images (OCR). Runs as a background job returning a `job_id`. | [step1_ingestion.py](app/services/rag/step1_ingestion.py) |
| 2. Chunking | Recursive splitting by default; fixed and semantic strategies selectable. | [step2_chunking.py](app/services/rag/step2_chunking.py) |
| 3. Indexing | Embeds and upserts to Pinecone, tracking per-job status. | [step3_indexing_worker.py](app/services/rag/step3_indexing_worker.py) |
| 4. Retrieval | **Hybrid**: dense OpenAI embeddings for meaning + sparse BM25 for exact terms (IDs, error codes). Then Jina reranking and `LongContextReorder` to counter the "lost in the middle" effect. | [step4_retrieval.py](app/services/rag/step4_retrieval.py) |
| 5. Generation | Builds numbered, deduplicated context and maps citations back to sources. | [step5_generation.py](app/services/rag/step5_generation.py) |
| 6. Grounding | Pre-stream answerability gate; buffers the leading tokens so the internal refusal sentinel never leaks to the UI. | [step6_grounding.py](app/services/rag/step6_grounding.py) |
| 7. Evaluation | Optional online LLM-judge scoring — retrieval precision, recall, hallucination rate — sampled server-side to bound cost. | [step7_evaluation.py](app/services/rag/step7_evaluation.py) |

**Semantic cache** ([cache.py](app/services/cache.py)) sits in front of retrieval. It keys on the *query embedding* rather than the string, so "What is RAG?" and "what's RAG" hit the same entry. Entries are scoped by `(user_id, document set)` — a cached answer computed over one document selection can never be replayed for another, or for another user — and the whole user namespace is invalidated on ingest or delete so stale answers can't outlive their sources.

### API

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/rag/ingest/{action}` | Queue a file or URL for ingestion → `job_id` |
| `GET` | `/api/rag/get-ingestions` | List the caller's ingestions and their status |
| `DELETE` | `/api/rag/ingest/{doc_id}` | Drop vectors + log row, invalidate cached answers |
| `POST` | `/api/rag/query/stream` | Ask a question; SSE stream of stages, sources, tokens, citations |

---

## The agent system

A supervisor graph routes a message to the right skill, over three deliberately different transports — in-process subgraphs for the tightly-coupled skills, MCP for the one that should be independently consumable. Approvals raised deep inside a subgraph pause the parent thread and resume from a single `Command`, which is what makes human-in-the-loop work across the protocol boundary.

Full detail in **[AGENT_SYSTEM.md](AGENT_SYSTEM.md)**; the most developed agent has its own reference in **[LEARNING_SYSTEM.md](LEARNING_SYSTEM.md)** (spaced repetition, mastery tracking, misconception handling).

The meal planner is mounted at `/mcp` and works with any MCP client — point Claude Desktop or the MCP Inspector at it.

---

## Evaluation

Two harnesses, deliberately separate:

```bash
# Offline, deterministic, no API keys or network — guards the scoring logic itself
python -m app.services.rag.eval_harness --self-test

# Real eval against the golden set
python -m app.services.rag.eval_harness --data app/services/rag/datasets/rag_golden.jsonl

# Same, but exit non-zero below threshold
python -m app.services.rag.eval_harness --data app/services/rag/datasets/rag_golden.jsonl --gate
```

Metrics: **correctness** (deterministic key-fact coverage, so it's stable in CI), **faithfulness** (LLM judge, swappable for RAGAS via `--scorer ragas`), **refusal accuracy**, and **over-refusal** — the last one guarding the grounding gate against becoming so strict it declines answerable questions.

The eval runs on frozen context: each golden row stores its retrieved chunks, so generation regressions are isolated from retrieval ones and the run needs no Pinecone.

[`.github/workflows/eval.yml`](.github/workflows/eval.yml) runs this on every PR as a **smoke gate** — thresholds sit well below run-to-run judge noise, so it blocks only on a genuine break (inverted gate, broken prompt, ungrounded generator) rather than false-failing normal work.

---

## Observability

- `GET /metrics` — Prometheus scrape endpoint, optionally bearer-token protected via `METRICS_TOKEN`.
- Tracked: per-stage latency histograms, error rate, cache hit rate, refusal rate, and cost per query. Cost is captured via a LangChain callback on the shared model clients, so it propagates to every call site without touching agent code.
- A Grafana dashboard and Prometheus/Alertmanager config live in [monitoring/](monitoring/).
- LangSmith tracing is opt-in per request — a no-op unless `LANGSMITH_TRACING` is set.

Label cardinality is kept deliberately low (never by user, question, or chat id).

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

419 tests across 15 files, covering tenant isolation, rate limiting, cache invalidation, SSE stage emission, content safety, resilience, the eval harness, and each agent's workflow.

---

## Configuration

All dials live in [core/config.py](app/core/config.py). Most read from the environment; the two cache constants are edited in code. The ones worth knowing:

| Variable | Effect |
|---|---|
| `CACHE_SIMILARITY_THRESHOLD` *(constant)* | How close a question must be to replay a cached answer. Default `0.95` — lower means more hits and more risk of a wrong one. |
| `CACHE_TTL_SECONDS` *(constant)* | Cached answer lifetime, default 24h. |
| `RAG_*_RATE_LIMIT` / `_WINDOW` | Per-user caps on query, ingest, and delete. |
| `LLM_DAILY_BUDGET_USD` | Hard daily spend cap. On breach, queries return a graceful message instead of erroring. |
| `RAG_EVAL_SAMPLE_RATE` | Fraction of eval-requested queries actually scored, bounding per-query cost. |
| `METRICS_TOKEN` | Requires a bearer token on `/metrics`. |

---

## Deployment

Multi-stage Docker build: C extensions compile in a builder stage, the runtime image ships only Python, a prebuilt venv, and app code, running as a non-root user. NLTK corpora for the BM25 tokenizer are baked in at build time so the first query never pays a download.

The container runs a **single** gunicorn worker on purpose — the APScheduler cron loop and in-memory job state assume one process. Scaling out requires an external queue first.

[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) deploys to EC2 over SSH on push to `main`.

---

## Documentation

| Doc | Covers |
|---|---|
| [RAG_SYSTEM.md](RAG_SYSTEM.md) | Complete RAG reference — components, data model, API, security, config, evaluation |
| [AGENT_SYSTEM.md](AGENT_SYSTEM.md) | Supervisor, subgraphs, MCP transport, human-in-the-loop |
| [LEARNING_SYSTEM.md](LEARNING_SYSTEM.md) | Learning agent — roadmaps, digests, checkpoints, spaced repetition, mastery |

Each `step*.py` file also carries an extended module docstring explaining not just what it does but which alternatives exist and when you'd switch.

---

## Known limitations

- The golden eval set is small (13 rows), so single-row noise moves a metric noticeably. Thresholds are calibrated to absorb that; growing the set is the path to a strict gate.
- The semantic cache brute-forces cosine similarity across a scope — O(n) per lookup, fine for hundreds of entries, needs an ANN index beyond that.
- Single-worker deployment, as above.
- BM25 uses pretrained MS MARCO statistics rather than being fit on the indexed corpus.
