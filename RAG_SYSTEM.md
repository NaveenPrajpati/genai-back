# RAG System — Complete Reference (A → Z)

The full retrieval-augmented-generation subsystem: what it is, every component, how
data flows, the API, security model, configuration, and how it's evaluated.

- **Companion docs:** [RAG_EVALUATION_PLAN.md](RAG_EVALUATION_PLAN.md) (offline eval + CI).
- **Code root:** [`app/routers/rag.py`](app/routers/rag.py) (HTTP) + [`app/services/rag/`](app/services/rag/) (pipeline, files ordered `step1_…`→`step7_…`).

---

## 1. What it is

A multi-tenant "chat over your documents" service. Users upload PDFs, text files, or
URLs; the system chunks and embeds them into a hybrid vector index, and answers
questions **strictly from the retrieved content** — with inline citations, a refusal
when the documents don't support an answer, streaming responses, and a semantic cache.

Two flows:

- **Ingestion (write path)** — happens once per document, in the background.
- **Query (read path)** — happens per question; retrieve → rerank → ground-check → generate.

> A visual of both flows exists as an artifact (RAG flow diagram) and the grounding
> before/after comparison. This document is the text-of-record.

---

## 2. Architecture at a glance

```
INGESTION   POST /rag/ingest ──▶ [background] load ─▶ chunk ─▶ embed(dense+sparse) ─▶ upsert Pinecone
                                                                                  └─▶ log status (Supabase)

QUERY       POST /rag/query/stream
              └▶ embed question ─▶ ◇ semantic cache? ──hit──▶ replay answer+sources ─▶ done
                                        │miss
                                        ▼
                 hybrid retrieve (Pinecone, top_k=10, filter user_id)
                                        ▼
                 rerank (Jina, top_n=5) ─▶ prepare context (dedup, number [n], reorder)
                                        ▼
                 ◇ answerable gate? ──no──▶ refuse ─▶ done(grounded=false)
                                        │yes
                                        ▼
                 stream answer (cite [n], sentinel-guarded) ─▶ parse citations
                                        ▼
                 cache + persist ─▶ (optional eval) ─▶ done(grounded)
```

---

## 3. Tech stack / external services

| Service       | Role                                                                                          | Where configured                                            |
| ------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Pinecone**  | Hybrid (dense + sparse) vector index, serverless                                              | [`core/config.py`](app/core/config.py) `get_pinecone_index` |
| **OpenAI**    | Embeddings (`OpenAIEmbeddings`, 1536-dim) + `gpt-4o` (answers) + `gpt-4o-mini` (gate, judges) | [`core/llm.py`](app/core/llm.py)                            |
| **Jina**      | Hosted cross-encoder reranker (`jina-reranker-v2-base-multilingual`)                          | [`services/rag/step4_retrieval.py`](app/services/rag/step4_retrieval.py)        |
| **BM25**      | Sparse keyword encoder (`pinecone_text`, MS-MARCO default)                                    | [`services/rag/step4_retrieval.py`](app/services/rag/step4_retrieval.py)        |
| **Redis**     | Semantic cache (embedding-keyed, 24 h TTL)                                                    | [`services/cache.py`](app/services/cache.py)                |
| **Supabase**  | Relational store: ingestion logs, chats, messages                                             | [`services/rag/storage.py`](app/services/rag/storage.py)            |
| **LangSmith** | Optional tracing (opt-in via env)                                                             | [`core/observability.py`](app/core/observability.py)        |

---

## 4. Data model

### Pinecone vectors (one per chunk)

Written by [`retrieval.hybrid_add_texts`](app/services/rag/step4_retrieval.py). Each vector carries
dense **and** sparse values plus metadata:

| Metadata key                  | Meaning                                                                      |
| ----------------------------- | ---------------------------------------------------------------------------- |
| `user_id`                     | **Tenant key** — every query filters on this                                 |
| `doc_id`                      | The ingestion job id (groups a document's chunks; used for scoping & delete) |
| `source`                      | Original filename or URL                                                     |
| `file_type`                   | `pdf` / `text` / `url`                                                       |
| `ingested_at`                 | ISO timestamp                                                                |
| `chunk_index`, `total_chunks` | Position within the document                                                 |
| `relevance_score`             | Written by the reranker at query time; surfaced as citation confidence       |

### Supabase tables

- **`rag_ingestion_logs`** — `doc_id, source, file_type, status, ingested_at, user_id, chunks, completed_at, error`. Source of truth for ingestion status & ownership.
- **`rag_chats`** — `id, title, created_at, updated_at` (+ user).
- **`rag_messages`** — persisted Q&A turns (`save_messages`): question, answer, sources, ingestions, chat_id, user_id.

### Redis cache

- Value key: `rag:cache:<uuid>` → JSON `{embedding, value:{sources, answer}}`, TTL 24 h.
- Index set: `rag:cache_idx:<scope>` → member keys, where **scope = `user_id::doc_ids`**.

---

## 5. The pipeline, component by component

### Step 1 — Ingestion / loading · [`services/rag/step1_ingestion.py`](app/services/rag/step1_ingestion.py)

Turns a raw source into LangChain `Document`s. Supported inputs:

- **PDF** → `PyPDFLoader` (one Document per page, sets `metadata["page"]`).
- **.txt** → `TextLoader`.
- **URL** → `load_url` (requests + BeautifulSoup; strips `script/style/nav/footer/header/aside` so only article text is embedded).

`SUPPORTED_FILE_TYPES` maps MIME → loader; the router also falls back to the filename
extension because clients often send PDFs as `application/octet-stream`.

### Step 2 — Chunking · [`services/rag/step2_chunking.py`](app/services/rag/step2_chunking.py)

`get_splitter(strategy)` factory. Default **RECURSIVE** (`RecursiveCharacterTextSplitter`,
**chunk_size 1000 / overlap 200**). Also supports FIXED and SEMANTIC. Ingestion uses the
recursive default.

### Step 3 — Embedding (hybrid) · [`services/rag/step4_retrieval.py`](app/services/rag/step4_retrieval.py)

Every chunk gets two vectors:

- **Dense** — `OpenAIEmbeddings` (meaning: "car" ≈ "automobile").
- **Sparse** — `BM25Encoder` (exact keywords: IDs, error codes, names).

Both are cached (`@cache`) and built lazily. Dense + sparse = **hybrid search**, the
strong modern default. (Known improvement: BM25 uses `.default()`/MS-MARCO stats rather
than being fit on your corpus — see §12.)

### Step 4 — Retrieval · [`services/rag/step4_retrieval.py`](app/services/rag/step4_retrieval.py)

`build_retriever(user_id, doc_ids)` → `PineconeHybridSearchRetriever` subclass that
**always** applies a metadata filter:

```python
{"user_id": user_id}                              # empty ingestions = all of MY docs
{"user_id": user_id, "doc_id": {"$in": doc_ids}}  # scoped to chosen documents
```

Pulls **top_k = 10** candidates. The `user_id` filter is the multi-tenant isolation
boundary (§7).

### Step 5 — Reranking & ordering · [`services/rag/step4_retrieval.py`](app/services/rag/step4_retrieval.py)

- **Jina reranker** (`ContextualCompressionRetriever`) re-scores the 10 candidates with a
  cross-encoder and keeps the best **top_n = 5**, writing `relevance_score` into metadata.
- **LongContextReorder** (applied in `prepare_context`) puts the most relevant chunks at
  the START and END to fight the "lost in the middle" effect.

### Step 6 — Context prep + citations · [`services/rag/step5_generation.py`](app/services/rag/step5_generation.py)

`prepare_context(docs)`:

1. **Dedup** near-duplicate hits (first-120-char key).
2. **Number** each unique chunk `[1] [2] …` by relevance order.
3. **Reorder** for lost-in-the-middle (labels stay pinned, so citations survive reorder).
4. Returns `(context_string, sources)` where each source carries its `citation` number,
   `chunk_text`, `source`, `page_number`, `confidence_score`, `doc_id`.

`cited_sources(sources, cited)` filters to the sources the answer actually cited.

### Step 7 — Grounding enforcement · [`services/rag/step6_grounding.py`](app/services/rag/step6_grounding.py)

Three layers so the system **declines rather than confabulates**:

1. **Answerability gate** (`is_answerable`) — a cheap `gpt-4o-mini` structured check _before_
   generation: do the chunks contain the facts asked for? If not → refuse. Fails **open** on
   error (the sentinel still guards). Kill switch: `RAG_GROUNDING_GATE`.
2. **Inline citations** — numbered context + a prompt rule to tag every claim `[n]`.
3. **Refusal sentinel** (`INSUFFICIENT_CONTEXT`) — a backstop: even past the gate, the model
   can declare the context insufficient; the route detects the token and returns
   `REFUSAL_MESSAGE`. Streaming buffers leading tokens so the sentinel never leaks on screen.

### Step 8 — Generation · [`core/prompts.py`](app/core/prompts.py) `RAG_ANSWER`

`gpt-4o`, `temperature=0`. The prompt: answer only from numbered context, cite `[n]` per
claim, emit `INSUFFICIENT_CONTEXT` if unsupported. Streamed token-by-token over SSE.

### Semantic cache · [`services/cache.py`](app/services/cache.py)

Sits **in front of** retrieval. Keys on the query **embedding** (not exact text), scoped by
`user_id::doc_ids`, with cosine ≥ **0.95**. On a hit, replays the stored `{answer, sources}`
— no retrieval, no LLM. Degrades gracefully (a cache error never breaks a request). Refusals
are never cached.

### Persistence · [`services/rag/storage.py`](app/services/rag/storage.py)

`save_messages` writes each Q&A turn (question, answer, sources, ingestions) to
`rag_messages`, auto-creating a chat when `chat_id` is omitted. Fire-and-forget so it never
blocks the stream.

### Evaluation (online, optional) · [`services/rag/step7_evaluation.py`](app/services/rag/step7_evaluation.py)

When a query sets `evaluate=true`, three LLM-judge metrics run concurrently:

- `retrieval_precision` — fraction of retrieved chunks that were relevant.
- `recall_score` — did the context cover the answer?
- `hallucination_rate` — fraction of the answer NOT supported by context.

(Offline batch evaluation + CI is a separate system — §10.)

---

## 6. API reference · [`app/routers/rag.py`](app/routers/rag.py)

All routes require a Bearer token (`get_current_user`); `user_id = current_user["uid"]`.

| Method   | Path                          | Purpose                                                             |
| -------- | ----------------------------- | ------------------------------------------------------------------- |
| `GET`    | `/rag/get-files`              | List the caller's ingestions                                        |
| `POST`   | `/rag/ingest/{action}`        | Queue a URL (`action=url`) or file upload; returns `202` + `job_id` |
| `GET`    | `/rag/ingest/status/{job_id}` | Poll ingestion status (ownership-checked)                           |
| `DELETE` | `/rag/ingest/{doc_id}`        | Delete an ingestion's vectors + log (ownership-checked)             |
| `POST`   | `/rag/query`                  | Non-streaming answer                                                |
| `POST`   | `/rag/query/stream`           | Streaming answer (SSE) + semantic cache                             |

### Request body (`/query`, `/query/stream`)

```jsonc
{
  "question": "What was Q3 revenue?",
  "evaluate": false, // run the online judges
  "ingestions": [], // doc_ids to scope to; [] = all of the user's docs
  "chat_id": null, // omit to auto-create a chat
}
```

### `/query` response

```jsonc
{
  "answer": "Q3 revenue was $4.2M [2].",
  "sources": [
    {
      "citation": 2,
      "chunk_text": "...",
      "source": "financials_q3.pdf",
      "page_number": 3,
      "confidence_score": 0.94,
      "doc_id": "...",
    },
  ],
  "grounded": true,
  "evaluation": {
    "retrieval_precision": 0.8,
    "recall_score": 1.0,
    "hallucination_rate": 0.0,
  },
}
```

On refusal: `{ "answer": REFUSAL_MESSAGE, "sources": [], "grounded": false }`.

### `/query/stream` — Server-Sent Events

Each frame is `data: {json}\n\n`. Event `type`s:

| `type`       | Payload                        | When                                |
| ------------ | ------------------------------ | ----------------------------------- |
| `sources`    | `{sources, cached?}`           | Once, up front                      |
| `token`      | `{token}`                      | Per streamed chunk                  |
| `citations`  | `{cited:[n,…]}`                | After generation (grounded answers) |
| `evaluation` | `{evaluation}`                 | If `evaluate=true`                  |
| `done`       | `{grounded, cached?, chat_id}` | Terminal                            |
| `error`      | `{message}`                    | On failure / no documents           |

---

## 7. Multi-tenancy & security

- **Every** write stamps `user_id`; **every** read/cache/delete is scoped to the caller.
  Empty `ingestions` means "all of _my_ documents", never everyone's.
- **Cache isolation**: scope is prefixed with `user_id`, so one user's cached answer can
  never be replayed to another.
- **Ownership checks**: `DELETE` and status routes 404 (not 403) when the `doc_id` isn't the
  caller's — so they don't reveal another tenant's ids.
- **Auth**: JWT Bearer, with token-version invalidation on password reset ([`dependencies.py`](app/dependencies.py)).

> ⚠️ Vectors ingested before the multi-tenancy change lack `user_id` and are filtered out —
> re-ingest to restore them.

---

## 8. Prompt registry · [`core/prompts.py`](app/core/prompts.py)

Prompts are versioned architecture, not scattered strings. Each is a `Prompt(name, version,
template)`; bump the version on any wording change (never edit silently). `REGISTRY` maps
name → Prompt for logging/lookup.

| Name                                                    | Used by                                   |
| ------------------------------------------------------- | ----------------------------------------- |
| `RAG_ANSWER`                                            | Generation (citations + refusal sentinel) |
| `ANSWERABILITY`                                         | Grounding gate                            |
| `EVAL_RELEVANCE` / `EVAL_RECALL` / `EVAL_HALLUCINATION` | Online judges                             |

---

## 9. Configuration dials · [`core/config.py`](app/core/config.py)

| Constant                     | Default                              | Meaning                        |
| ---------------------------- | ------------------------------------ | ------------------------------ |
| `INDEX_NAME`                 | `rag-hybrid`                         | Pinecone index                 |
| `EMBEDDING_DIM`              | `1536`                               | Must match the embedding model |
| `PINECONE_METRIC`            | `dotproduct`                         | Required for native hybrid     |
| `RETRIEVER_TOP_K`            | `10`                                 | Candidates pulled              |
| `RERANK_TOP_N`               | `5`                                  | Kept after rerank              |
| `RAG_GROUNDING_GATE`         | `on`                                 | Toggle the answerability gate  |
| `LLM_MODEL`                  | `gpt-4o`                             | Answers / judges               |
| `FAST_LLM_MODEL`             | `gpt-4o-mini`                        | Gate / cheap tasks             |
| `RERANKER_MODEL`             | `jina-reranker-v2-base-multilingual` | Hosted reranker                |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95`                               | Cache-hit cosine bar           |
| `CACHE_TTL_SECONDS`          | `86400`                              | Cache entry lifetime           |
| chunk_size / overlap         | `1000 / 200`                         | In `chunking.get_splitter`     |

**Env vars:** `OPENAI_API_KEY`, `PINECONE_KEY`, `JINA_API_KEY`, `SUPABASE_URL`,
`SUPABASE_KEY`, `REDIS_URL`, plus optional `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` and the
`LLM_MODEL` / `RAG_GROUNDING_GATE` overrides.

---

## 10. Offline evaluation & CI · [`app/services/rag/eval_harness.py`](app/services/rag/eval_harness.py)

A golden-set grader that measures answer quality and can gate PRs. Metrics: **correctness**
(deterministic key-fact matching), **faithfulness** (LLM judge, or RAGAS via `--scorer
ragas`), **refusal_accuracy**, **over_refusal**. Frozen-context mode grades generation
without touching Pinecone.

```bash
python -m app.services.rag.eval_harness --self-test                                   # offline, no keys
python -m app.services.rag.eval_harness --data app/services/rag/datasets/rag_golden.jsonl    # real eval
python -m app.services.rag.eval_harness --data ... --gate                             # enforce thresholds
```

CI runs it **report-only** on every PR ([`.github/workflows/eval.yml`](.github/workflows/eval.yml)).
Full rationale + how to make it a hard gate: [RAG_EVALUATION_PLAN.md](RAG_EVALUATION_PLAN.md).

---

## 11. File map

| Concern                               | File                                                                                                                           |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| HTTP routes                           | [`app/routers/rag.py`](app/routers/rag.py)                                                                                     |
| Loading (Step 1)                      | [`app/services/rag/step1_ingestion.py`](app/services/rag/step1_ingestion.py)                                                                       |
| Chunking (Step 2)                     | [`app/services/rag/step2_chunking.py`](app/services/rag/step2_chunking.py)                                                                               |
| Embed / retrieve / rerank (Steps 3–5) | [`app/services/rag/step4_retrieval.py`](app/services/rag/step4_retrieval.py)                                                                       |
| Background ingest worker              | [`app/services/rag/step3_indexing_worker.py`](app/services/rag/step3_indexing_worker.py)                                                         |
| Context + citations (Step 6)          | [`app/services/rag/step5_generation.py`](app/services/rag/step5_generation.py)                                                                     |
| Grounding gate / sentinel             | [`app/services/rag/step6_grounding.py`](app/services/rag/step6_grounding.py)                                                                       |
| Semantic cache                        | [`app/services/cache.py`](app/services/cache.py)                                                                               |
| Online judges                         | [`app/services/rag/step7_evaluation.py`](app/services/rag/step7_evaluation.py)                                                                     |
| Persistence                           | [`app/services/rag/storage.py`](app/services/rag/storage.py)                                                                           |
| LLM clients                           | [`app/core/llm.py`](app/core/llm.py)                                                                                           |
| Prompt registry                       | [`app/core/prompts.py`](app/core/prompts.py)                                                                                   |
| Config / clients                      | [`app/core/config.py`](app/core/config.py)                                                                                     |
| Offline eval + golden set             | [`app/services/rag/eval_harness.py`](app/services/rag/eval_harness.py), [`app/services/rag/datasets/rag_golden.jsonl`](app/services/rag/datasets/rag_golden.jsonl) |

---

## 12. Known limitations & future work

- **BM25 not fit on the corpus** — uses generic MS-MARCO stats; fitting it on your documents
  would sharpen the keyword half of hybrid search.
- **No conversation history in retrieval** — follow-ups like "what about its price?" aren't
  contextualized before retrieval.
- **`RETRIEVER_TOP_K = 10`** is modest for a rerank pipeline; 20–30 often improves recall.
- **Embeddings default to `text-embedding-ada-002`** — `text-embedding-3-small` is cheaper/better at the same 1536 dims.
- **No token budgeting** in `prepare_context` (fine at top_n=5).
- **Ingestion via `BackgroundTasks`** — for real scale, move to a task queue (Celery/RQ/Arq)
  for retries and cross-restart durability.
- **Cache invalidation on new ingest** — a freshly added doc doesn't bust the `__all__`-scope
  cache for up to 24 h.
- **Golden eval set** is a 10-row seed; grow to 50–200 verified pairs.

## 13. future work need to implement

The biggest missing piece is feedback on latency: for slow first queries, stream a "Retrieving…/Reranking…/Generating…" status tied to the actual pipeline stage instead of a generic spinner, so the 22s wait feels intentional. Beyond that, consider: a copy button and thumbs up/down on answers; the ability to filter Top Sources by document or hide the low-relevance 0.02–0.04 hits automatically; showing the page number on citations (your API returns page_number, but the UI doesn't surface it); and a visible ingestion progress indicator when uploading (the API supports status polling per your own Features doc). The Analytics and Datasets tabs are marked "SOON" — an analytics view showing hallucination_rate and latency trends over time would pair naturally with the eval metrics you already compute.
