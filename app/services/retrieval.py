"""
services/retrieval.py
=====================
STEPS 3, 4 & 5 OF THE RAG PIPELINE, assembled in one place because they form a
single chain:  EMBEDDING → RETRIEVAL → RE-RANKING → CONTEXT ORDERING.

This module builds the retriever once at import time and exposes a single
`build_retriever(doc_ids)` helper that the routes call.

══════════════════════════════════════════════════════════════════════════════
STEP 3 — EMBEDDING
══════════════════════════════════════════════════════════════════════════════
Embedding = turning text into a vector so "similar meaning" ≈ "nearby vectors".

WHAT YOU HAVE: OpenAIEmbeddings (dense) + BM25Encoder (sparse).
  • DENSE  vectors capture *meaning* ("car" ≈ "automobile").
  • SPARSE vectors (BM25) capture *exact keywords* ("error code E-4021").
Using both = HYBRID search. This is a strong, modern choice.

IMPROVEMENTS / ALTERNATIVES:
  • Newer dense models: text-embedding-3-small (cheap) / -3-large (best). Match
    EMBEDDING_DIM in config to whatever you pick.
  • Open-source / self-hosted: BAAI bge-*, GTE, E5 via HuggingFaceEmbeddings —
    no per-call cost, data stays in-house.
  • Domain-specific: fine-tune or pick a domain embedding (e.g. legal/medical)
    when general models confuse your jargon.
  • Your BM25Encoder uses `.default()` (pre-trained on MS MARCO). For a
    specialized corpus, FIT BM25 on YOUR documents (`bm25_encoder.fit(texts)`)
    and persist it — keyword stats then match your vocabulary, not generic web text.

══════════════════════════════════════════════════════════════════════════════
STEP 4 — RETRIEVAL
══════════════════════════════════════════════════════════════════════════════
Retrieval = given a query vector, find the top_k nearest chunks.

WHAT YOU HAVE: PineconeHybridSearchRetriever (dense + sparse, dotproduct).

OTHER RETRIEVAL STRATEGIES & WHEN TO USE THEM:
  • PURE DENSE (similarity / MMR)
        Simplest. MMR ("maximal marginal relevance") trades a little relevance
        for diversity so you don't get 5 near-duplicate chunks.
        USE WHEN: conceptual questions, no rare keywords.
  • HYBRID (yours)
        Best general default — catches both meaning and exact terms (IDs, names,
        error codes) that pure-dense search misses.
  • METADATA-FILTERED  ← you already do this with `{"doc_id": {"$in": ...}}`
        Restrict search to a subset (one document, one user, a date range).
        USE WHEN: multi-tenant apps, "search only these 3 files".
  • MULTI-QUERY (you have it commented out — worth enabling)
        LLM rewrites the question into several phrasings, retrieves for each,
        unions the results. Fixes "user worded it differently than the doc".
        USE WHEN: recall matters and latency budget allows the extra LLM call.
  • SELF-QUERY
        LLM extracts metadata filters FROM the question ("invoices from 2023" →
        filter year=2023). USE WHEN: questions mix semantic + structured constraints.
  • PARENT-DOCUMENT / SMALL-TO-BIG
        Retrieve on small chunks, return big parents (see chunking.py).
  • GRAPH / HyDE / step-back
        Advanced: HyDE embeds a hypothetical answer instead of the question;
        graph retrieval walks entity relationships. USE WHEN: multi-hop reasoning.

══════════════════════════════════════════════════════════════════════════════
STEP 5 — RE-RANKING & CONTEXT ORDERING
══════════════════════════════════════════════════════════════════════════════
The retriever is fast but approximate. Re-ranking is a slower, more accurate
second pass over the top_k candidates.

WHAT YOU HAVE:
  • CrossEncoderReranker (bge-reranker-base): a cross-encoder reads (query, chunk)
    TOGETHER and scores true relevance, then keeps the top_n. This is the highest-
    ROI quality upgrade after hybrid search — keep it. It also writes
    `relevance_score` into each doc's metadata, which we surface as a citation
    confidence score.
  • LongContextReorder: LLMs attend best to the START and END of their context
    and "lose" the middle ("lost in the middle" effect). This reorders the kept
    chunks so the most relevant ones sit at the edges. Cheap, no model call — keep it.

RE-RANKING ALTERNATIVES:
  • Hosted rerankers: Cohere Rerank, Jina Reranker, Voyage Rerank — often stronger
    than bge-base, no GPU to manage (paid API).
  • Bigger local cross-encoder: bge-reranker-large / -v2-m3 for more accuracy at
    higher latency.
  • LLM-as-reranker: ask the LLM to score/order chunks. Most accurate, most
    expensive — reserve for low-volume, high-stakes queries.
RULE OF THUMB: retrieve MANY (top_k 20–50) cheaply, then rerank DOWN to a few
(top_n 3–8). A wider retrieve + rerank usually beats a narrow retrieve alone.

══════════════════════════════════════════════════════════════════════════════
STEP 6 — CONTEXT WINDOW MANAGEMENT (handled where context is built)
══════════════════════════════════════════════════════════════════════════════
After reranking you have a few high-quality chunks. Managing what actually goes
into the LLM prompt is its own discipline:
  • TOKEN BUDGETING: cap total context tokens; don't blindly concatenate. If the
    reranked set exceeds budget, drop lowest-scored chunks first.
  • DEDUPLICATION: `build_sources()` already dedupes by chunk prefix — good.
  • CONTEXTUAL COMPRESSION: summarise/extract only the relevant spans of each
    chunk before sending (LLMChainExtractor). Saves tokens on long chunks.
  • CONVERSATION HISTORY: for multi-turn chat, you must also budget prior turns.
    Summarise older turns or keep a rolling window so history + context fit.
  • ORDERING: LongContextReorder (above) is your "lost in the middle" mitigation.
"""

from typing import Any, Dict, List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_transformers import LongContextReorder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pinecone_text.sparse import BM25Encoder
from pydantic import ConfigDict

from app.core.config import (
    pinecone_index,
    RETRIEVER_TOP_K,
    RERANK_TOP_N,
    RERANKER_MODEL,
)


class _FilteredHybridRetriever(PineconeHybridSearchRetriever):
    """Thin subclass that adds optional metadata filtering to Pinecone queries."""

    metadata_filter: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        if self.metadata_filter:
            kwargs["filter"] = self.metadata_filter
        return super()._get_relevant_documents(query, run_manager=run_manager, **kwargs)


# ── Step 3: embedders (built once) ───────────────────────────────────────────
embeddings = OpenAIEmbeddings()

# Pre-trained BM25 (MS MARCO). For a specialized corpus, fit on your own texts
# and persist with bm25_encoder.dump(path) / load with .load(path).
bm25_encoder = BM25Encoder().default()

# ── Step 5: reranker + reorderer (built once; models are expensive to load) ──
_cross_encoder = None


def _get_reranker():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    return CrossEncoderReranker(model=_cross_encoder, top_n=RERANK_TOP_N)


reorder = LongContextReorder()


def _base_hybrid_retriever(doc_ids: Optional[List[str]]) -> _FilteredHybridRetriever:
    """
    Step 4: the hybrid (dense + sparse) retriever, optionally scoped to a set of
    document ids via Pinecone metadata filtering.
    """
    return _FilteredHybridRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=pinecone_index,
        top_k=RETRIEVER_TOP_K,
        metadata_filter={"doc_id": {"$in": doc_ids}} if doc_ids else None,
    )


# Unscoped retriever reused for ingestion writes and "search everything" reads.
_default_hybrid = _base_hybrid_retriever(doc_ids=None)


def hybrid_add_texts(texts: List[str], metadatas: List[dict]) -> None:
    """
    INGESTION-SIDE write: embed `texts` with BOTH dense (OpenAI) and sparse (BM25)
    encoders and upsert them into Pinecone in one call.

    The same hybrid retriever object knows how to write the dual vectors, which
    is why ingestion lives next to retrieval — they must use the same encoders,
    or your stored vectors won't match your query vectors.
    """
    _default_hybrid.add_texts(texts, metadatas=metadatas)


def build_retriever(
    doc_ids: Optional[List[str]] = None,
) -> ContextualCompressionRetriever:
    """
    Assemble the full retrieval chain:  hybrid retrieve → cross-encoder rerank.

    `reorder` (LongContextReorder) is applied AFTER retrieval, when building the
    context string (see generation.build_context) — it's a transform on the final
    doc list, not part of the retriever.

    Args:
        doc_ids: restrict search to these ingestion ids. None = search everything.
    """
    base = _default_hybrid if not doc_ids else _base_hybrid_retriever(doc_ids)
    return ContextualCompressionRetriever(
        base_compressor=_get_reranker(), base_retriever=base
    )
