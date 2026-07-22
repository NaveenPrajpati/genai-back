"""
services/rag/step5_generation.py
======================
STEP 6 HELPERS: turning reranked docs into (a) the numbered context string the
LLM sees and (b) the source citations the user sees.

Context and sources share ONE citation numbering so a `[n]` the model writes in
its answer maps back to exactly `sources[n]`. That mapping is what makes citation
enforcement possible (see step6_grounding.py): we can check the answer only
cites real, retrieved sources.

These are small, pure functions deliberately separated from the route so they can
be unit-tested without spinning up FastAPI, Pinecone, or the LLM.
"""

from typing import List, Tuple

from app.services.rag.step4_retrieval import reorder


def _dedup(docs: list) -> list:
    """Drop near-duplicate hits (same first 120 chars), preserving rank order."""
    seen: set = set()
    unique: list = []
    for doc in docs:
        key = doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _source_of(doc, citation: int) -> dict:
    """One citation record: the chunk, where it came from, and its rerank score."""
    raw_score = doc.metadata.get("relevance_score")
    return {
        "citation": citation,  # the [n] the model cites; maps answer → source
        "chunk_text": doc.page_content,
        "source": doc.metadata.get("source", "unknown"),
        "page_number": doc.metadata.get("page"),  # PyPDFLoader sets "page" (0-indexed)
        "confidence_score": round(float(raw_score), 4) if raw_score is not None else None,
        "doc_id": doc.metadata.get("doc_id"),
    }


def prepare_context(docs: list) -> Tuple[str, List[dict]]:
    """
    Build (context_string, sources) from reranked docs, sharing one numbering.

    Citation numbers are assigned by RELEVANCE order (the reranked order, before
    reorder) so the best source is [1]. The context string is then LongContext-
    Reorder'd (mitigates "lost in the middle" — see retrieval.py) but keeps each
    chunk's original [n] label, so reordering never breaks the answer→source map.
    """
    unique = _dedup(docs)
    numbers = {id(doc): i for i, doc in enumerate(unique, start=1)}
    sources = [_source_of(doc, numbers[id(doc)]) for doc in unique]

    ordered = reorder.transform_documents(unique)
    context = "\n\n".join(f"[{numbers[id(doc)]}] {doc.page_content}" for doc in ordered)
    return context, sources


def cited_sources(sources: List[dict], cited: set) -> List[dict]:
    """Filter `sources` to those the answer actually cited. Falls back to all
    sources when the answer cited nothing parseable (so the UI still shows
    provenance) — the caller decides how to treat an uncited answer."""
    used = [s for s in sources if s["citation"] in cited]
    return used or sources
