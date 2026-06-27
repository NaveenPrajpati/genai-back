"""
services/generation.py
======================
STEP 6 HELPERS: turning reranked docs into (a) the context string the LLM sees
and (b) the source citations the user sees.

These are small, pure functions deliberately separated from the route so they
can be unit-tested without spinning up FastAPI, Pinecone, or the LLM.
"""

from typing import List

from app.services.retrieval import reorder


def build_context(docs: list) -> str:
    """
    Build the LLM context string from reranked docs.

    Applies LongContextReorder first (mitigates "lost in the middle" — see
    retrieval.py) then joins with blank lines so the model sees clear chunk
    boundaries.

    PRO TIP: this is where you'd enforce a TOKEN BUDGET. Right now we concatenate
    everything; for long chunks, count tokens and drop the lowest-ranked chunks
    until you're under your context limit, leaving room for the answer.
    """
    ordered = reorder.transform_documents(docs)
    return "\n\n".join(doc.page_content for doc in ordered)


def build_sources(docs: list) -> List[dict]:
    """
    Deduplicated source citations with chunk text, page, and confidence score.

    Dedup key is the first 120 chars of the chunk — cheap way to drop near-
    duplicate hits. `relevance_score` is written into metadata by the
    JinaRerank reranker, so it doubles as a confidence signal for the UI.
    """
    seen: set = set()
    sources: List[dict] = []
    for doc in docs:
        key = doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)

        raw_score = doc.metadata.get("relevance_score")
        sources.append({
            "chunk_text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page_number": doc.metadata.get("page"),   # PyPDFLoader sets "page" (0-indexed)
            "confidence_score": round(float(raw_score), 4) if raw_score is not None else None,
            "doc_id": doc.metadata.get("doc_id"),
        })
    return sources
