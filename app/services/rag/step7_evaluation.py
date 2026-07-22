"""
services/rag/step7_evaluation.py
======================
STEP 7 (OPTIONAL): EVALUATION

After generating an answer you often want to know: was the retrieval good, and
is the answer grounded? These LLM-as-judge metrics give a cheap, automatic signal.

WHAT YOU MEASURE:
  • retrieval_precision — of the chunks we retrieved, what fraction were actually
    relevant? (Did the retriever pull junk?)
  • recall_score       — did the context contain enough to FULLY answer? (Did we
    miss something we should have retrieved?)
  • hallucination_rate — what fraction of the answer is NOT supported by context?
    (Is the LLM making things up?)

These three map almost 1:1 to the core RAGAS metrics (context precision, context
recall, faithfulness). When you outgrow hand-rolled judges, consider the RAGAS
or DeepEval libraries, which add answer-relevance, semantic similarity, and
dataset-level aggregation.

CAVEAT: LLM judges are noisy. Use a strong judge model, keep prompts strict, and
treat scores as directional, not absolute. For anything high-stakes, build a
small human-labelled gold set and measure against it.
"""

import re
import asyncio
from typing import List

from app.core.llm import llm
from app.core.prompts import EVAL_RELEVANCE, EVAL_RECALL, EVAL_HALLUCINATION

# The judge must see WHAT THE SYSTEM ACTUALLY USED, or its score describes a
# different input than the reranker/generator saw and can't be trusted. The old
# code truncated a chunk to 600 chars and the whole context to 2000; with the
# SemanticChunker (percentile-95 breakpoints) a single chunk routinely runs past
# 600 chars, so the answer-bearing sentence fell outside the judged window and
# the relevance judge returned NO on chunks the answer correctly cited —
# retrieval_precision read 0.00 next to a grounded answer. These caps are only a
# runaway-token guard, set well above real chunk/context sizes, not a content
# window: judging must not silently drop the part that matters.
_MAX_DOC_CHARS = 6000       # one chunk; comfortably above p95 SemanticChunker size
_MAX_CONTEXT_CHARS = 24000  # the assembled multi-chunk context

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def _to_unit_float(text: str) -> float:
    """Parse a judge's reply into a clamped 0–1 float; 0.0 if no number is found.

    Judges are told to return a bare decimal, but a strict `float(text)` breaks the
    moment one adds a word ("0.8 — mostly covered"), silently scoring 0.0 and
    dragging the metric down. Pull the first number out instead of trusting the
    whole reply to be numeric."""
    match = _NUMBER_RE.search(text)
    if not match:
        return 0.0
    return round(max(0.0, min(1.0, float(match.group()))), 3)


async def _score_doc_relevance(question: str, content: str) -> float:
    """Binary relevance: 1.0 if this chunk helps answer the question, else 0.0."""
    msg = await (EVAL_RELEVANCE | llm).ainvoke(
        {"question": question, "content": content[:_MAX_DOC_CHARS]}
    )
    return 1.0 if "YES" in msg.content.upper() else 0.0


async def _score_recall(question: str, context: str) -> float:
    """How completely does the context cover the answer? 0 = useless, 1 = complete."""
    msg = await (EVAL_RECALL | llm).ainvoke(
        {"question": question, "context": context[:_MAX_CONTEXT_CHARS]}
    )
    return _to_unit_float(msg.content)


async def _score_hallucination(context: str, answer: str) -> float:
    """Fraction of the answer NOT supported by context. 0 = grounded, 1 = made up."""
    msg = await (EVAL_HALLUCINATION | llm).ainvoke(
        {"context": context[:_MAX_CONTEXT_CHARS], "answer": answer}
    )
    return _to_unit_float(msg.content)


async def run_evaluation(question: str, docs: list, context: str, answer: str) -> dict:
    """Run precision (per-doc), recall, and hallucination checks concurrently."""
    relevance_tasks = [_score_doc_relevance(question, d.page_content) for d in docs]
    results = await asyncio.gather(
        *relevance_tasks,
        _score_recall(question, context),
        _score_hallucination(context, answer),
    )
    relevance_scores: List[float] = list(results[: len(docs)])
    recall_score = results[-2]
    hallucination_rate = results[-1]
    precision = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    return {
        "retrieval_precision": round(precision, 3),
        "recall_score": recall_score,
        "hallucination_rate": hallucination_rate,
    }
