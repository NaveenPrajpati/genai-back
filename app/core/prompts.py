"""
core/prompts.py
===============
VERSIONED PROMPT REGISTRY.

Prompts are part of the system's architecture — a wording change can move
accuracy, cost, and safety as much as a code change can — so they live here under
explicit version tags instead of being scattered as string literals across routes
and services.

CONVENTION
  • Every prompt is a `Prompt(name, version, template)`.
  • BUMP `version` (date.seq) whenever you change the wording; never edit a prompt
    silently. Log `prompt.version` alongside answers so you can A/B wording changes
    and roll back a regression to the exact text that shipped.
  • `REGISTRY` maps name → Prompt for that logging / lookup.

CHANGELOG
  2026-07-06.1  Initial extraction from core/llm.py & services/evaluation.py.
                RAG answer prompt gains inline-citation + refusal-sentinel rules
                (grounding enforcement). Added the answerability gate prompt.
  2026-07-08.1  eval_relevance: judge RELEVANCE, not sufficiency. The old wording
                ("does this chunk answer the query") scored NO on every chunk for
                "which/main/best" questions where no single chunk is dispositive,
                collapsing retrieval_precision to a false 0.0 even when retrieval
                and the final answer were correct.
  2026-07-21.1  rag_answer + rag_answerability: partial answers over all-or-
                nothing refusal. Both prompts framed sufficiency against the WHOLE
                question, so a compound ask ("job title AND years of experience")
                refused outright when only one part was thinly supported — even
                with correct chunks retrieved at 0.77/0.74. Each part answered
                fine in isolation. Same failure mode as 2026-07-08.1 above.
  2026-07-22.1  eval_hallucination: stop penalising honest "not covered"
                disclaimers. A partial-coverage compound answer ("…pricing is X
                [1]. The documents don't cover the refund policy.") scored 0.5
                faithfulness because the judge counted the absence-of-fact
                disclaimer as an unsupported claim — dragging the metric down for
                exactly the grounded behaviour rag_answer 2026-07-21.1 introduced.
                Now only POSITIVE claims are judged for support.
"""

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

PROMPTS_VERSION = "2026-07-22.1"

# ── Grounding sentinels ──────────────────────────────────────────────────────
# The exact string the model must emit when the context can't support an answer.
# The route detects it and substitutes REFUSAL_MESSAGE instead of showing it.
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

# User-facing text shown when we decline to answer (gate failed or sentinel hit).
REFUSAL_MESSAGE = (
    "I couldn't find enough support for that in the provided documents, so I "
    "won't guess. Try rephrasing your question, or add a document that covers it."
)


@dataclass(frozen=True)
class Prompt:
    """A named, versioned prompt template.

    Supports `prompt | llm` (delegates to the underlying template) so call sites
    read the same as before, while carrying `name`/`version` for tracing.
    """

    name: str
    version: str
    template: ChatPromptTemplate

    def __or__(self, other):
        return self.template | other


# ── RAG answer (grounding-enforced) ──────────────────────────────────────────
# The core anti-hallucination contract: answer ONLY from the numbered context,
# cite every claim as [n], and emit the refusal sentinel when the context is
# insufficient rather than inventing a plausible-sounding answer.
RAG_ANSWER = Prompt(
    name="rag_answer",
    version="2026-07-21.1",
    template=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions strictly from the provided context. The "
                "context is a list of numbered sources, each prefixed like [1], [2].\n"
                "RULES:\n"
                "1. Use ONLY facts stated in the context. Never rely on outside or "
                "prior knowledge.\n"
                "2. After each sentence, cite the source number(s) it came from "
                "inline, e.g. 'Revenue grew 12% [2].' Every claim must be traceable "
                "to a numbered source.\n"
                "3. A question may ask for several things at once. Answer EVERY "
                "part the context supports, and for any part it does not, say "
                "plainly that the documents don't cover it. Never withhold a "
                "supported answer because a different part of the question is "
                "unsupported.\n"
                "4. Only if the context supports NO part of the question, reply "
                "with EXACTLY the following token and nothing else: "
                f"{INSUFFICIENT_CONTEXT}\n"
                "5. When refusing, do not apologise, speculate, or add caveats — "
                "output only that token.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    ),
)


# ── Answerability gate (pre-generation grounding check) ──────────────────────
# A cheap, deterministic gate run BEFORE generation: decide whether the context
# actually contains the facts the question asks for. Used with structured output
# (see services/grounding.py) so the decision is a hard boolean, not free text.
ANSWERABILITY = Prompt(
    name="rag_answerability",
    version="2026-07-21.1",
    template=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a gate deciding whether a question can be answered using "
                "ONLY the provided context. Set can_answer=true if the context "
                "states the specific facts for ANY part of the question — a "
                "question with several parts is answerable when at least one part "
                "is supported, since the answering step reports the uncovered "
                "parts itself. Set can_answer=false only when the context supports "
                "no part of the question, or relates to it only tangentially. "
                "Never use outside knowledge.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    ),
)


# ── Evaluation judges (LLM-as-judge) ─────────────────────────────────────────
EVAL_RELEVANCE = Prompt(
    name="eval_relevance",
    version="2026-07-08.1",
    template=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a relevance judge for a retrieval system. Decide whether the "
                "document chunk is RELEVANT to the query — that is, it is on-topic and "
                "contains information that could contribute to an answer, even "
                "partially. Judge relevance, NOT sufficiency: a chunk counts as "
                "relevant if it supplies any useful fact toward the query, even if it "
                "does not fully or directly answer it on its own. Reply with exactly "
                "one word: YES if relevant, or NO if it is off-topic or useless for "
                "the query.",
            ),
            ("human", "Query: {question}\n\nDocument: {content}"),
        ]
    ),
)

EVAL_RECALL = Prompt(
    name="eval_recall",
    version="2026-07-06.1",
    template=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an evaluation assistant. Rate how completely the context "
                "contains the information needed to fully answer the question. "
                "0.0 = context is useless, 1.0 = context fully covers the answer. "
                "Return only a decimal number.",
            ),
            ("human", "Question: {question}\n\nContext: {context}"),
        ]
    ),
)

EVAL_HALLUCINATION = Prompt(
    name="eval_hallucination",
    version="2026-07-22.1",
    template=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a hallucination detector. Rate what fraction of the answer "
                "contains claims NOT supported by the provided context. "
                "0.0 = fully grounded, 1.0 = fully hallucinated.\n"
                "A hallucination is a POSITIVE claim of fact that the context does "
                "not support. Do NOT count as hallucination a statement that the "
                "context LACKS some information — e.g. 'the documents don't cover "
                "the refund policy' or 'the context does not mention X'. Such a "
                "disclaimer asserts the ABSENCE of a fact, which is the honest, "
                "grounded thing to do, not an invented fact. Judge only the "
                "positive claims for support. Return only a decimal number.",
            ),
            ("human", "Context: {context}\n\nAnswer: {answer}"),
        ]
    ),
)


REGISTRY = {
    p.name: p
    for p in (
        RAG_ANSWER,
        ANSWERABILITY,
        EVAL_RELEVANCE,
        EVAL_RECALL,
        EVAL_HALLUCINATION,
    )
}
