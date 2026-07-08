"""
services/grounding.py
=====================
CITATION / GROUNDING ENFORCEMENT.

The goal: the system should DECLINE to answer when the retrieved chunks don't
actually support a response, instead of emitting something plausible-sounding but
unsupported. Two layers, defence in depth:

  1. ANSWERABILITY GATE (`is_answerable`) — a cheap fast-model call BEFORE
     generation that decides, deterministically (structured bool), whether the
     context contains the facts the question asks for. This is the primary gate,
     and it works for streaming too: we decide to refuse BEFORE any token is sent,
     so we never have to "un-stream" a bad answer.

  2. SENTINEL BACKSTOP (`is_refusal`) — even past the gate, the answer prompt is
     instructed to emit INSUFFICIENT_CONTEXT if it finds the context lacking. The
     route detects that token and shows the refusal message instead.

Plus `cited_numbers` extracts the [n] citations the model wrote, so the route can
verify the answer cites real retrieved sources and surface only those.

The gate FAILS OPEN (returns True on error): a transient LLM/network blip should
degrade to "let the answer prompt's sentinel handle it", not hard-block every
query. The sentinel backstop still guards grounding in that case.
"""

import re
import logging

from pydantic import BaseModel, Field

from app.core.llm import fast_llm
from app.core.prompts import ANSWERABILITY, INSUFFICIENT_CONTEXT
from app.core.config import RAG_GROUNDING_GATE

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


class _Answerable(BaseModel):
    can_answer: bool = Field(
        description="True only if the context states the specific facts the "
        "question asks for; False if missing, partial, or tangential."
    )


async def is_answerable(question: str, context: str) -> bool:
    """Gate: can this question be answered from this context alone? Cheap fast-LLM
    structured call. Disabled (always True) when RAG_GROUNDING_GATE is off."""
    if not RAG_GROUNDING_GATE:
        return True
    try:
        chain = ANSWERABILITY.template | fast_llm.with_structured_output(_Answerable)
        result = await chain.ainvoke({"question": question, "context": context})
        return bool(result.can_answer)
    except Exception:
        # Fail open — the answer prompt's INSUFFICIENT_CONTEXT sentinel still guards
        # grounding, so a gate hiccup shouldn't block every request.
        logger.warning("Answerability gate failed — falling back to sentinel guard")
        return True


def is_refusal(answer: str) -> bool:
    """True if the model emitted the insufficient-context sentinel."""
    return answer.strip().upper().startswith(INSUFFICIENT_CONTEXT)


def cited_numbers(answer: str) -> set:
    """The set of source numbers the answer cited inline as [n]."""
    return {int(n) for n in _CITATION_RE.findall(answer)}
