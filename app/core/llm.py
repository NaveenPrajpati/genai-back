"""
core/llm.py
===========
The shared LLM client and the RAG answer prompt.

Kept separate from config.py because both the generation route and the
evaluation judges import the same `llm` instance — defining it once avoids
spinning up multiple clients and keeps the model choice in one obvious place.
"""

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from app.core.config import LLM_MODEL, FAST_LLM_MODEL
from app.core.llm_capture import build_capture_callbacks
from app.core.metrics import build_metrics_callbacks
from langchain.chat_models import init_chat_model

# temperature=0 → deterministic, grounded answers (you almost always want this
# for RAG; creativity here just means more hallucination).
#
# `callbacks` attaches two propagating handlers (both no-op-safe):
#   • the distillation capture handler when LLM_CAPTURE=1 (see core/llm_capture.py)
#   • the Prometheus cost/token handler (see core/metrics.py), always on
# Because callbacks propagate to the underlying model, they observe EVERY call —
# plain, with_structured_output, and bind_tools — across all agents, with zero
# changes to any agent code. Built once and shared by both clients (each record's
# `model` field still distinguishes top vs fast).
_callbacks = build_capture_callbacks() + build_metrics_callbacks()

# stream_usage=True makes OpenAI emit token counts on the FINAL streamed chunk, so
# the cost handler can price streaming RAG answers (astream) — without it,
# usage_metadata is absent on streamed calls and cost-per-query undercounts.
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    stream_usage=True,
    callbacks=_callbacks or None,
)

# Fast model: trivial classification / extraction / selection (intent routing,
# memory extraction, task/note/selector parsing). ~10-15x cheaper than the top
# model with no meaningful quality loss on these narrow tasks.
fast_llm = ChatOpenAI(
    model=FAST_LLM_MODEL,
    temperature=0,
    stream_usage=True,
    callbacks=_callbacks or None,
)
# llm = ChatOllama(
#     model="llama3:latest",
#     temperature=0,
#     # other params...
# )

# Prompts now live in core/prompts.py (versioned registry). The RAG answer prompt
# is prompts.RAG_ANSWER — it carries the citation + refusal-sentinel rules that
# enforce grounding.
