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
from langchain.chat_models import init_chat_model

# temperature=0 → deterministic, grounded answers (you almost always want this
# for RAG; creativity here just means more hallucination).
#
# `callbacks` attaches the distillation capture handler when LLM_CAPTURE=1 (no-op
# otherwise). Because callbacks propagate to the underlying model, this records
# every call — plain, with_structured_output, and bind_tools — for fine-tuning
# data, without touching any agent code. See core/llm_capture.py. Built once and
# shared by both clients so every call lands in one capture file (the record's
# `model` field still distinguishes top vs fast).
_capture = build_capture_callbacks()

# Top model: generation-heavy / correctness-critical work (roadmap, plan,
# research, synthesis, RAG answers, eval judges).
llm = ChatOpenAI(model=LLM_MODEL, temperature=0, callbacks=_capture or None)

# Fast model: trivial classification / extraction / selection (intent routing,
# memory extraction, task/note/selector parsing). ~10-15x cheaper than the top
# model with no meaningful quality loss on these narrow tasks.
fast_llm = ChatOpenAI(model=FAST_LLM_MODEL, temperature=0, callbacks=_capture or None)
# llm = ChatOllama(
#     model="llama3:latest",
#     temperature=0,
#     # other params...
# )

# Prompts now live in core/prompts.py (versioned registry). The RAG answer prompt
# is prompts.RAG_ANSWER — it carries the citation + refusal-sentinel rules that
# enforce grounding.
