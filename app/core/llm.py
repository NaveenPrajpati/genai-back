"""
core/llm.py
===========
The shared LLM client and the RAG answer prompt.

Kept separate from config.py because both the generation route and the
evaluation judges import the same `llm` instance — defining it once avoids
spinning up multiple clients and keeps the model choice in one obvious place.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.core.config import LLM_MODEL
from app.core.llm_capture import build_capture_callbacks

# temperature=0 → deterministic, grounded answers (you almost always want this
# for RAG; creativity here just means more hallucination).
#
# `callbacks` attaches the distillation capture handler when LLM_CAPTURE=1 (no-op
# otherwise). Because callbacks propagate to the underlying model, this records
# every call — plain, with_structured_output, and bind_tools — for fine-tuning
# data, without touching any agent code. See core/llm_capture.py.
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    callbacks=build_capture_callbacks() or None,
)
# llm = ChatOllama(
#     model="llama3:latest",
#     temperature=0,
#     # other params...
# )

# The single source of truth for how we instruct the model to answer.
# "based only on the provided context" + "say so if not enough info" is the core
# anti-hallucination instruction of RAG. Strengthen it further if needed, e.g.
# "Cite the source after each claim" or "If unsure, reply 'I don't know'".
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the user's question based only on the "
        "provided context. If the context doesn't contain enough information to "
        "answer, say so.",
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
