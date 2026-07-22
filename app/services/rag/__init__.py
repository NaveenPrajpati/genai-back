"""
services/rag/
=============
The RAG pipeline as one self-contained package, files ordered by pipeline stage:

  step1_ingestion       — load raw files/URLs into documents
  step2_chunking         — split documents into chunks
  step3_indexing_worker  — background job: chunk → embed → upsert to the index
  step4_retrieval        — hybrid retrieve + rerank + context ordering
  step5_generation       — build the numbered context + citation sources
  step6_grounding        — answerability gate + refusal-sentinel enforcement
  step7_evaluation       — online LLM-as-judge metrics (optional)
  storage                — Supabase persistence (chats, messages, ingestion logs)
  eval_harness           — offline golden-set eval + CI gate

Shared infrastructure deliberately lives OUTSIDE this package because the agents
use it too: core.llm, core.prompts, core.config, and services.cache.
"""
