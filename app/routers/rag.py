"""
routers/rag.py
==============
The HTTP layer for the RAG pipeline. ROUTES SHOULD BE THIN: parse the request,
call services, shape the response. All real work lives in app/services/*.

Endpoints:
  GET    /rag/get-files          list ingestion logs
  POST   /rag/ingest/{action}    queue a file/url for ingestion (returns job_id)
  GET    /rag/ingest/status/{id} poll ingestion status
  DELETE /rag/ingest/{doc_id}    delete an ingestion (vectors + log row)
  POST /rag/query                one-shot answer (non-streaming)
  POST /rag/query/stream         token-by-token SSE answer, with semantic cache
"""

import os
import json
import uuid
import shutil
import asyncio
import logging
import tempfile
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, BackgroundTasks, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.llm import llm, RAG_PROMPT
from app.services import storage, cache
from app.services.ingestion import load_url, build_file_loader, SUPPORTED_FILE_TYPES
from app.services.ingestion_worker import run_ingestion, INGESTION_JOBS
from app.services.retrieval import build_retriever, get_embeddings, delete_doc_vectors
from app.services.generation import build_context, build_sources
from app.services.evaluation import run_evaluation

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rag", tags=["rag"], responses={404: {"description": "Not found"}}
)


# ── Models ───────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str
    evaluate: bool = False
    ingestions: List[str] = []  # doc_ids to scope the search; empty = all
    chat_id: Optional[str] = None  # omit to auto-create a new chat


def _sse(event: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"data: {json.dumps(event)}\n\n"


# ── Ingestion routes ─────────────────────────────────────────────────────────


@router.get("/get-files", status_code=200)
async def get_all_files():
    try:
        return {"message": "list fetched", "data": storage.list_ingestion_logs()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch files: {e}")


def _resolve_file_mime(file: UploadFile) -> Optional[str]:
    """
    Map an upload to a supported MIME key. Prefer the declared content_type, but
    fall back to the filename extension — clients often send PDFs as
    `application/octet-stream` or `application/x-pdf`.
    """
    if file.content_type in SUPPORTED_FILE_TYPES:
        return file.content_type
    name = (file.filename or "").lower()
    for mime, spec in SUPPORTED_FILE_TYPES.items():
        if name.endswith(spec["suffix"]):
            return mime
    return None


@router.post("/ingest/{action}", status_code=202)
async def ingest_document(
    background_tasks: BackgroundTasks,
    action: str,
    data: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Queue a source for ingestion and return immediately with a job_id.

    action == "url"  → body `data` is JSON {"url": "..."}
    otherwise        → multipart file upload (pdf or txt); `data` is not required
    """
    job_id = str(uuid.uuid4())
    tmp_path: Optional[str] = None

    if action == "url":
        if not data:
            raise HTTPException(
                status_code=400, detail='`data` form field with JSON {"url": "..."} is required'
            )
        url = json.loads(data)["url"]
        loader_fn = lambda: load_url(url)
        display_source, file_type = url, "url"
    else:
        if not file:
            raise HTTPException(status_code=400, detail="File is required")
        mime = _resolve_file_mime(file)
        if not mime:
            raise HTTPException(
                status_code=400, detail="Only PDF and text files are supported"
            )

        suffix = SUPPORTED_FILE_TYPES[mime]["suffix"]
        file_type = SUPPORTED_FILE_TYPES[mime]["file_type"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        loader_fn = build_file_loader(mime, tmp_path)
        display_source = file.filename

    queued_at = datetime.now(timezone.utc).isoformat()
    INGESTION_JOBS[job_id] = {
        "status": "queued",
        "source": display_source,
        "queued_at": queued_at,
    }
    await asyncio.to_thread(
        storage.create_ingestion_log, job_id, display_source, file_type, queued_at
    )

    background_tasks.add_task(
        run_ingestion, job_id, loader_fn, display_source, file_type, tmp_path
    )
    return {"job_id": job_id, "status": "queued", "source": display_source}


@router.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    job = INGESTION_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/ingest/{doc_id}", status_code=200)
async def delete_ingestion(doc_id: str):
    """
    Delete an ingestion entirely: drop its vectors from Pinecone and remove its
    Supabase log row. `doc_id` is the job_id returned by POST /rag/ingest/{action}.
    """
    log = await asyncio.to_thread(storage.get_ingestion_log, doc_id)
    if not log:
        raise HTTPException(status_code=404, detail="Ingestion not found")

    try:
        vectors_deleted = await asyncio.to_thread(delete_doc_vectors, doc_id)
        await asyncio.to_thread(storage.delete_ingestion_log, doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete ingestion: {e}")

    INGESTION_JOBS.pop(doc_id, None)  # drop any stale in-memory job state
    return {
        "message": "ingestion deleted",
        "doc_id": doc_id,
        "vectors_deleted": vectors_deleted,
    }


@router.post("/query")
async def query_documents(request: QueryRequest):
    retriever = build_retriever(request.ingestions or None)
    docs = retriever.invoke(request.question)

    if not docs:
        return {
            "answer": "No relevant documents found. Please upload documents first.",
            "sources": [],
        }

    context = build_context(docs)
    response = await (RAG_PROMPT | llm).ainvoke(
        {"context": context, "question": request.question}
    )

    result: dict = {"answer": response.content, "sources": build_sources(docs)}
    if request.evaluate:
        result["evaluation"] = await run_evaluation(
            request.question, docs, context, response.content
        )
    return result


# ── Query (streaming + semantic cache) ───────────────────────────────────────


@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    scope = cache.scope_key(request.ingestions)

    # Resolve or create the chat session before streaming starts.
    chat_id = request.chat_id or await asyncio.to_thread(
        create_chat_title, request.question
    )

    async def generate():
        try:
            # Embed once; reused for both the cache check and (on miss) retrieval.
            query_embedding = await asyncio.to_thread(
                get_embeddings().embed_query, request.question
            )

            # ── Semantic cache check ────────────────────────────────────────
            cached = await cache.lookup(query_embedding, scope)
            if cached:
                logger.info("Semantic cache hit: %s", request.question[:60])
                yield _sse(
                    {"type": "sources", "sources": cached["sources"], "cached": True}
                )
                for word in cached["answer"].split(" "):
                    yield _sse({"type": "token", "token": word + " "})
                    await asyncio.sleep(0)
                asyncio.ensure_future(
                    asyncio.to_thread(
                        storage.save_messages,
                        chat_id,
                        request.question,
                        cached["answer"],
                        cached["sources"],
                        request.ingestions,
                    )
                )
                yield _sse({"type": "done", "cached": True, "chat_id": chat_id})
                return

            # ── Cache miss: retrieve → rerank → generate ────────────────────
            retriever = build_retriever(request.ingestions or None)
            docs = await asyncio.to_thread(retriever.invoke, request.question)
            if not docs:
                yield _sse(
                    {
                        "type": "error",
                        "message": "No relevant documents found. Please upload documents first.",
                    }
                )
                return

            context = build_context(docs)
            sources = build_sources(docs)
            yield _sse({"type": "sources", "sources": sources})

            full_answer = ""
            async for chunk in (RAG_PROMPT | llm).astream(
                {"context": context, "question": request.question}
            ):
                if chunk.content:
                    full_answer += chunk.content
                    yield _sse({"type": "token", "token": chunk.content})

            # Fire-and-forget: cache the result + persist the messages.
            asyncio.ensure_future(
                cache.save(query_embedding, scope, sources, full_answer)
            )
            asyncio.ensure_future(
                asyncio.to_thread(
                    storage.save_messages,
                    chat_id,
                    request.question,
                    full_answer,
                    sources,
                    request.ingestions,
                )
            )

            if request.evaluate:
                evaluation = await run_evaluation(
                    request.question, docs, context, full_answer
                )
                yield _sse({"type": "evaluation", "evaluation": evaluation})

            yield _sse({"type": "done", "chat_id": chat_id})

        except Exception as exc:
            logger.exception("Stream generation failed")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_chat_title(question: str) -> str:
    """Create a chat using the first 200 chars of the question as the title."""
    return storage.create_chat(question[:200])
