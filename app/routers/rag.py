import os
import json
import time
import uuid
import shutil
import asyncio
import logging
import tempfile
from datetime import datetime, timezone
from typing import Optional, List, Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Form,
    UploadFile,
    File,
    HTTPException,
    Depends,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.dependencies import get_current_user
from app.core.config import EMBEDDING_MODEL
from app.core.llm import llm
from app.core.prompts import RAG_ANSWER, REFUSAL_MESSAGE, INSUFFICIENT_CONTEXT
from app.services import cache
from app.services.rag import storage
from app.services.rag.step1_ingestion import (
    load_url,
    load_pdf,
    load_txt,
    load_docx,
    load_image,
    SUPPORTED_FILE,
)
from app.services.rag.step3_indexing_worker import run_ingestion, INGESTION_JOBS
from app.services.rag.step4_retrieval import (
    build_retriever,
    get_embeddings,
    delete_doc_vectors,
    retrieve_and_rerank,
)
from app.services.rag.step5_generation import prepare_context, cited_sources
from app.services.rag.step6_grounding import is_answerable, is_refusal, cited_numbers
from app.services.rag.step7_evaluation import run_evaluation

logger = logging.getLogger(__name__)

# Strong references to fire-and-forget background tasks. asyncio.ensure_future
# returns a task the event loop keeps only a WEAK reference to, so an unreferenced
# one can be garbage-collected mid-await — which is why backgrounded cache/DB
# writes were silently lost and repeat questions kept missing the cache. Holding a
# reference until the task finishes prevents that.
_background_tasks: set = set()


def _spawn(coro) -> None:
    """Schedule `coro` as a background task and keep it alive until it completes."""
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


router = APIRouter(
    prefix="/rag", tags=["rag"], responses={404: {"description": "Not found"}}
)


class QueryRequest(BaseModel):
    question: str
    evaluate: bool = False
    ingestions: List[str] = []  # doc_ids to scope the search; empty = all
    chat_id: Optional[str] = None  # omit to auto-create a new chat


def _sse(event: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"data: {json.dumps(event)}\n\n"


@router.get("/get-ingestions", status_code=200)
async def get_all_files(
    current_user: Annotated[dict, Depends(get_current_user)],
):
    user_id = current_user["uid"]
    try:

        return {"message": "list fetched", "data": storage.list_ingestion_logs(user_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch files: {e}")


# Images are OCR'd (Tesseract), which is slow and memory-hungry, so uploads are
# capped — see MAX_IMAGE_BYTES and the size check in ingest_document.
MAX_IMAGE_BYTES = 2 * 1024 * 1024  # 2 MB

FILE_LOADERS = {
    "application/pdf": ("pdf", load_pdf),
    "text/plain": ("text", load_txt),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        "docx",
        load_docx,
    ),
    "image/jpeg": ("image", load_image),
    "image/png": ("image", load_image),
    "image/webp": ("image", load_image),
    "image/bmp": ("image", load_image),
    "image/tiff": ("image", load_image),
}


def _resolve_file_mime(file: UploadFile) -> Optional[str]:
    """
    Map an upload to a supported MIME key. Prefer the declared content_type, but
    fall back to the filename extension — clients often send PDFs as
    `application/octet-stream` or `application/x-pdf`.
    """
    if file.content_type in FILE_LOADERS:
        return file.content_type
    name = (file.filename or "").lower()
    for mime in FILE_LOADERS:
        if name.endswith(SUPPORTED_FILE[mime]):
            return mime
    return None


@router.post("/ingest/{action}", status_code=202)
async def ingest_document(
    background_tasks: BackgroundTasks,
    action: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    data: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Queue a source for ingestion and return immediately with a job_id.

    action == "url"  → body `data` is JSON {"url": "..."}
    otherwise        → multipart file upload (pdf, txt, docx, or image);
                       `data` is not required. Images (≤5 MB) are OCR'd.
    """
    job_id = str(uuid.uuid4())
    tmp_path: Optional[str] = None
    user_id = current_user["uid"]
    if action == "url":
        if not data:
            raise HTTPException(
                status_code=400,
                detail='`data` form field with JSON {"url": "..."} is required',
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
                status_code=400,
                detail="Only PDF, text, DOCX, and image (JPEG/PNG/WEBP/BMP/TIFF) files are supported",
            )

        file_type, loader = FILE_LOADERS[mime]
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=SUPPORTED_FILE[mime]
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Images are OCR'd — cap them at 5 MB to bound OCR latency/memory.
        if file_type == "image" and os.path.getsize(tmp_path) > MAX_IMAGE_BYTES:
            os.unlink(tmp_path)
            raise HTTPException(status_code=413, detail="Image exceeds the 5 MB limit")

        loader_fn = lambda: loader(tmp_path)
        display_source = file.filename

    queued_at = datetime.now(timezone.utc).isoformat()
    INGESTION_JOBS[job_id] = {
        "status": "queued",
        "source": display_source,
        "queued_at": queued_at,
    }
    await asyncio.to_thread(
        storage.create_ingestion_log,
        job_id,
        display_source,
        file_type,
        queued_at,
        user_id,
    )

    background_tasks.add_task(
        run_ingestion, job_id, loader_fn, display_source, file_type, tmp_path, user_id
    )
    return {"job_id": job_id, "status": "queued", "source": display_source}


@router.delete("/ingest/{doc_id}", status_code=200)
async def delete_ingestion(
    doc_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """
    Delete an ingestion entirely: drop its vectors from Pinecone and remove its
    Supabase log row. `doc_id` is the job_id returned by POST /rag/ingest/{action}.
    """
    log = await asyncio.to_thread(storage.get_ingestion_log, doc_id)
    # Return 404 (not 403) when the ingestion isn't the caller's — don't reveal
    # that a doc_id belonging to another user exists.
    if not log or log.get("user_id") != current_user["uid"]:
        raise HTTPException(status_code=404, detail="Ingestion not found")

    try:
        vectors_deleted = await asyncio.to_thread(delete_doc_vectors, doc_id)
        await asyncio.to_thread(storage.delete_ingestion_log, doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete ingestion: {e}")

    # Drop this user's cached answers — some may cite the doc we just removed.
    await cache.invalidate_user(current_user["uid"])

    INGESTION_JOBS.pop(doc_id, None)  # drop any stale in-memory job state
    return {
        "message": "ingestion deleted",
        "doc_id": doc_id,
        "vectors_deleted": vectors_deleted,
    }


@router.post("/query")
async def query_documents(
    request: QueryRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    retriever = build_retriever(current_user["uid"], request.ingestions or None)
    docs = retriever.invoke(request.question)

    if not docs:
        return {
            "answer": "No relevant documents found. Please upload documents first.",
            "sources": [],
        }

    context, sources = prepare_context(docs)

    # Grounding gate: refuse up front if the context can't support an answer.
    if not await is_answerable(request.question, context):
        return {"answer": REFUSAL_MESSAGE, "sources": [], "grounded": False}

    response = await (RAG_ANSWER | llm).ainvoke(
        {"context": context, "question": request.question}
    )
    answer = response.content

    # Backstop: the model itself judged the context insufficient.
    if is_refusal(answer):
        return {"answer": REFUSAL_MESSAGE, "sources": [], "grounded": False}

    cited = cited_numbers(answer)
    result: dict = {
        "answer": answer,
        "sources": cited_sources(sources, cited),
        "grounded": bool(cited),  # an answer that cited nothing isn't traceable
    }
    if request.evaluate:
        result["evaluation"] = await run_evaluation(
            request.question, docs, context, answer
        )
    return result


@router.post("/query/stream")
async def query_documents_stream(
    request: QueryRequest, current_user: Annotated[dict, Depends(get_current_user)]
):
    chat_id = request.chat_id
    user_id = current_user["uid"]
    scope = cache.scope_key(user_id, request.ingestions)

    async def generate():
        nonlocal chat_id
        try:
            t_start = time.perf_counter()
            timings: dict = {}  # stage name -> ms, summarised on the `done` event

            # Open the chat BEFORE streaming when the client didn't name one, and
            # announce the full row as the first frame so the UI can render it
            # immediately. Persistence is fire-and-forget, so a chat created down
            # in _persist wouldn't exist yet when `done` is emitted — the client
            # would get chat_id: null, have nothing to send back, and start a new
            # chat on every question.
            if not chat_id:
                chat = await asyncio.to_thread(
                    storage.create_chat, request.question, user_id
                )
                chat_id = chat["id"]
                yield _sse({"type": "chat", "chat": chat, "created": True})

            def _stage(
                name: str, ms: float = 0.0, info: str = "", skipped: bool = False
            ) -> str:
                """One pipeline-stage SSE frame with a REAL server-side timing."""
                evt: dict = {"type": "stage", "name": name}
                if skipped:
                    evt["skipped"] = True
                else:
                    evt["ms"] = round(ms, 1)
                    timings[name] = evt["ms"]
                if info:
                    evt["info"] = info
                return _sse(evt)

            def _done(**extra) -> str:
                total_ms = round((time.perf_counter() - t_start) * 1000, 1)
                return _sse(
                    {
                        "type": "done",
                        "chat_id": chat_id,
                        "timings": timings,
                        "total_ms": total_ms,
                        **extra,
                    }
                )

            # ── Embed (once; reused for the cache check and retrieval) ──────
            t = time.perf_counter()
            query_embedding = await asyncio.to_thread(
                get_embeddings().embed_query, request.question
            )
            yield _stage("embed", (time.perf_counter() - t) * 1000, EMBEDDING_MODEL)

            # ── Semantic cache check ────────────────────────────────────────
            t = time.perf_counter()
            cached = await cache.lookup(query_embedding, scope)
            # Report the scope on a miss: a cached answer is only replayed within
            # the same (user, doc set), so a doc selection that changes between
            # questions misses every time — and looks identical to a broken cache.
            n_scoped = await cache.scope_size(scope)
            yield _stage(
                "cache",
                (time.perf_counter() - t) * 1000,
                "hit" if cached else f"miss (scope={scope}, entries={n_scoped})",
            )

            if cached:
                logger.info("Semantic cache hit: %s", request.question[:60])
                for skipped_stage in ("retrieve", "rerank", "gate"):
                    yield _stage(skipped_stage, skipped=True)
                yield _sse(
                    {"type": "sources", "sources": cached["sources"], "cached": True}
                )
                t = time.perf_counter()
                for word in cached["answer"].split(" "):
                    yield _sse({"type": "token", "token": word + " "})
                    await asyncio.sleep(0)
                yield _stage("stream", (time.perf_counter() - t) * 1000, "cache replay")
                t = time.perf_counter()
                _spawn(
                    asyncio.to_thread(
                        storage.save_messages,
                        chat_id=chat_id,
                        question=request.question,
                        answer=cached["answer"],
                        user_id=user_id,
                        sources=cached["sources"],
                        ingestions=request.ingestions,
                    )
                )
                yield _stage("persist", (time.perf_counter() - t) * 1000, "async")
                yield _done(cached=True)
                return

            docs, rt = await asyncio.to_thread(
                retrieve_and_rerank,
                user_id,
                request.ingestions or None,
                request.question,
            )
            yield _stage(
                "retrieve", rt["retrieve_ms"], f"{rt['candidates']} candidates"
            )
            if not docs:
                yield _stage("rerank", skipped=True)
                yield _sse(
                    {
                        "type": "error",
                        "message": "No relevant documents found. Please upload documents first.",
                    }
                )
                return
            yield _stage("rerank", rt["rerank_ms"], f"kept {len(docs)}")

            context, sources = prepare_context(docs)

            def _persist(answer: str, srcs: list) -> None:
                """Fire-and-forget: save one Q&A turn to chat history."""
                _spawn(
                    asyncio.to_thread(
                        storage.save_messages,
                        chat_id=chat_id,
                        question=request.question,
                        answer=answer,
                        user_id=user_id,
                        sources=srcs,
                        ingestions=request.ingestions,
                    )
                )

            # ── Grounding gate: decide to refuse BEFORE emitting any token ────
            t = time.perf_counter()
            answerable = await is_answerable(request.question, context)
            yield _stage(
                "gate",
                (time.perf_counter() - t) * 1000,
                "answerable" if answerable else "refused",
            )
            if not answerable:
                yield _sse({"type": "sources", "sources": []})
                yield _sse({"type": "token", "token": REFUSAL_MESSAGE})
                yield _stage("stream", skipped=True)
                t = time.perf_counter()
                _persist(REFUSAL_MESSAGE, [])
                yield _stage("persist", (time.perf_counter() - t) * 1000, "async")
                yield _done(grounded=False)
                return

            yield _sse({"type": "sources", "sources": sources})

            # Stream, but hold back the leading text until we know it isn't the
            # refusal sentinel — so we never leak "INSUFFICIENT_CONTEXT" to the UI.
            t_gen = time.perf_counter()
            full_answer = ""
            buffer = ""
            streaming = False
            async for chunk in (RAG_ANSWER | llm).astream(
                {"context": context, "question": request.question}
            ):
                if not chunk.content:
                    continue
                full_answer += chunk.content
                if streaming:
                    yield _sse({"type": "token", "token": chunk.content})
                    continue
                buffer += chunk.content
                if len(buffer) < len(INSUFFICIENT_CONTEXT):
                    continue
                if is_refusal(buffer):
                    break  # sentinel confirmed — nothing has been streamed yet
                streaming = True  # safe: flush what we held and go live
                yield _sse({"type": "token", "token": buffer})

            # Stream ended (or broke) while still buffering — a short answer or the
            # sentinel. If it wasn't a refusal, flush the held text now.
            if not streaming and not is_refusal(full_answer):
                if buffer:
                    yield _sse({"type": "token", "token": buffer})
                    streaming = True

            if is_refusal(full_answer):
                yield _sse({"type": "token", "token": REFUSAL_MESSAGE})
                yield _stage("stream", (time.perf_counter() - t_gen) * 1000, "refused")
                t = time.perf_counter()
                _persist(REFUSAL_MESSAGE, [])
                yield _stage("persist", (time.perf_counter() - t) * 1000, "async")
                yield _done(grounded=False)
                return

            yield _stage(
                "stream",
                (time.perf_counter() - t_gen) * 1000,
                f"{len(full_answer)} chars",
            )

            # Surface which sources the answer actually cited, and keep only those.
            cited = cited_numbers(full_answer)
            used = cited_sources(sources, cited)
            yield _sse({"type": "citations", "cited": sorted(cited)})

            # Cache the result, then persist the messages. Only real, grounded
            # answers are cached — never a refusal.
            t = time.perf_counter()
            # Await the cache write: its entire purpose is to be found by the NEXT
            # identical question, and those Redis writes on the already-computed
            # embedding take ~1ms. Backgrounding it (as this once did) meant the
            # write routinely hadn't landed — or was GC'd — before the repeat came
            # in, so the same question missed over and over.
            await cache.save(query_embedding, scope, used, full_answer)
            # Chat-history persist stays backgrounded (a slower Supabase round-trip
            # the response doesn't need to wait on), now with a strong task ref.
            _persist(full_answer, used)
            yield _stage("persist", (time.perf_counter() - t) * 1000, "async")

            if request.evaluate:
                t = time.perf_counter()
                evaluation = await run_evaluation(
                    request.question, docs, context, full_answer
                )
                timings["evaluate"] = round((time.perf_counter() - t) * 1000, 1)
                yield _sse({"type": "evaluation", "evaluation": evaluation})

            yield _done(grounded=bool(cited))

        except Exception as exc:
            logger.exception("Stream generation failed")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
