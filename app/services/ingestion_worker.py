"""
ingestion_worker

The BACKGROUND TASK that runs the full ingest pipeline for one source:

    load (Step 1) → chunk (Step 2) → embed + upsert (Steps 3) → log status

It runs off the request thread (via FastAPI BackgroundTasks) so the HTTP call
returns immediately with a job_id, and blocking library calls are pushed to a
worker thread with `asyncio.to_thread` so they don't block the event loop.

The in-memory `INGESTION_JOBS` dict gives instant status checks without a DB
round-trip. NOTE: it's per-process — fine for a single instance, but if you run
multiple workers/replicas, rely on the Supabase log (the source of truth) for
status instead, or move job state to Redis.

PRODUCTION UPGRADE: for real scale, replace BackgroundTasks with a proper task
queue (Celery / RQ / Arq / Dramatiq). You get retries, concurrency limits,
visibility, and survival across restarts — none of which BackgroundTasks offers.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from app.utils.chunking import get_splitter, SplitStrategy
from app.services.retrieval import hybrid_add_texts
from app.services import storage, cache

logger = logging.getLogger(__name__)

# Per-process job tracker for fast status polling.
INGESTION_JOBS: dict[str, dict] = {}

_splitter = get_splitter(SplitStrategy.RECURSIVE)


async def run_ingestion(
    job_id: str,
    loader_fn: Callable,
    display_source: str,
    file_type: str,
    tmp_path: Optional[str],
    user_id: str,
) -> None:
    try:
        INGESTION_JOBS[job_id]["status"] = "processing"
        await asyncio.to_thread(
            storage.update_ingestion_log, job_id, {"status": "processing"}
        )

        # Step 1: load  ──────────────────────────────────────────────────────
        documents = await asyncio.to_thread(loader_fn)

        # Step 2: chunk ──────────────────────────────────────────────────────
        chunks = await asyncio.to_thread(_splitter.split_documents, documents)

        ingested_at = datetime.now(timezone.utc).isoformat()
        texts = [c.page_content for c in chunks]
        total = len(chunks)
        metadatas = [
            {
                "doc_id": job_id,
                "user_id": user_id,
                "source": display_source,
                "file_type": file_type,
                "ingested_at": ingested_at,
                "chunk_index": i,
                "total_chunks": total,
            }
            for i in range(total)
        ]

        # Step 3: embed (dense + sparse) and upsert into Pinecone ─────────────
        await asyncio.to_thread(hybrid_add_texts, texts, metadatas)

        # The new vectors are now searchable — drop this user's stale cached
        # answers so the next question reflects the freshly-ingested content
        # instead of replaying a pre-ingest reply for up to CACHE_TTL_SECONDS.
        await cache.invalidate_user(user_id)

        completed_at = datetime.now(timezone.utc).isoformat()
        INGESTION_JOBS[job_id].update(
            {
                "status": "completed",
                "chunks": total,
                "source": display_source,
                "completed_at": completed_at,
            }
        )
        await asyncio.to_thread(
            storage.update_ingestion_log,
            job_id,
            {"status": "completed", "chunks": total, "completed_at": completed_at},
        )

    except Exception as exc:
        logger.exception("Ingestion failed for job %s", job_id)
        INGESTION_JOBS[job_id].update({"status": "failed", "error": str(exc)})
        await asyncio.to_thread(
            storage.update_ingestion_log,
            job_id,
            {"status": "failed", "error": str(exc)},
        )

    finally:
        # Always clean up the temp file, success or failure.
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
