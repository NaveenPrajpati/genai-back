"""HTTP routes for the job-helper agent.

Agent logic lives in app.agents.job_helper; this file is transport only —
auth, ownership, multipart, and the two byte-returning download routes.

WHY FILES ARE NOT PART OF THE CHAT
──────────────────────────────────
Uploading a resume and downloading one are plain REST, not agent turns. A file
upload has nothing for a model to decide, and a browser download needs a byte
stream with a Content-Disposition header, which an SSE chat response cannot be.
The agent produces a resume *version*; these routes move bytes in and out of it.

THREE THINGS A DOWNLOAD CAN MEAN
────────────────────────────────
  original — the exact file the user uploaded, byte for byte. Their layout,
             their typography. This is what they send to employers, and nothing
             we generate replaces it.
  docx/pdf — the stored structure rendered fresh. The right answer for a resume
             that was typed in (there is no original) and for a tailored
             version (which never had one).

Serving the original through this route rather than handing out the storage URL
keeps one authenticated door on the file; the signed links that go to the client
are short-lived for the same reason (see agents/job_helper/files.py).
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
from typing import Annotated, List, Literal, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from app.agents.approval_store import get_pending, list_pending, to_legacy
from app.agents.job_helper import files, repository as repo
from app.agents.job_helper import service
from app.agents.job_helper.documents import (
    UnsupportedResumeFile,
    extract_text,
    render,
)
from app.agents.job_helper.jobsearch import configured_providers
from app.agents.job_helper.state import (
    ApplicationStatus,
    JOB_MEMORY_INSTRUCTIONS,
    JobMemoryExtract,
    ResumeContent,
)
from app.agents.job_helper.workflow import RESUME_APPROVAL
from app.agents.memory_store import extract_and_save
from app.agents.trigger_store import toggle
from app.core.config import (
    JOB_QUERY_RATE_LIMIT,
    JOB_QUERY_RATE_WINDOW,
    JOB_WRITE_RATE_LIMIT,
    JOB_WRITE_RATE_WINDOW,
)
from app.dependencies import get_current_user
from app.services import rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/job-helper",
    tags=["job-helper"],
    responses={404: {"description": "Not found"}},
)

# A resume is a handful of pages. The cap bounds both the upload and the
# parsing prompt behind it; anything larger is a different kind of document.
MAX_RESUME_BYTES = 5 * 1024 * 1024

EXCLUDED_USER_FIELDS = {"_id", "expires_at", "password_hash"}


def _user_payload(current_user: dict) -> dict:
    return {k: v for k, v in current_user.items() if k not in EXCLUDED_USER_FIELDS}


def _sse(event: dict) -> str:
    """Serialize one event as a Server-Sent Events frame."""
    return f"data: {json.dumps(event, default=str)}\n\n"


# ── chat ─────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    text: str
    application_id: Optional[str] = None
    resume_id: Optional[str] = None
    thread_id: Optional[str] = None


async def _guard_refs(user_id: str, body: QueryRequest) -> None:
    """Fail fast on an id the caller does not own, before any model spend."""
    if body.application_id and not await repo.verify_application_ownership(
        body.application_id, user_id
    ):
        raise HTTPException(
            status_code=403, detail="You do not have access to this application."
        )
    if body.resume_id and not await repo.verify_resume_ownership(
        body.resume_id, user_id
    ):
        raise HTTPException(
            status_code=403, detail="You do not have access to this resume."
        )


def _agent_input(body: QueryRequest, current_user: dict, thread_id: str) -> dict:
    return {
        "query": body.text,
        "user_id": current_user["uid"],
        "thread_id": thread_id,
        "application_id": body.application_id,
        "resume_id": body.resume_id,
        "current_user": _user_payload(current_user),
    }


@router.post("/query")
async def ask(
    body: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    user_id = current_user["uid"]
    await rate_limit.limit_user(
        "job_query", user_id, JOB_QUERY_RATE_LIMIT, JOB_QUERY_RATE_WINDOW
    )
    await _guard_refs(user_id, body)

    agent = request.app.state.job_agent
    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(_agent_input(body, current_user, thread_id), config)

    # Fire-and-forget: learn durable job-search facts after the response is sent.
    background_tasks.add_task(
        extract_and_save,
        user_id,
        body.text,
        JobMemoryExtract,
        JOB_MEMORY_INSTRUCTIONS,
        result.get("memory"),
    )

    if "__interrupt__" in result:
        return {
            "status": "needs_approval",
            "thread_id": thread_id,  # the client sends this back to /approve
            "proposal": result["__interrupt__"][0].value,
        }
    return {"status": "done", "thread_id": thread_id, "result": result}


@router.post("/query/stream")
async def ask_stream(
    body: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Streaming counterpart of /query. The tool loop runs inside one node, so
    there are no tokens to stream mid-node; each completed node is emitted as a
    `step` event for live progress, then the final state as `done` (or
    `needs_approval` when a tailored resume is waiting on the user)."""
    user_id = current_user["uid"]
    await rate_limit.limit_user(
        "job_query", user_id, JOB_QUERY_RATE_LIMIT, JOB_QUERY_RATE_WINDOW
    )
    await _guard_refs(user_id, body)

    agent = request.app.state.job_agent
    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    inputs = _agent_input(body, current_user, thread_id)

    async def generate():
        try:
            yield _sse({"type": "thread", "thread_id": thread_id})

            async for update in agent.astream(
                inputs, config=config, stream_mode="updates"
            ):
                for node in update:
                    yield _sse({"type": "step", "node": node})

            snapshot = await agent.aget_state(config)
            values = snapshot.values if snapshot else {}
            interrupts = snapshot.interrupts if snapshot else None
            if snapshot and snapshot.next and interrupts:
                yield _sse(
                    {
                        "type": "needs_approval",
                        "thread_id": thread_id,
                        "proposal": interrupts[0].value,
                    }
                )
                return

            background_tasks.add_task(
                extract_and_save,
                user_id,
                body.text,
                JobMemoryExtract,
                JOB_MEMORY_INSTRUCTIONS,
                values.get("memory"),
            )
            yield _sse({"type": "done", "result": values})
        except Exception as exc:
            logger.exception("job helper stream failed")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["approved", "rejected"]


@router.post("/approve")
async def approve(
    body: ApproveRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Resolve a pending tailored-resume approval and let the run finish."""
    agent = request.app.state.job_agent
    config = {"configurable": {"thread_id": body.thread_id}}

    # The approval must belong to the caller, or a guessed thread_id would let
    # one user approve another's resume.
    try:
        approval = await get_pending(body.thread_id, [RESUME_APPROVAL])
    except Exception as e:
        logger.error("job approval lookup error: %s", e)
        approval = None

    if not approval:
        raise HTTPException(
            status_code=404, detail="No pending approval for this thread."
        )
    if approval["user_id"] != current_user["uid"]:
        raise HTTPException(
            status_code=403, detail="You do not have access to this approval."
        )

    snapshot = await agent.aget_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=404,
            detail="No pending approval for this thread. The server may have "
            "restarted — please ask for the tailored resume again.",
        )

    result = await agent.ainvoke(Command(resume=body.decision), config=config)
    return {"status": "done", "result": result}


@router.get("/approve")
async def list_approvals(current_user: Annotated[dict, Depends(get_current_user)]):
    docs = await list_pending(current_user["uid"], [RESUME_APPROVAL])
    return {"status": "done", "result": [to_legacy(d) for d in docs]}


# ── resumes ──────────────────────────────────────────────────────────────────
@router.post("/resumes/upload", status_code=201)
async def upload_resume(
    current_user: Annotated[dict, Depends(get_current_user)],
    file: Annotated[UploadFile, File(description=".docx, .pdf or .txt")],
    label: Annotated[Optional[str], Form()] = None,
):
    """Upload a resume: keep the file exactly as it is, and parse its text into
    structure the agent can reason over.

    Both, not either. The structure is what scores a fit and drives a rewrite;
    the file is what the user actually sends to an employer, carrying a layout
    nothing we generate can reproduce. Storing the original is best-effort — if
    the bucket is unreachable the resume is still parsed and usable, and the
    response says the file was not kept.
    """
    user_id = current_user["uid"]
    await rate_limit.limit_user(
        "job_write", user_id, JOB_WRITE_RATE_LIMIT, JOB_WRITE_RATE_WINDOW
    )

    suffix = os.path.splitext(file.filename or "")[1].lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp, length=1024 * 1024)
            tmp_path = tmp.name
        if os.path.getsize(tmp_path) > MAX_RESUME_BYTES:
            raise HTTPException(status_code=413, detail="Resume exceeds the 5 MB limit")

        # Both readers are synchronous and CPU-bound; off the event loop they go.
        text = await asyncio.to_thread(extract_text, tmp_path, file.filename or "")
        with open(tmp_path, "rb") as fh:
            payload = fh.read()
    except UnsupportedResumeFile as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # The Supabase client is synchronous, so it goes to a thread like the RAG
    # routes' ingestion-log writes.
    file_meta = await asyncio.to_thread(
        files.store, user_id, payload, file.filename or "resume"
    )
    resume = await service.store_uploaded_resume(
        user_id, text, filename=file.filename, label=label, file_meta=file_meta
    )
    return {
        "status": "done",
        "result": files.with_link(resume),
        "original_stored": bool(file_meta),
        **(
            {}
            if file_meta
            else {
                "warning": "Your resume was read and saved, but the original "
                "file could not be stored — re-upload it later to keep a copy."
            }
        ),
    }


class ManualResumeRequest(BaseModel):
    """A resume the user types in instead of uploading."""

    content: ResumeContent
    label: Optional[str] = None


@router.post("/resumes", status_code=201)
async def create_resume(
    body: ManualResumeRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Save a resume the user typed in. Same schema as a parsed upload, so
    everything downstream works identically — this is the path for someone who
    has no resume file yet, and the one where the generated .docx/.pdf is the
    point rather than a substitute for their own document."""
    resume = await service.save_manual_resume(
        current_user["uid"], body.content.model_dump(), body.label
    )
    return {"status": "done", "result": resume}


class ResumeContentPatch(BaseModel):
    content: ResumeContent
    label: Optional[str] = None


@router.patch("/resumes/{resume_id}")
async def update_resume(
    resume_id: str,
    body: ResumeContentPatch,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Edit a stored resume's structure — fixing what a parse got wrong, or
    keeping it current. Sections sent replace their old value; sections left at
    their default are untouched. The uploaded original is never modified."""
    user_id = current_user["uid"]
    changes = body.content.model_dump(exclude_defaults=True)
    if body.label:
        await repo.update_resume(user_id, resume_id, {"label": body.label})
    updated = await service.update_resume_content(user_id, resume_id, changes)
    if not updated:
        raise HTTPException(status_code=404, detail="Resume not found.")
    return {"status": "done", "result": files.with_link(updated)}


@router.get("/resumes")
async def list_resumes(
    current_user: Annotated[dict, Depends(get_current_user)],
    kind: Optional[Literal["base", "tailored"]] = None,
    application_id: Optional[str] = None,
):
    resumes = await repo.list_resumes(current_user["uid"], kind, application_id)
    return {"status": "done", "result": [files.with_link(r) for r in resumes]}


@router.get("/resumes/{resume_id}")
async def get_resume(
    resume_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    resume = await repo.get_resume(current_user["uid"], resume_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found.")
    return {"status": "done", "result": files.with_link(resume)}


@router.get("/resumes/{resume_id}/download")
async def download_resume(
    resume_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    fmt: Literal["original", "pdf", "docx"] = "original",
):
    """Download a stored resume. See THREE THINGS A DOWNLOAD CAN MEAN above.

    `original` is the default, and falls back to a render only when there is no
    stored file — a typed-in resume or a tailored version. The renders are done
    on demand rather than at save time: both formats come from the same
    structure, so storing them would mean two blobs to keep in step with every
    edit, for a render that takes milliseconds.
    """
    resume = await repo.get_resume(current_user["uid"], resume_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found.")

    stored = resume.get("file") or {}
    if fmt == "original":
        if not stored.get("path"):
            raise HTTPException(
                status_code=404,
                detail="This resume has no uploaded file — download it as pdf "
                "or docx instead.",
            )
        payload = await asyncio.to_thread(files.fetch, stored["path"])
        if payload is None:
            raise HTTPException(
                status_code=502, detail="The stored resume file could not be read."
            )
        media_type = stored.get("content_type") or "application/octet-stream"
        filename = stored.get("filename") or "resume"
    else:
        try:
            payload, media_type, filename = await asyncio.to_thread(
                render, resume.get("content") or {}, fmt, resume.get("label") or "Resume"
            )
        except Exception as e:
            logger.exception("resume render failed id=%s fmt=%s", resume_id, fmt)
            raise HTTPException(
                status_code=500, detail=f"Could not render the resume: {e}"
            )

    return Response(
        content=payload,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            # The filename is derived from user-supplied content, so let the
            # browser see the length rather than guess at a chunked body.
            "Content-Length": str(len(payload)),
        },
    )


@router.delete("/resumes/{resume_id}")
async def delete_resume(
    resume_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Delete a resume and the file stored with it."""
    deleted = await repo.delete_resume(current_user["uid"], resume_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Resume not found.")
    stored = deleted.get("file") or {}
    if stored.get("path"):
        await asyncio.to_thread(files.remove, stored["path"])
    return {"status": "done", "deleted": resume_id}


# ── search ───────────────────────────────────────────────────────────────────
@router.get("/search")
async def search_jobs(
    current_user: Annotated[dict, Depends(get_current_user)],
    q: str,
    location: Optional[str] = None,
    remote: bool = False,
    limit: int = 8,
):
    """Browse job boards directly, without spending a chat turn.

    Rate-limited on the write bucket: it costs no model call, but it is an
    outbound call to a third-party board on every request, and some of them
    price by the month.
    """
    await rate_limit.limit_user(
        "job_write", current_user["uid"], JOB_WRITE_RATE_LIMIT, JOB_WRITE_RATE_WINDOW
    )
    result = await service.find_jobs(q, location, remote, limit)
    return {"status": "done", **result}


@router.get("/search/providers")
async def search_providers(current_user: Annotated[dict, Depends(get_current_user)]):
    """Which job boards are usable right now, best first. The keyless ones are
    always present, so this never comes back empty."""
    return {
        "status": "done",
        "result": {
            "available": configured_providers(),
            "active": os.getenv("JOB_SEARCH_PROVIDER", "auto"),
        },
    }


# ── applications ─────────────────────────────────────────────────────────────
class TrackRequest(BaseModel):
    jd_text: Optional[str] = None
    source_url: Optional[str] = None
    status: ApplicationStatus = "saved"


@router.post("/applications", status_code=201)
async def track_application(
    body: TrackRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Track a posting from pasted text or a link."""
    user_id = current_user["uid"]
    await rate_limit.limit_user(
        "job_write", user_id, JOB_WRITE_RATE_LIMIT, JOB_WRITE_RATE_WINDOW
    )
    try:
        application = await service.track_application(
            user_id, body.jd_text, body.source_url, body.status
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "done", "result": application}


@router.get("/applications")
async def list_applications(
    current_user: Annotated[dict, Depends(get_current_user)],
    status: Optional[ApplicationStatus] = None,
):
    applications = await repo.list_applications(current_user["uid"], status)
    return {"status": "done", "result": applications}


@router.get("/applications/{application_id}")
async def get_application(
    application_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    application = await repo.get_application(current_user["uid"], application_id)
    if not application:
        raise HTTPException(status_code=404, detail="Application not found.")
    return {"status": "done", "result": application}


class UpdateApplicationRequest(BaseModel):
    status: Optional[ApplicationStatus] = None
    note: Optional[str] = None
    deadline: Optional[str] = None


@router.patch("/applications/{application_id}")
async def update_application(
    application_id: str,
    body: UpdateApplicationRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    user_id = current_user["uid"]
    if not await repo.verify_application_ownership(application_id, user_id):
        raise HTTPException(status_code=404, detail="Application not found.")

    updated = None
    if body.status:
        updated = await service.set_status(user_id, application_id, body.status)
    if body.deadline:
        updated = await repo.update_application(
            user_id, application_id, {"deadline": body.deadline}
        )
    if body.note:
        updated = await repo.add_application_note(user_id, application_id, body.note)
    if updated is None:
        updated = await repo.get_application(user_id, application_id)
    return {"status": "done", "result": updated}


@router.delete("/applications/{application_id}")
async def delete_application(
    application_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    if not await repo.delete_application(current_user["uid"], application_id):
        raise HTTPException(status_code=404, detail="Application not found.")
    return {"status": "done", "deleted": application_id}


class FitRequest(BaseModel):
    resume_id: Optional[str] = None


@router.post("/applications/{application_id}/fit")
async def assess_fit(
    application_id: str,
    body: FitRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Score a resume against one posting, without going through the chat."""
    user_id = current_user["uid"]
    await rate_limit.limit_user(
        "job_query", user_id, JOB_QUERY_RATE_LIMIT, JOB_QUERY_RATE_WINDOW
    )
    try:
        gap = await service.assess_fit(user_id, application_id, body.resume_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "done", "result": gap}


class FollowupRequest(BaseModel):
    titles: List[str]


@router.post("/applications/{application_id}/followups", status_code=201)
async def add_followups(
    application_id: str,
    body: FollowupRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Push next steps onto the user's personal-assistant task list."""
    user_id = current_user["uid"]
    if not await repo.verify_application_ownership(application_id, user_id):
        raise HTTPException(status_code=404, detail="Application not found.")
    created = await service.push_followups(user_id, application_id, body.titles)
    return {"status": "done", "result": created}


# ── schedule ─────────────────────────────────────────────────────────────────
@router.post("/toggle-trigger")
async def toggle_trigger(current_user: Annotated[dict, Depends(get_current_user)]):
    """Opt in or out of the follow-up nudge. Default cadence: 09:00 daily in the
    user's timezone; the sweep only raises anything when something is stale."""
    enabled = await toggle(
        current_user["uid"],
        "job_followup",
        defaults={"name": "job application follow-ups", "schedule_hour": 9},
    )
    return {"status": "done", "enabled": enabled}
