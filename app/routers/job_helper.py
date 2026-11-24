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
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
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
from pydantic import BaseModel, Field

from app.agents.approval_store import get_pending, list_pending, to_legacy
from app.agents.job_helper import files, repository as repo
from app.agents.job_helper import apply, feed, service
from app.agents.job_helper.documents import (
    UnsupportedResumeFile,
    extract_text,
    render,
)
from app.agents.job_helper.jobsearch import configured_providers
from app.agents.job_helper.state import (
    ApplyProfile,
    ApplicationStatus,
    FormField,
    FeedState,
    JOB_MEMORY_INSTRUCTIONS,
    JobListing,
    JobMemoryExtract,
    JobPreferences,
    ResumeContent,
)
from app.agents.job_helper.triggers import FEED_TRIGGER_TYPE
from app.agents.job_helper.workflow import RESUME_APPROVAL
from app.agents.memory_store import extract_and_save
from app.agents.trigger_store import get_trigger, toggle, update_trigger
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


async def _resume_response(resume: dict, fmt: str) -> Response:
    """A stored resume as a file download: the original upload, or a fresh
    render of its structure. `auto` means the original when there is one."""
    stored = resume.get("file") or {}
    if fmt == "auto":
        fmt = "original" if stored.get("path") else "pdf"
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
            logger.exception("resume render failed id=%s fmt=%s", resume.get("id"), fmt)
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
    return await _resume_response(resume, fmt)


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


# ── daily job feed ───────────────────────────────────────────────────────────
# One ranked list across every board (see agents/job_helper/feed.py). Reads are
# free; refresh and import spend model calls on scoring, so they sit on the
# query bucket like /fit does.
FEED_TRIGGER_DEFAULTS = {"name": "daily job feed", "schedule_hour": 8}


@router.get("/preferences")
async def get_preferences(current_user: Annotated[dict, Depends(get_current_user)]):
    """The feed's search preferences. Unsaved ones come back pre-filled from what
    the job helper already knows (`saved: false`) for the user to confirm."""
    user_id = current_user["uid"]
    prefs, saved = await feed.get_preferences(user_id)
    trigger = await get_trigger(user_id, FEED_TRIGGER_TYPE)
    return {
        "status": "done",
        "result": prefs.model_dump(),
        "saved": saved,
        "schedule": {
            "enabled": bool(trigger and trigger.get("enabled", True)),
            "schedule_hour": (trigger or {}).get(
                "schedule_hour", FEED_TRIGGER_DEFAULTS["schedule_hour"]
            ),
            "timezone": (trigger or {}).get("timezone", "UTC"),
        },
    }


@router.put("/preferences")
async def save_preferences(
    body: JobPreferences,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Save preferences. The first save also turns the daily feed on (08:00
    local), since that is what setting them is for; PUT /feed/schedule sets
    the hour and timezone or turns it off."""
    user_id = current_user["uid"]
    await rate_limit.limit_user(
        "job_write", user_id, JOB_WRITE_RATE_LIMIT, JOB_WRITE_RATE_WINDOW
    )
    prefs = await feed.save_preferences(user_id, body)
    if not await get_trigger(user_id, FEED_TRIGGER_TYPE):
        await toggle(user_id, FEED_TRIGGER_TYPE, defaults=FEED_TRIGGER_DEFAULTS)
    return {"status": "done", "result": prefs.model_dump()}


@router.get("/feed")
async def list_feed(
    current_user: Annotated[dict, Depends(get_current_user)],
    state: Literal[FeedState, "all"] = "new",
    limit: int = 50,
):
    """The ranked feed, best match first. `state=filtered` shows what the
    dealbreakers hid and why; `state=all` shows everything."""
    items = await repo.list_feed(
        current_user["uid"],
        None if state == "all" else state,
        max(1, min(limit, 200)),
    )
    return {"status": "done", "result": items}


@router.post("/feed/refresh")
async def refresh_feed(current_user: Annotated[dict, Depends(get_current_user)]):
    """Search every board now instead of waiting for the morning run."""
    user_id = current_user["uid"]
    await rate_limit.limit_user(
        "job_query", user_id, JOB_QUERY_RATE_LIMIT, JOB_QUERY_RATE_WINDOW
    )
    try:
        result = await feed.refresh_feed(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "done", "result": result}


class FeedImportRequest(BaseModel):
    source: str
    listings: List[JobListing]


MAX_IMPORT_LISTINGS = 50


@router.post("/feed/import")
async def import_feed(
    body: FeedImportRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Postings read from a portal page the user has open (the browser
    extension's "Import this page"). Deduped and ranked like the rest."""
    user_id = current_user["uid"]
    if not body.listings:
        raise HTTPException(status_code=400, detail="No listings to import.")
    if len(body.listings) > MAX_IMPORT_LISTINGS:
        raise HTTPException(
            status_code=400,
            detail=f"Import at most {MAX_IMPORT_LISTINGS} listings at a time.",
        )
    await rate_limit.limit_user(
        "job_query", user_id, JOB_QUERY_RATE_LIMIT, JOB_QUERY_RATE_WINDOW
    )
    source = body.source.strip().lower()[:40] or "import"
    result = await feed.import_listings(
        user_id, [listing.model_dump() for listing in body.listings], source
    )
    return {"status": "done", "result": result}


@router.get("/feed/{item_id}")
async def get_feed_item(
    item_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    item = await repo.get_feed_item(current_user["uid"], item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"status": "done", "result": item}


@router.post("/feed/{item_id}/dismiss")
async def dismiss_feed_item(
    item_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Hide a posting. It stays hidden when the boards return it again."""
    item = await repo.update_feed_item(
        current_user["uid"], item_id, {"state": "dismissed"}
    )
    if not item:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"status": "done", "result": item}


@router.post("/feed/{item_id}/track", status_code=201)
async def track_feed_item(
    item_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Start tracking a feed posting as an application (analyses the full
    posting). Tracking the same posting twice returns the same application."""
    user_id = current_user["uid"]
    await rate_limit.limit_user(
        "job_write", user_id, JOB_WRITE_RATE_LIMIT, JOB_WRITE_RATE_WINDOW
    )
    try:
        application = await feed.track_item(user_id, item_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not application:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"status": "done", "result": application}



# ── applying: profile, form answers, the resume to attach ────────────────────
# What the browser extension calls on an application form. The user reviews
# every value and clicks Submit themselves; nothing here submits anything.
class AnswerSnapshot(BaseModel):
    label: str = Field(max_length=500)
    value: Optional[str] = Field(default=None, max_length=5000)


@router.get("/apply-profile")
async def get_apply_profile(current_user: Annotated[dict, Depends(get_current_user)]):
    """The answers every form asks for. Unsaved, it comes back pre-filled from
    the resume (`saved: false`) for the user to check and save."""
    profile, saved, remembered = await apply.get_apply_profile(current_user["uid"])
    return {
        "status": "done",
        "result": profile.model_dump(),
        "saved": saved,
        "remembered": remembered,
    }


@router.put("/apply-profile")
async def save_apply_profile(
    body: ApplyProfile,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    user_id = current_user["uid"]
    await rate_limit.limit_user(
        "job_write", user_id, JOB_WRITE_RATE_LIMIT, JOB_WRITE_RATE_WINDOW
    )
    await repo.save_apply_profile(user_id, body.model_dump())
    return {"status": "done", "result": body.model_dump()}


class RememberRequest(BaseModel):
    answers: List[AnswerSnapshot] = Field(max_length=100)


@router.post("/apply-profile/answers")
async def remember_answers(
    body: RememberRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Remember answers to reuse on the next form that asks the same question.
    Personal and consent questions are never stored (`skipped`)."""
    user_id = current_user["uid"]
    await rate_limit.limit_user(
        "job_write", user_id, JOB_WRITE_RATE_LIMIT, JOB_WRITE_RATE_WINDOW
    )
    try:
        result = await apply.remember(user_id, [a.model_dump() for a in body.answers])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "done", "result": result}


@router.delete("/apply-profile/answers/{answer_key}")
async def forget_answer(
    answer_key: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    # Normalised again so a crafted key can never address another field path.
    norm = apply.norm_label(answer_key)
    if not norm or not await repo.forget_answer(current_user["uid"], norm):
        raise HTTPException(status_code=404, detail="No remembered answer by that name.")
    return {"status": "done", "deleted": norm}


MAX_FORM_FIELDS = 100


class FillRequest(BaseModel):
    fields: List[FormField]
    application_id: Optional[str] = None
    # The posting's text as the extension read it off the page — used when the
    # form belongs to a job that is not tracked yet.
    job_text: Optional[str] = Field(default=None, max_length=30_000)
    page_url: Optional[str] = Field(default=None, max_length=2000)


@router.post("/fill")
async def fill_form(
    body: FillRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Values for an application form's fields. Every field comes back either
    with a value (and where it came from) or `needs_you` with a reason."""
    user_id = current_user["uid"]
    if not body.fields:
        raise HTTPException(status_code=400, detail="No form fields were sent.")
    if len(body.fields) > MAX_FORM_FIELDS:
        raise HTTPException(
            status_code=400, detail=f"At most {MAX_FORM_FIELDS} fields per request."
        )
    if len({f.key for f in body.fields}) != len(body.fields):
        raise HTTPException(status_code=400, detail="Field keys must be unique.")

    application = None
    if body.application_id:
        application = await repo.get_application(user_id, body.application_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found.")
    await rate_limit.limit_user(
        "job_query", user_id, JOB_QUERY_RATE_LIMIT, JOB_QUERY_RATE_WINDOW
    )
    result = await apply.fill_form(user_id, body.fields, application, body.job_text)
    return {"status": "done", "result": result}


@router.get("/resume-file")
async def resume_file(
    current_user: Annotated[dict, Depends(get_current_user)],
    application_id: Optional[str] = None,
    fmt: Literal["auto", "original", "pdf", "docx"] = "auto",
):
    """The file to attach to an application form: the approved tailored version
    for that application when there is one (rendered), else the user's own
    uploaded resume, byte for byte."""
    user_id = current_user["uid"]
    application = None
    if application_id:
        application = await repo.get_application(user_id, application_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found.")
    resume = await apply.resume_for(user_id, application)
    if not resume:
        raise HTTPException(status_code=404, detail="No resume on file yet.")
    return await _resume_response(resume, fmt)


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
    # Sent by the browser extension when the user marks a form as submitted:
    # where they applied and what went out, so "what did I tell them my notice
    # period was?" has an answer when the recruiter calls.
    applied_via: Optional[Literal["portal", "company_site", "other"]] = None
    apply_url: Optional[str] = Field(default=None, max_length=2000)
    resume_id_sent: Optional[str] = None
    answers_snapshot: Optional[List[AnswerSnapshot]] = Field(default=None, max_length=150)


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
    if body.resume_id_sent and not await repo.verify_resume_ownership(
        body.resume_id_sent, user_id
    ):
        raise HTTPException(status_code=403, detail="You do not have access to this resume.")
    submission = {
        "applied_via": body.applied_via,
        "apply_url": body.apply_url,
        "resume_id_sent": body.resume_id_sent,
        "answers_snapshot": (
            [a.model_dump() for a in body.answers_snapshot]
            if body.answers_snapshot is not None
            else None
        ),
    }
    if any(v is not None for v in submission.values()):
        updated = await repo.update_application(user_id, application_id, submission)
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



class FeedScheduleRequest(BaseModel):
    """When the daily feed runs. Hour resolution only: the sweep is hourly."""

    enabled: Optional[bool] = None
    schedule_hour: Optional[int] = Field(default=None, ge=0, le=23)
    timezone: Optional[str] = Field(default=None, description="IANA, e.g. Asia/Kolkata")


@router.put("/feed/schedule")
async def set_feed_schedule(
    body: FeedScheduleRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Turn the daily feed on/off and set its local hour and timezone."""
    if body.timezone:
        try:
            ZoneInfo(body.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise HTTPException(status_code=400, detail="Unknown timezone.")
    trigger = await update_trigger(
        current_user["uid"],
        FEED_TRIGGER_TYPE,
        body.model_dump(exclude_none=True),
        defaults=FEED_TRIGGER_DEFAULTS,
    )
    return {
        "status": "done",
        "result": {
            "enabled": trigger.get("enabled", True),
            "schedule_hour": trigger.get("schedule_hour"),
            "timezone": trigger.get("timezone", "UTC"),
        },
    }
