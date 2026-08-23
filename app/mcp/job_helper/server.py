"""The job helper, published as a Model Context Protocol server.

Everything here is a thin wrapper over `agents.job_helper.service` — the MCP
layer owns protocol concerns (tool schemas, identity, ownership) and nothing
else, so the graph and the protocol can never drift apart.

──────────────────────────────────────────────────────────────────────────────
IDENTITY
──────────────────────────────────────────────────────────────────────────────
Every tool acts on one user's applications and resumes, but `user_id` is
deliberately NOT a tool argument: the calling model would have to supply it, and
a model that can name the tenant it reads is a model that can hallucinate (or be
talked into) someone else's. Instead identity travels out-of-band, the way a
bearer token does:

  • streamable-HTTP  → `X-User-Id` request header, set by the client per user
  • stdio            → `MCP_USER_ID` env var (single-user: Claude Desktop, the
                       MCP Inspector)

Every repository read is scoped by that id, so an `application_id` guessed by
any client resolves to nothing rather than to someone else's row.

──────────────────────────────────────────────────────────────────────────────
PROPOSE / SAVE SPLIT
──────────────────────────────────────────────────────────────────────────────
`propose_tailored_resume` rewrites without persisting; `save_tailored_resume`
persists an already-approved rewrite. MCP is stateless request/response — a tool
call has nowhere to pause for a human — so the approval step lives in the caller
(the job-helper graph raises a LangGraph `interrupt()` between the two calls).
The gate is not ceremony here: a tailored resume is what the user sends to an
employer under their own name, so nothing reaches their document list until they
have read the changelog and said yes.

──────────────────────────────────────────────────────────────────────────────
RUNNING IT
──────────────────────────────────────────────────────────────────────────────
Mounted at /mcp/jobs by app.main. Standalone, for an external MCP client:

    MCP_USER_ID=<uid> python -m app.mcp.job_helper          # stdio
    python -m app.mcp.job_helper --http                     # :8101/mcp
"""

import logging
import os
from typing import Annotated, List, Literal, Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from app.agents.job_helper import repository as repo
from app.agents.job_helper import service
from app.agents.job_helper.jobsearch import configured_providers
from app.agents.job_helper.state import (
    ApplicationStatus,
    ResumeContent,
    ResumeEdit,
)
from app.agents.memory_store import get_profile

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Tools for one user's job search: find openings, track the postings they are
applying to, compare their resume against one, tailor it, draft a cover letter,
prepare for the interview, and keep each application's status current.

Finding and tracking are different steps. search_jobs browses a job board and
saves nothing; track_job_posting analyses ONE posting and starts tracking it.
Everything else takes the application_id that track_job_posting returns. Pasted
job-description text is the reliable path — many job sites block link fetching.

The user's resume exists in two forms and both matter. The structure is what
these tools reason over; their uploaded file is what they actually send to
employers, and it is never modified. Tailoring therefore returns `edits` —
before/after pairs they apply inside their own document so its layout survives.
Show them the edits, not just a claim that it was rewritten.

Tailoring is two steps on purpose: call propose_tailored_resume, show the user
what changed, and only call save_tailored_resume once they have approved it.
Never call save_tailored_resume with a rewrite the user has not seen.

If they have no resume yet, save_resume_details takes one they type out; there is
no requirement to upload a file.

Never claim the user has a skill their resume does not evidence. When a posting
needs something they lack, say so and offer start_roadmap_for_gap.
"""

mcp = FastMCP("job-helper", instructions=INSTRUCTIONS, stateless_http=True)


# ── identity ─────────────────────────────────────────────────────────────────
def _caller_id(ctx: Context) -> str:
    """The user this call acts for. See IDENTITY in the module docstring."""
    request = getattr(ctx.request_context, "request", None)
    if request is not None:
        header = request.headers.get("x-user-id")
        if header:
            return header
    env_uid = os.getenv("MCP_USER_ID")
    if env_uid:
        return env_uid
    raise ToolError(
        "No user identity on this MCP connection: send an X-User-Id header "
        "(HTTP) or set MCP_USER_ID (stdio)."
    )


async def _require_application(user_id: str, application_id: str) -> dict:
    application = await repo.get_application(user_id, application_id)
    if not application:
        raise ToolError(
            f"Application {application_id} does not exist or does not belong to you."
        )
    return application


# ── read ─────────────────────────────────────────────────────────────────────
@mcp.tool()
async def list_applications(
    ctx: Context,
    status: Annotated[
        Optional[ApplicationStatus],
        Field(description="Only applications in this stage; omit for all"),
    ] = None,
) -> list[dict]:
    """List the jobs the user is tracking, most recently touched first. Each
    entry has an `id` (pass it as application_id to other tools), the company and
    role, its status, and the cached fit score if one has been computed. Call this
    first when the user refers to a job without naming an id."""
    return await repo.list_applications(_caller_id(ctx), status)


@mcp.tool()
async def get_application(
    ctx: Context,
    application_id: Annotated[str, Field(description="Id from list_applications")],
) -> dict:
    """Read one tracked application in full: the posting's structured
    requirements, the fit assessment if one has run, status, deadline and notes."""
    user_id = _caller_id(ctx)
    application = await _require_application(user_id, application_id)
    # The raw posting is large and the structured analysis says the same thing;
    # sending both would spend most of the tool budget on duplicated prose.
    application.pop("jd_text", None)
    return application


@mcp.tool()
async def list_resumes(
    ctx: Context,
    kind: Annotated[
        Optional[Literal["base", "tailored"]],
        Field(description="'base' = uploaded originals, 'tailored' = generated"),
    ] = None,
) -> list[dict]:
    """List the user's resumes, newest first — their uploaded base resumes and
    every version tailored to a posting. Returns labels and ids, not content."""
    return await repo.list_resumes(_caller_id(ctx), kind)


@mcp.tool()
async def get_job_search_profile(ctx: Context) -> dict:
    """What is known about this user's search — the roles and locations they are
    targeting, current title, years of experience, work authorization, salary
    expectation, and whether they want remote, hybrid or onsite. Also says
    whether they have a resume on file and how it got there. Consult this before
    recommending roles or writing about what they want."""
    user_id = _caller_id(ctx)
    memory = await get_profile(user_id)
    resume = await repo.latest_base_resume(user_id)
    return {
        "target_roles": memory.get("target_roles"),
        "target_locations": memory.get("target_locations"),
        "current_title": memory.get("current_title"),
        "years_experience": memory.get("years_experience"),
        "work_authorization": memory.get("work_authorization"),
        "salary_expectation": memory.get("salary_expectation"),
        "job_search_status": memory.get("job_search_status"),
        "preferred_work_mode": memory.get("preferred_work_mode"),
        "has_base_resume": bool(resume),
        "base_resume_id": (resume or {}).get("id"),
        "base_resume_label": (resume or {}).get("label"),
        # "upload" means their own file is on record and must never be presented
        # as something we rewrote; "manual" means they typed it in.
        "base_resume_source": (resume or {}).get("source"),
        "original_file_stored": bool((resume or {}).get("file")),
        "job_search_providers": configured_providers(),
    }


@mcp.tool()
async def get_resume_details(
    ctx: Context,
    resume_id: Annotated[
        Optional[str], Field(description="Defaults to the latest base resume")
    ] = None,
) -> dict:
    """Read a resume's full structured content — contact, headline, summary,
    skills, experience with its bullets, projects, education, certifications.
    Use it before answering anything about what the user has actually done."""
    user_id = _caller_id(ctx)
    resume = await service.resolve_resume(user_id, resume_id)
    if not resume:
        raise ToolError(
            "No resume on file yet. The user can upload one, or you can record "
            "one they describe with save_resume_details."
        )
    return {
        "resume_id": resume["id"],
        "label": resume.get("label"),
        "source": resume.get("source"),
        "has_original_file": bool(resume.get("file")),
        "content": resume.get("content") or {},
    }


# ── find ─────────────────────────────────────────────────────────────────────
@mcp.tool()
async def search_jobs(
    ctx: Context,
    query: Annotated[
        str, Field(description="Role and skills, e.g. 'python backend engineer'")
    ],
    location: Annotated[
        Optional[str], Field(description="City or region; omit for anywhere")
    ] = None,
    remote: Annotated[bool, Field(description="Remote roles only")] = False,
    limit: Annotated[int, Field(description="How many listings, 1-20", ge=1, le=20)] = 8,
) -> dict:
    """Search job boards for openings. Saves nothing — this is browsing.

    Returns `listings` (title, company, location, url, description) plus which
    provider answered and a `note` about its coverage, which you should pass on
    when it limits the answer. When the user picks one, call track_job_posting
    with its description text to analyse and start tracking it.

    Check get_job_search_profile first so the query reflects the roles and
    locations they actually want."""
    return await service.find_jobs(query, location, remote, limit)


# ── track ────────────────────────────────────────────────────────────────────
@mcp.tool()
async def track_job_posting(
    ctx: Context,
    jd_text: Annotated[
        Optional[str],
        Field(description="The job description, pasted. The reliable path."),
    ] = None,
    source_url: Annotated[
        Optional[str],
        Field(
            description="Link to the posting. Tried when no text is given, but "
            "LinkedIn, Indeed and most large portals block it — fall back to text."
        ),
    ] = None,
) -> dict:
    """Analyse a job posting and start tracking it. Extracts the company, role,
    required and preferred skills, and the keywords a screen would look for.
    Returns the new application including its `id`."""
    try:
        return await service.track_application(
            _caller_id(ctx), jd_text=jd_text, source_url=source_url
        )
    except ValueError as e:
        raise ToolError(str(e))


@mcp.tool()
async def update_application_status(
    ctx: Context,
    application_id: str,
    status: Annotated[
        ApplicationStatus,
        Field(description="saved, applied, interviewing, offer, rejected, withdrawn"),
    ],
    note: Annotated[
        Optional[str], Field(description="What happened, in the user's words")
    ] = None,
) -> dict:
    """Move an application to a new stage. Moving to `applied` stamps the date,
    which is what the follow-up reminder counts from."""
    user_id = _caller_id(ctx)
    await _require_application(user_id, application_id)
    updated = await service.set_status(user_id, application_id, status, note)
    return {"status": (updated or {}).get("status"), "application_id": application_id}


@mcp.tool()
async def add_application_note(
    ctx: Context,
    application_id: str,
    note: Annotated[str, Field(description="Recruiter reply, interview outcome …")],
) -> dict:
    """Record something that happened on an application, dated."""
    user_id = _caller_id(ctx)
    await _require_application(user_id, application_id)
    updated = await repo.add_application_note(user_id, application_id, note)
    return {"notes": len(updated.get("notes") or []), "application_id": application_id}


# ── analyse ──────────────────────────────────────────────────────────────────
@mcp.tool()
async def assess_resume_fit(
    ctx: Context,
    application_id: str,
    resume_id: Annotated[
        Optional[str], Field(description="Defaults to the latest base resume")
    ] = None,
) -> dict:
    """Compare a resume against a tracked posting: which required skills it
    evidences, which are only adjacent, which are missing, a 0-100 fit score, and
    the missing skills worth learning first. The result is cached on the
    application, and tailoring uses it to know what it must not claim."""
    try:
        return await service.assess_fit(_caller_id(ctx), application_id, resume_id)
    except ValueError as e:
        raise ToolError(str(e))


@mcp.tool()
async def propose_tailored_resume(
    ctx: Context,
    application_id: str,
    resume_id: Annotated[
        Optional[str], Field(description="Defaults to the latest base resume")
    ] = None,
    instructions: Annotated[
        Optional[str], Field(description="Anything else the user asked for")
    ] = None,
) -> dict:
    """Rewrite a resume for one posting — reordered and re-worded to speak to it,
    never adding experience the original does not contain. NOTHING IS SAVED:
    this returns a proposal to show the user, with a `changes` list naming every
    edit. Pass the whole result to save_tailored_resume once they approve."""
    try:
        return await service.tailor_resume(
            _caller_id(ctx), application_id, resume_id, instructions
        )
    except ValueError as e:
        raise ToolError(str(e))


@mcp.tool()
async def draft_cover_letter(
    ctx: Context,
    application_id: str,
    resume_id: Annotated[
        Optional[str], Field(description="Defaults to the latest base resume")
    ] = None,
) -> dict:
    """Draft a short cover letter for one posting, grounded in the resume."""
    try:
        return await service.draft_cover_letter(
            _caller_id(ctx), application_id, resume_id
        )
    except ValueError as e:
        raise ToolError(str(e))


@mcp.tool()
async def prepare_for_interview(
    ctx: Context,
    application_id: str,
    resume_id: Annotated[
        Optional[str], Field(description="Defaults to the latest base resume")
    ] = None,
) -> str:
    """Likely interview questions for one posting — technical ones its
    requirements imply, behavioural ones its gaps invite — each with a line on
    what a strong answer needs from this user's own experience."""
    try:
        return await service.interview_prep(_caller_id(ctx), application_id, resume_id)
    except ValueError as e:
        raise ToolError(str(e))


# ── the resume itself ────────────────────────────────────────────────────────
@mcp.tool()
async def save_resume_details(
    ctx: Context,
    content: Annotated[
        ResumeContent,
        Field(description="Everything the user gave you, in structure"),
    ],
    label: Annotated[
        Optional[str], Field(description="What to call this resume")
    ] = None,
) -> dict:
    """Record a resume the user describes rather than uploads — for someone with
    no resume file yet, or building one from scratch.

    Write down only what they actually told you. Never fill a gap with a
    plausible-sounding employer, date or bullet: this becomes the evidence every
    later fit score and rewrite reasons from, so an invention here propagates
    into everything. Ask them for what is missing instead."""
    resume = await service.save_manual_resume(
        _caller_id(ctx), content.model_dump(), label
    )
    return {
        "resume_id": resume["id"],
        "label": resume["label"],
        "source": "manual",
        "saved": True,
    }


@mcp.tool()
async def update_resume_details(
    ctx: Context,
    changes: Annotated[
        ResumeContent,
        Field(
            description="Only the sections to change; anything left empty is "
            "kept as it is"
        ),
    ],
    resume_id: Annotated[
        Optional[str], Field(description="Defaults to the latest base resume")
    ] = None,
) -> dict:
    """Correct or extend a stored resume, section by section — a new job, a
    fixed phone number, a skill that was missed when the file was parsed.

    Sections you send replace their old value; sections you leave empty are
    untouched. To add one bullet to a job, send that whole job's experience
    entry with the full bullet list, or the others are lost. This edits the
    STRUCTURE only — the user's uploaded file is never modified."""
    user_id = _caller_id(ctx)
    target = await service.resolve_resume(user_id, resume_id)
    if not target:
        raise ToolError("No resume on file yet — use save_resume_details first.")
    # exclude_defaults: an omitted section arrives as an empty list, which is
    # indistinguishable from "delete everything here" unless it is dropped.
    updated = await service.update_resume_content(
        user_id, target["id"], changes.model_dump(exclude_defaults=True)
    )
    return {
        "resume_id": target["id"],
        "updated_sections": sorted(changes.model_dump(exclude_defaults=True).keys()),
        "content": (updated or {}).get("content"),
    }


# ── write ────────────────────────────────────────────────────────────────────
@mcp.tool()
async def save_tailored_resume(
    ctx: Context,
    application_id: str,
    content: Annotated[
        ResumeContent, Field(description="The approved resume, in full")
    ],
    base_resume_id: Annotated[
        Optional[str], Field(description="The resume this was derived from")
    ] = None,
    edits: Annotated[
        Optional[List[ResumeEdit]],
        Field(description="The before/after changes shown to the user"),
    ] = None,
    changes: Annotated[
        Optional[List[str]], Field(description="The changelog shown to the user")
    ] = None,
    keywords_added: Optional[List[str]] = None,
) -> dict:
    """Persist a tailored resume THE USER HAS ALREADY APPROVED.

    It becomes a NEW version. The base resume and the original file the user
    uploaded are never touched, so the document they have been sending out is
    still there unchanged. The saved version is downloadable as .docx or .pdf,
    and its `edits` stay available for applying by hand in their own file."""
    user_id = _caller_id(ctx)
    application = await _require_application(user_id, application_id)
    resume = await service.save_tailored_resume(
        user_id,
        {
            "application_id": application_id,
            "base_resume_id": base_resume_id,
            "content": content.model_dump(),
            "edits": [e.model_dump() for e in (edits or [])],
            "changes": changes or [],
            "keywords_added": keywords_added or [],
            "company": application.get("company"),
            "role_title": application.get("role_title"),
        },
    )
    return {
        "resume_id": resume["id"],
        "label": resume["label"],
        "saved": True,
        "edits": len(resume.get("edits") or []),
        "download": {
            "docx": f"/api/job-helper/resumes/{resume['id']}/download?fmt=docx",
            "pdf": f"/api/job-helper/resumes/{resume['id']}/download?fmt=pdf",
        },
    }


# ── hand off ─────────────────────────────────────────────────────────────────
@mcp.tool()
async def add_followup_tasks(
    ctx: Context,
    application_id: str,
    titles: Annotated[
        List[str],
        Field(description="Short task titles, e.g. 'Email recruiter about timeline'"),
    ],
) -> dict:
    """Put next steps for an application on the user's to-do list, where their
    agenda and reminders already live. Use for anything with a date attached."""
    user_id = _caller_id(ctx)
    await _require_application(user_id, application_id)
    created = await service.push_followups(user_id, application_id, titles)
    return {"created": len(created), "titles": [c["title"] for c in created]}


@mcp.tool()
async def start_roadmap_for_gap(ctx: Context, application_id: str) -> dict:
    """Start a structured learning roadmap on the most valuable skill this
    application is missing, and add its steps to the user's task list. Requires
    assess_resume_fit to have run — that is what identifies the gap."""
    user_id = _caller_id(ctx)
    await _require_application(user_id, application_id)
    result = await service.suggest_learning(user_id, application_id)
    if not result:
        raise ToolError(
            "No skill gap recorded for this application yet — run "
            "assess_resume_fit first."
        )
    return result


def main() -> None:
    """Standalone entry point. Invoked by __main__.py — see RUNNING IT above."""
    import sys

    logging.basicConfig(level=logging.INFO)
    if "--http" in sys.argv:
        mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.getenv("JOB_MCP_PORT", "8101"))
        mcp.run(transport="streamable-http")
    else:
        # stdio speaks the protocol over the pipe — nothing may be printed to
        # stdout, so logging goes to stderr (basicConfig's default).
        mcp.run(transport="stdio")
