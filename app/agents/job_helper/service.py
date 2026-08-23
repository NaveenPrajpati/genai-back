"""Intent-free domain API for the job helper.

Everything the agent can *do*, as plain awaitable functions: read a posting,
analyse it, parse a resume, score the fit, tailor a resume, draft a letter. The
MCP server, the HTTP routes and the cross-agent handoffs all call these — no
caller has to go through the graph, which is what keeps a handoff from dragging
another agent's intent classification and HITL interrupts into its own run (the
same rule personal_assistant/service.py is built on).

THE ONE RULE THE PROMPTS ENFORCE
────────────────────────────────
Tailoring rewrites how real experience is presented. It never invents any. A
resume that claims Kubernetes because the posting asked for it is a resume that
fails at the interview, so the tailoring prompt is explicit about it, the missing
skills come back as a learning suggestion instead, and `TailoredResume.changes`
makes every edit visible before the user approves it.
"""

import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

# Every call here is generation or judgement over a full posting or resume —
# the extraction the fast model is good at is the memory pass, which runs in
# memory_store, not in this file.
from app.core.llm import llm

from . import repository as repo
from .documents import MAX_RESUME_CHARS
from .jobsearch import search_jobs
from .state import (
    CoverLetter,
    JobAnalysis,
    ResumeContent,
    SkillGap,
    TailoredResume,
)

logger = logging.getLogger(__name__)

# Postings are long and mostly boilerplate past the requirements; this is what
# the analysis prompt sees. Storage keeps its own (larger) cap.
MAX_JD_PROMPT_CHARS = 12_000

FETCH_TIMEOUT = 20.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── reading a posting ────────────────────────────────────────────────────────
async def _tavily_extract(url: str) -> str:
    """Page text via Tavily's extract endpoint, using the key the PA's web
    search already needs. Empty string when unconfigured or unsuccessful."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return ""
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as client:
            resp = await client.post(
                "https://api.tavily.com/extract",
                json={"api_key": api_key, "urls": [url]},
            )
            if resp.status_code != 200:
                logger.info("tavily extract %s: %s", resp.status_code, resp.text[:200])
                return ""
            results = resp.json().get("results") or []
            return (results[0].get("raw_content") or "").strip() if results else ""
    except Exception as e:
        logger.info("tavily extract error for %s: %s", url, e)
        return ""


async def _plain_fetch(url: str) -> str:
    """Last resort: fetch the page and strip the tags ourselves."""
    try:
        from bs4 import BeautifulSoup

        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; JobHelper/1.0)"},
        ) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "form"]):
            tag.decompose()
        return "\n".join(
            line for line in (t.strip() for t in soup.get_text("\n").splitlines()) if line
        )
    except Exception as e:
        logger.info("plain fetch error for %s: %s", url, e)
        return ""


async def fetch_job_posting(url: str) -> str:
    """The text of a posting at `url`, or "" if it could not be read.

    Returning empty rather than raising is the point: the big portals
    (LinkedIn, Indeed) serve a login wall to any unauthenticated fetch, and that
    is a normal outcome the caller answers by asking the user to paste the text,
    not an error that should fail their turn.
    """
    text = await _tavily_extract(url)
    if not text:
        text = await _plain_fetch(url)
    return text[:MAX_JD_PROMPT_CHARS]


ANALYZE_SYSTEM = """\
You read a job posting and extract what it actually asks for.

Separate genuine must-haves (`required_skills`) from things the posting frames \
as bonuses, "nice to have", or "you might also know" (`preferred_skills`). \
Write skills as short canonical names — "PostgreSQL", "CI/CD", "React" — not as \
sentences copied from the posting.

`keywords` is different: those are for a keyword screen, so keep the posting's \
own spelling and include acronyms and tool names as written.

Leave a field null when the posting does not say. Do not infer a salary, a \
deadline, or a seniority the text does not state.
"""


async def analyze_jd(jd_text: str) -> dict:
    """Structured requirements for one posting."""
    chain = llm.with_structured_output(JobAnalysis)
    result: JobAnalysis = await chain.ainvoke(
        [
            SystemMessage(content=ANALYZE_SYSTEM),
            HumanMessage(content=jd_text[:MAX_JD_PROMPT_CHARS]),
        ]
    )
    return result.model_dump()


# ── resumes ──────────────────────────────────────────────────────────────────
PARSE_SYSTEM = """\
You turn the plain text of a resume into structured data.

Copy what is there; do not improve it. Keep each bullet's own wording, including \
its numbers and metrics — those are the evidence everything downstream reasons \
about. Do not merge bullets, invent dates, or add skills the resume does not \
list.

The text comes from a .docx or .pdf, so section order may be odd and a line may \
be split mid-sentence. Reassemble what clearly belongs together. If a section is \
absent, leave its list empty.
"""


async def parse_resume(text: str) -> dict:
    """Resume text → `ResumeContent`."""
    chain = llm.with_structured_output(ResumeContent)
    result: ResumeContent = await chain.ainvoke(
        [
            SystemMessage(content=PARSE_SYSTEM),
            HumanMessage(content=text[:MAX_RESUME_CHARS]),
        ]
    )
    return result.model_dump()


async def store_uploaded_resume(
    user_id: str,
    text: str,
    filename: Optional[str] = None,
    label: Optional[str] = None,
    file_meta: Optional[dict] = None,
) -> dict:
    """Parse an uploaded resume and store it as the user's base resume.

    `file_meta` is where the untouched original was stored (files.store). It is
    optional because storage is allowed to fail without losing the resume — the
    structure is what the agent needs, the file is what the user sends out.
    """
    content = await parse_resume(text)
    resume = await repo.insert_resume(
        user_id,
        {
            "kind": "base",
            "label": label or (content.get("headline") or "Base resume"),
            "content": content,
            "source": "upload",
            "file": file_meta,
        },
    )
    logger.info(
        "job helper: stored base resume %s for %s (original kept: %s)",
        resume["id"],
        user_id,
        bool(file_meta),
    )
    return resume


async def save_manual_resume(
    user_id: str, content: dict, label: Optional[str] = None
) -> dict:
    """Store a resume the user typed in rather than uploaded.

    The same schema either way, so everything downstream — fit scoring,
    tailoring, rendering — works identically. This is the path for someone with
    no resume file yet, and the repair path for a parse that got something
    wrong.
    """
    validated = ResumeContent.model_validate(content or {}).model_dump()
    resume = await repo.insert_resume(
        user_id,
        {
            "kind": "base",
            "label": label or (validated.get("headline") or "My resume"),
            "content": validated,
            "source": "manual",
        },
    )
    logger.info("job helper: stored typed resume %s for %s", resume["id"], user_id)
    return resume


async def update_resume_content(
    user_id: str, resume_id: str, changes: dict
) -> Optional[dict]:
    """Patch fields of a stored resume's content, section by section.

    A merge rather than a replace: correcting a mis-parsed phone number should
    not require re-sending five years of bullets, and a tool that takes the
    whole resume to change one line invites the model to re-type — and quietly
    reword — everything else.
    """
    resume = await repo.get_resume(user_id, resume_id)
    if not resume:
        return None
    merged = {**(resume.get("content") or {}), **{k: v for k, v in changes.items() if v is not None}}
    validated = ResumeContent.model_validate(merged).model_dump()
    return await repo.update_resume(user_id, resume_id, {"content": validated})


async def resolve_resume(user_id: str, resume_id: Optional[str]) -> Optional[dict]:
    """The resume a request means: the one named, else the latest base."""
    if resume_id:
        return await repo.get_resume(user_id, resume_id)
    return await repo.latest_base_resume(user_id)


# ── tracking an application ──────────────────────────────────────────────────
async def find_jobs(
    query: str,
    location: Optional[str] = None,
    remote: bool = False,
    limit: int = 8,
    provider: Optional[str] = None,
) -> dict:
    """Search job boards. See jobsearch.py for how a provider is chosen."""
    result = await search_jobs(query, location, remote, limit, provider)
    logger.info(
        "job search %r via %s: %d listings",
        query,
        result["provider"],
        len(result["listings"]),
    )
    return result


async def track_application(
    user_id: str,
    jd_text: Optional[str] = None,
    source_url: Optional[str] = None,
    status: str = "saved",
) -> dict:
    """Analyse a posting and start tracking it.

    Accepts pasted text or a URL. A URL that cannot be read (most big job
    portals block unauthenticated fetches) raises with a message that tells the
    user to paste the description instead.
    """
    text = (jd_text or "").strip()
    if not text and source_url:
        text = await fetch_job_posting(source_url)
        if not text:
            raise ValueError(
                "That link could not be read — many job sites block automated "
                "access. Paste the job description text instead."
            )
    if not text:
        raise ValueError("Provide the job description text or a link to it.")

    analysis = await analyze_jd(text)
    application = await repo.insert_application(
        user_id,
        {
            "jd_text": text,
            "source_url": source_url,
            "analysis": analysis,
            "status": status,
            "applied_at": _now() if status == "applied" else None,
        },
    )
    logger.info(
        "job helper: tracking %s at %s (%s)",
        application.get("role_title"),
        application.get("company"),
        application["id"],
    )
    return application


async def set_status(
    user_id: str,
    application_id: str,
    status: str,
    note: Optional[str] = None,
) -> Optional[dict]:
    """Move an application to a new stage.

    Stamping `applied_at` on the move to `applied` is what the follow-up sweep
    counts from, so it belongs here rather than in each caller — a status set
    through a route that forgot it would be invisible to the reminder.
    """
    updates: dict = {"status": status}
    if status == "applied":
        updates["applied_at"] = _now()
    updated = await repo.update_application(user_id, application_id, updates)
    if updated and note:
        updated = await repo.add_application_note(user_id, application_id, note)
    return updated


# ── fit ──────────────────────────────────────────────────────────────────────
GAP_SYSTEM = """\
You compare one resume against one job posting and report the fit honestly.

A skill is `matched` only when the resume shows evidence of it — a bullet, a \
project, a listed skill. It is `partial` when the evidence is adjacent rather \
than direct (Postgres experience against a MySQL requirement), and `notes` should \
say what the adjacent evidence was. Everything else required is `missing`.

`match_score` is your judgement of fit on the required skills alone, 0-100. Do \
not inflate it to be encouraging — the user is deciding where to spend their \
time, and a flattering score costs them a week.

`learn_first` is at most three missing skills, ordered by how much closing them \
would change this application's odds.
"""


def _resume_digest(content: dict) -> str:
    """The resume as compact text for a prompt. Sending the raw JSON burns
    tokens on field names and punctuation the model does not need."""
    resume = ResumeContent.model_validate(content or {})
    parts: List[str] = []
    if resume.headline:
        parts.append(f"Headline: {resume.headline}")
    if resume.summary:
        parts.append(f"Summary: {resume.summary}")
    if resume.skills:
        parts.append("Skills: " + ", ".join(resume.skills))
    for job in resume.experience:
        header = " — ".join(b for b in (job.title, job.company) if b)
        dates = " ".join(b for b in (job.start, job.end) if b)
        parts.append(f"\n{header} ({dates})")
        parts.extend(f"  - {b}" for b in job.bullets)
    for project in resume.projects:
        parts.append(f"\nProject: {project.name or 'untitled'}")
        parts.extend(f"  - {b}" for b in project.bullets)
    for school in resume.education:
        parts.append(
            "Education: "
            + " ".join(b for b in (school.degree, school.institution, school.year) if b)
        )
    if resume.certifications:
        parts.append("Certifications: " + ", ".join(resume.certifications))
    return "\n".join(parts)


def _jd_digest(application: dict) -> str:
    """The posting as the prompts see it: the structured analysis, falling back
    to raw text for an application stored before analysis existed."""
    analysis = application.get("analysis") or {}
    if not analysis:
        return (application.get("jd_text") or "")[:MAX_JD_PROMPT_CHARS]
    lines = [
        f"Role: {analysis.get('role_title') or '—'} at "
        f"{analysis.get('company') or '—'}",
        f"Seniority: {analysis.get('seniority') or '—'}",
        "Required: " + ", ".join(analysis.get("required_skills") or []),
        "Preferred: " + ", ".join(analysis.get("preferred_skills") or []),
        "Keywords: " + ", ".join(analysis.get("keywords") or []),
    ]
    responsibilities = analysis.get("responsibilities") or []
    if responsibilities:
        lines.append("Responsibilities:")
        lines.extend(f"  - {r}" for r in responsibilities)
    return "\n".join(lines)


async def assess_fit(
    user_id: str, application_id: str, resume_id: Optional[str] = None
) -> dict:
    """Score one resume against one tracked posting, and cache it on the
    application so the list screen can show it without re-spending a call."""
    application = await repo.get_application(user_id, application_id)
    if not application:
        raise ValueError("That application does not exist.")
    resume = await resolve_resume(user_id, resume_id)
    if not resume:
        raise ValueError(
            "No resume on file yet — upload one, or type in your details, "
            "before comparing it to a posting."
        )

    chain = llm.with_structured_output(SkillGap)
    result: SkillGap = await chain.ainvoke(
        [
            SystemMessage(content=GAP_SYSTEM),
            HumanMessage(
                content=(
                    f"JOB POSTING\n{_jd_digest(application)}\n\n"
                    f"RESUME\n{_resume_digest(resume.get('content'))}"
                )
            ),
        ]
    )
    gap = result.model_dump()
    await repo.update_application(
        user_id, application_id, {"gap": {**gap, "resume_id": resume["id"]}}
    )
    return gap


# ── tailoring ────────────────────────────────────────────────────────────────
TAILOR_SYSTEM = """\
You rewrite a resume so a specific posting's reader sees the relevant parts \
first. You are editing presentation, never facts.

Allowed:
  - Rewrite the summary and headline to speak to this role.
  - Reorder experience bullets, and reorder skills, so what matters here is on top.
  - Re-word a bullet to use the posting's vocabulary WHEN the underlying work is \
the same thing under a different name.
  - Drop a bullet that is irrelevant to this role.
  - Surface a project the posting makes relevant.

Never:
  - Add a skill, tool, employer, title, date, degree or certification the \
original does not contain.
  - Change a number, metric or date.
  - Claim experience with something listed as missing. If the posting wants it \
and the resume has no evidence, it stays absent.

Return three things:

1. `content` — the COMPLETE tailored resume, not a diff. Every section carried \
over, edited or not.
2. `edits` — the same changes as before/after pairs the user can apply inside \
their own document. `before` must be the EXACT existing line, copied character \
for character, so they can find it; use null only when you are adding something \
new. Keep them in the order they appear in the resume.
3. `changes` — one line per edit naming the section, for a quick read: \
"Summary: rewritten around payments infrastructure".

`keywords_added` is only posting vocabulary the user's real experience already \
supported.

Most people have already styled their resume and will apply your edits by hand \
in their own file, so `edits` is the part that gets used. An edit whose `before` \
does not match their document word for word is an edit they cannot find.
"""


async def tailor_resume(
    user_id: str,
    application_id: str,
    resume_id: Optional[str] = None,
    instructions: Optional[str] = None,
) -> dict:
    """Produce a tailored resume for one posting. NOTHING IS SAVED.

    The proposal comes back with its own changelog so the caller can show the
    user what changed and gate the save behind their approval — the same
    propose/save split the meal planner uses, for the same reason: an MCP tool
    call has nowhere to pause for a human.
    """
    application = await repo.get_application(user_id, application_id)
    if not application:
        raise ValueError("That application does not exist.")
    resume = await resolve_resume(user_id, resume_id)
    if not resume:
        raise ValueError(
            "No resume on file yet — upload one, or type in your details, "
            "before tailoring it to a posting."
        )

    gap = application.get("gap") or {}
    context = [
        f"JOB POSTING\n{_jd_digest(application)}",
        f"CURRENT RESUME (structured)\n{_resume_digest(resume.get('content'))}",
    ]
    if gap.get("missing"):
        # Handing the gap over stops the model reaching for the obvious fix.
        context.append(
            "SKILLS THIS RESUME DOES NOT EVIDENCE — do not claim any of them: "
            + ", ".join(gap["missing"])
        )
    if instructions:
        context.append(f"THE USER ALSO ASKED FOR\n{instructions}")

    chain = llm.with_structured_output(TailoredResume)
    result: TailoredResume = await chain.ainvoke(
        [
            SystemMessage(content=TAILOR_SYSTEM),
            HumanMessage(content="\n\n".join(context)),
        ]
    )
    proposal = result.model_dump()
    return {
        **proposal,
        "application_id": application_id,
        "base_resume_id": resume["id"],
        "company": application.get("company"),
        "role_title": application.get("role_title"),
        "saved": False,
    }


async def save_tailored_resume(user_id: str, proposal: dict) -> dict:
    """Persist a tailored resume THE USER HAS ALREADY APPROVED.

    A new version — the base resume and its stored original file are never
    touched, so the document the user has been sending out is still there
    unchanged if they decide they preferred it.
    """
    application_id = proposal.get("application_id")
    company = proposal.get("company")
    role = proposal.get("role_title")
    label = " — ".join(b for b in (role, company) if b) or "Tailored resume"
    resume = await repo.insert_resume(
        user_id,
        {
            "kind": "tailored",
            "label": label,
            "content": proposal.get("content") or {},
            "source": "tailored",
            "application_id": application_id,
            "base_resume_id": proposal.get("base_resume_id"),
            "edits": proposal.get("edits") or [],
            "changes": proposal.get("changes") or [],
            "keywords_added": proposal.get("keywords_added") or [],
        },
    )
    if application_id:
        await repo.update_application(
            user_id, application_id, {"tailored_resume_id": resume["id"]}
        )
    return resume


# ── cover letter ─────────────────────────────────────────────────────────────
COVER_SYSTEM = """\
You draft a cover letter in the user's voice: direct, specific, and short — \
three or four paragraphs, under 300 words.

Ground every claim in the resume. Pick the two or three things in it that this \
posting actually cares about and say what the user did, not that they are \
passionate. No filler openings ("I am writing to apply for"), no restating the \
whole resume, no invented enthusiasm about the company's mission.

Body text only: no address block, no "Dear Hiring Manager", no signature.
"""


async def draft_cover_letter(
    user_id: str, application_id: str, resume_id: Optional[str] = None
) -> dict:
    application = await repo.get_application(user_id, application_id)
    if not application:
        raise ValueError("That application does not exist.")
    resume = await resolve_resume(user_id, resume_id)
    if not resume:
        raise ValueError(
            "No resume on file yet — upload one or type in your details first."
        )

    chain = llm.with_structured_output(CoverLetter)
    result: CoverLetter = await chain.ainvoke(
        [
            SystemMessage(content=COVER_SYSTEM),
            HumanMessage(
                content=(
                    f"JOB POSTING\n{_jd_digest(application)}\n\n"
                    f"RESUME\n{_resume_digest(resume.get('content'))}"
                )
            ),
        ]
    )
    return result.model_dump()


# ── interview prep ───────────────────────────────────────────────────────────
PREP_SYSTEM = """\
You prepare someone for one specific interview.

Write the questions this posting and this resume make likely: the technical ones \
the required skills imply, and the behavioural ones the gaps invite ("you have \
not worked with X — how would you approach it?"). For each, add one line on what \
a strong answer needs, drawing on what is actually in their resume.

Six to eight questions. Plain markdown, no preamble.
"""


async def interview_prep(
    user_id: str, application_id: str, resume_id: Optional[str] = None
) -> str:
    application = await repo.get_application(user_id, application_id)
    if not application:
        raise ValueError("That application does not exist.")
    resume = await resolve_resume(user_id, resume_id)
    digest = _resume_digest(resume.get("content")) if resume else "(no resume on file)"
    response = await llm.ainvoke(
        [
            SystemMessage(content=PREP_SYSTEM),
            HumanMessage(
                content=f"JOB POSTING\n{_jd_digest(application)}\n\nRESUME\n{digest}"
            ),
        ]
    )
    return (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )


# ── cross-agent ──────────────────────────────────────────────────────────────
async def push_followups(
    user_id: str, application_id: str, titles: List[str]
) -> List[dict]:
    """Turn next steps for one application into personal-assistant to-dos.

    Imported here rather than at module scope: personal_assistant.service is a
    leaf, but going through the package's __init__ at import time would pull the
    PA graph into every job-helper import, including the MCP server's.
    """
    from app.agents.personal_assistant.service import TaskSpec, create_tasks

    application = await repo.get_application(user_id, application_id)
    label = " at ".join(
        b
        for b in (
            (application or {}).get("role_title"),
            (application or {}).get("company"),
        )
        if b
    )
    specs = [
        TaskSpec(
            title=title,
            details=f"Application: {label}" if label else None,
            # Stable per application+task so a replayed graph run does not
            # create the same follow-up twice.
            source_ref=f"{application_id}:{title[:60]}",
        )
        for title in titles
        if title.strip()
    ]
    return await create_tasks(user_id, specs, source="job_helper")


async def suggest_learning(user_id: str, application_id: str) -> Optional[dict]:
    """Hand the top missing skill to the learning tracker as a roadmap.

    Only fires when a fit assessment has already run — without a gap there is
    nothing to name, and asking the model to invent one would start a roadmap on
    a skill the user may already have.
    """
    from app.agents.learning_tracker.service import generate_roadmap

    application = await repo.get_application(user_id, application_id)
    gap = (application or {}).get("gap") or {}
    topic = next(iter(gap.get("learn_first") or gap.get("missing") or []), None)
    if not topic:
        return None
    logger.info("job helper → learning roadmap for %r (user=%s)", topic, user_id)
    return await generate_roadmap(user_id, topic)
