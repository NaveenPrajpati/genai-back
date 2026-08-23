"""State and structured-output schemas for the job-helper agent.

`ResumeContent` is the centre of gravity here. A resume arrives as a .docx or
.pdf, but the agent never edits that file: it is parsed once into this schema and
everything downstream — the skill gap against a job description, the tailoring
pass, the suggested edits the user applies — reads and writes the structure, not
the document. A model asked to rewrite a binary it cannot see would guess; a
model handed `experience[2].bullets` can rewrite exactly that.

TWO REPRESENTATIONS, AND NEITHER REPLACES THE OTHER
───────────────────────────────────────────────────
  • the ORIGINAL FILE — stored untouched, downloadable byte-for-byte. It is the
    user's own layout and typography, and nothing we generate can reproduce
    that, so we do not try to. This is what they send to employers.
  • the STRUCTURE — what the agent reasons over. Filled by parsing an upload,
    by the user typing it in, or by editing either afterwards.

`ResumeEdit` is what tailoring returns: a before/after per bullet, so the user
can apply changes inside their own document and keep its design. Rendering the
structure to a fresh .docx/.pdf stays available for the typed-in path, where
there is no original file to preserve.
"""

from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# The lifecycle of one tracked application. `saved` is a posting the user is
# interested in but has not sent anything for yet — the state a JD lands in.
APPLICATION_STATUSES = (
    "saved",
    "applied",
    "interviewing",
    "offer",
    "rejected",
    "withdrawn",
)

ApplicationStatus = Literal[
    "saved", "applied", "interviewing", "offer", "rejected", "withdrawn"
]


class JobHelperState(TypedDict, total=False):
    query: str
    user_id: str
    thread_id: str
    current_user: dict
    memory: dict
    messages: Annotated[List[AnyMessage], add_messages]
    # Carried in from the client when the turn is about one specific item.
    application_id: Optional[str]
    resume_id: Optional[str]
    # Produced by the run.
    summary: Optional[str]
    resume_status: Optional[str]
    resume_version_id: Optional[str]
    tailored: Optional[dict]


# ── job descriptions ─────────────────────────────────────────────────────────
class JobAnalysis(BaseModel):
    """What a job description actually asks for, pulled out of its prose."""

    company: Optional[str] = None
    role_title: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = Field(
        default=None, description="full-time, contract, internship …"
    )
    seniority: Optional[str] = Field(
        default=None, description="intern, junior, mid, senior, lead"
    )
    min_years_experience: Optional[int] = None
    required_skills: List[str] = Field(
        default_factory=list,
        description="Skills the posting treats as must-haves, as short names "
        "('PostgreSQL', 'Kubernetes') rather than sentences",
    )
    preferred_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(
        default_factory=list,
        description="Terms a keyword screen would look for, including tools and "
        "acronyms spelled exactly as the posting spells them",
    )
    salary_text: Optional[str] = None
    deadline: Optional[str] = Field(
        default=None, description="ISO date if the posting states one"
    )


class JobListing(BaseModel):
    """One search result, normalised across providers (see jobsearch.py)."""

    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    remote: bool = False
    url: str = ""
    description: str = ""
    salary: Optional[str] = None
    posted_at: Optional[str] = None
    source: Optional[str] = Field(default=None, description="Which board answered")
    description_truncated: bool = Field(
        default=False,
        description="True when the board returns a teaser rather than the full "
        "posting — analysing it properly needs a fetch of the URL",
    )


class SkillGap(BaseModel):
    """How one resume stacks up against one posting."""

    matched: List[str] = Field(
        default_factory=list, description="Required skills the resume evidences"
    )
    partial: List[str] = Field(
        default_factory=list,
        description="Skills with adjacent but not direct evidence — name what is "
        "adjacent in `notes`",
    )
    missing: List[str] = Field(
        default_factory=list, description="Required skills with no evidence at all"
    )
    match_score: int = Field(
        default=0, ge=0, le=100, description="Rough fit, 0-100, on required skills"
    )
    notes: Optional[str] = Field(
        default=None, description="Two or three sentences of honest assessment"
    )
    learn_first: List[str] = Field(
        default_factory=list,
        description="At most three missing skills worth studying first, highest "
        "leverage first",
    )


# ── resumes ──────────────────────────────────────────────────────────────────
class ResumeContact(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    links: List[str] = Field(
        default_factory=list, description="Portfolio, GitHub, LinkedIn URLs"
    )


class ResumeExperience(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    start: Optional[str] = Field(default=None, description="As written, e.g. 'Jan 2023'")
    end: Optional[str] = Field(default=None, description="As written, or 'Present'")
    bullets: List[str] = Field(default_factory=list)


class ResumeEducation(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    year: Optional[str] = None
    details: Optional[str] = None


class ResumeProject(BaseModel):
    name: Optional[str] = None
    link: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)


class ResumeContent(BaseModel):
    """A resume as data. Rendered to .docx/.pdf by `documents.py`."""

    contact: ResumeContact = Field(default_factory=ResumeContact)
    headline: Optional[str] = Field(
        default=None, description="The one-line role label under the name"
    )
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[ResumeExperience] = Field(default_factory=list)
    projects: List[ResumeProject] = Field(default_factory=list)
    education: List[ResumeEducation] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)


class ResumeFile(BaseModel):
    """The user's uploaded resume, kept exactly as they made it."""

    path: Optional[str] = Field(default=None, description="Object path in storage")
    url: Optional[str] = Field(
        default=None, description="Signed download link; regenerated on read"
    )
    filename: Optional[str] = None
    content_type: Optional[str] = None
    size: Optional[int] = None
    uploaded_at: Optional[str] = None


class ResumeEdit(BaseModel):
    """One suggested change, in place, with the text it replaces.

    `before` is what makes this applicable by hand: the user finds that exact
    line in their own document and swaps it, keeping their layout. An edit with
    no `before` is an addition, and `section` says where it goes.
    """

    section: str = Field(
        description="summary, skills, experience[0], projects[1], education …"
    )
    before: Optional[str] = Field(
        default=None, description="The exact existing line, or null for an addition"
    )
    after: str = Field(description="What it should say instead")
    why: Optional[str] = Field(
        default=None, description="One short clause on what this earns"
    )


class TailoredResume(BaseModel):
    """A rewrite of one resume for one posting, as both a full version and a
    list of in-place edits.

    Both, because they serve different readers. `content` is the complete
    tailored resume the app stores and can render; `edits` is what the user
    applies by hand inside their own .docx so their design survives — which is
    the path most people actually take with a resume they have already styled.
    """

    content: ResumeContent
    edits: List[ResumeEdit] = Field(
        default_factory=list,
        description="Every change as before/after, in the order they appear",
    )
    changes: List[str] = Field(
        default_factory=list,
        description="One line per edit, each naming the section it changed",
    )
    keywords_added: List[str] = Field(
        default_factory=list,
        description="Posting keywords now present that were absent before — only "
        "ones the user's real experience supports",
    )


class CoverLetter(BaseModel):
    body: str = Field(description="The letter itself, no header block or signature")
    subject: Optional[str] = None


# ── memory ───────────────────────────────────────────────────────────────────
class JobMemoryExtract(BaseModel):
    """Durable job-search facts worth remembering across sessions. Every field is
    optional — only fill one when the message gives clear evidence for it."""

    target_roles: Optional[List[str]] = None
    target_locations: Optional[List[str]] = None
    current_title: Optional[str] = None
    years_experience: Optional[int] = None
    work_authorization: Optional[str] = None
    salary_expectation: Optional[str] = None
    job_search_status: Optional[str] = Field(
        default=None, description="actively looking | open to offers | not looking"
    )
    preferred_work_mode: Optional[str] = Field(
        default=None, description="remote | hybrid | onsite"
    )


JOB_MEMORY_INSTRUCTIONS = (
    "Extract durable job-search facts about the user from their message — the "
    "roles and locations they are targeting, their current title, years of "
    "experience, work authorization, salary expectation, how actively they are "
    "looking, and whether they want remote, hybrid or onsite work."
)
