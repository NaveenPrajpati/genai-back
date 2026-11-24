"""Answering application forms: the browser extension sends a form's fields,
this returns a value for each one it can answer honestly — and says which ones
it cannot.

THREE PASSES, CHEAPEST FIRST
────────────────────────────
  1. rules        — free. Label patterns for the dozen fields every form has
                    (name, email, phone, LinkedIn, notice period, CTC…), the
                    user's remembered answers by exact question, and the
                    questions that must never be auto-answered (see below).
  2. fast model   — one call for whatever the rules could not place: maps an
                    oddly-worded field onto a profile value or a remembered
                    answer ("Notice (days)?" ← "30 days"), and picks the right
                    option of a dropdown.
  3. main model   — one call, only if there are open questions: anything that
                    needs the resume or the job ("Years with PyTorch?", "Why do
                    you want to join us?"). Facts come off the resume; prose is
                    drafted and flagged for review.

WHAT IS NEVER FILLED
────────────────────
Demographic and personal questions (gender, ethnicity, disability, veteran
status, date of birth, religion, caste, marital status) and consent boxes ("I
agree", "I certify") always come back as `needs_you`. They are the user's to
answer, every time, and a tool that ticks "I certify this is accurate" on
their behalf defeats the point of the box.

NOTHING IS GUESSED
──────────────────
A field nobody can answer from the profile, the remembered answers or the
resume comes back `needs_you` with a note, and the extension highlights it. A
model value for a text field is checked against what the user actually told us
(`_supported`), so a plausible invented number cannot slip through as a
"profile" answer. "Do you know Kubernetes?" is answered Yes only on resume
evidence; a resume that is silent leaves it to the user, who may know it.

The user still reviews everything and clicks Submit themselves — see the
extension. This module only proposes values.
"""

import json
import logging
import re
from datetime import date
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.memory_store import get_profile
from app.core.llm import fast_llm, llm

from . import repository as repo
from .service import _jd_digest, _resume_digest
from .state import (
    PROFILE_KEYS,
    ApplyProfile,
    FieldMappings,
    FormField,
    QuestionAnswers,
)

logger = logging.getLogger(__name__)

# How much posting text the answer pass reads when there is no tracked
# application (the extension sends the page's text instead).
MAX_JOB_TEXT_CHARS = 8_000

# Remembered answers shown to the mapping model. Most recent first.
MAX_REMEMBERED_IN_PROMPT = 40

CHOICE_TYPES = {"select", "radio", "checkbox"}


# ── labels ───────────────────────────────────────────────────────────────────
def norm_label(label: str) -> str:
    """The identity of a question across forms: lowercase words only, with
    the asterisks and "(required)" that forms decorate labels with removed.
    Also safe as a Mongo field name (no dots, no leading $)."""
    text = (label or "").lower()
    text = re.sub(r"\(\s*(required|optional)\s*\)|\*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:120]


SENSITIVE = re.compile(
    r"\b(gender|sex|race|racial|ethnic\w*|disabilit\w*|disabled|veteran|"
    r"religio\w*|caste|date of birth|dob|age|marital|pronouns?|"
    r"sexual orientation|transgender|lgbt\w*)\b"
)
CONSENT = re.compile(
    r"\b(i agree|agree to|consent|terms and conditions|privacy (policy|notice)|"
    r"i certify|certify that|i acknowledge|acknowledge that|declaration)\b"
)

# (pattern on the normalised label, profile key). First match wins, so the
# specific patterns sit above the general ones ("first name" before "name").
RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(p), key)
    for p, key in (
        (r"\bfirst name\b|\bgiven name\b", "first_name"),
        (r"\blast name\b|\bsurname\b|\bfamily name\b", "last_name"),
        (r"^(your |candidate |applicant )?(full |legal )?name$", "full_name"),
        (r"\be ?mail\b", "email"),
        (r"\b(phone|mobile|contact number|whatsapp)\b", "phone"),
        (r"\blinked ?in\b", "linkedin_url"),
        (r"\bgit ?hub\b", "github_url"),
        (r"\b(portfolio|personal (website|site)|website)\b", "portfolio_url"),
        (r"\bnotice period\b", "notice_period"),
        (
            r"\b(expected|desired) (annual |yearly |fixed )?(ctc|salary|compensation|pay)\b"
            r"|\bsalary expectations?\b",
            "expected_ctc",
        ),
        (
            r"\bcurrent (annual |yearly |fixed )?(ctc|salary|compensation|pay)\b",
            "current_ctc",
        ),
        (r"\bcurrent (company|employer|organi[sz]ation)\b", "current_company"),
        (r"\bcurrent (job )?(title|designation|role|position)\b", "current_title"),
        (
            r"^total (years of )?(work )?experience|^(total )?years of experience$"
            r"|^experience in years$|^total exp",
            "total_experience_years",
        ),
        (r"\bsponsor", "requires_sponsorship"),
        (
            r"\b(authori[sz]ed|eligible|legally allowed) to work\b"
            r"|\bwork (authori[sz]ation|permit)\b",
            "work_authorization",
        ),
        (r"\brelocat", "willing_to_relocate"),
        (
            r"^(current )?(city|location)$|^current location"
            r"|^where are you (currently )?(based|located)",
            "city",
        ),
        (r"^country( of residence)?$", "country"),
        (r"\bgraduation (year|date)\b|\byear of (graduation|passing)\b", "graduation_year"),
        (r"\b(highest )?(degree|qualification)\b", "highest_degree"),
        (r"\b(university|college|institution)\b", "university"),
    )
]


# ── values ───────────────────────────────────────────────────────────────────
def profile_values(profile: ApplyProfile) -> Dict[str, str]:
    """Every known profile value as the string a form would take."""
    data = profile.model_dump()
    if data.get("full_name") and not (data.get("first_name") or data.get("last_name")):
        parts = data["full_name"].split()
        data["first_name"] = parts[0]
        data["last_name"] = " ".join(parts[1:]) or None
    if not data.get("full_name") and data.get("first_name"):
        data["full_name"] = " ".join(
            b for b in (data["first_name"], data.get("last_name")) if b
        )

    out: Dict[str, str] = {}
    for key, value in data.items():
        if value is None or value == [] or value == "":
            continue
        if isinstance(value, bool):
            out[key] = "Yes" if value else "No"
        elif isinstance(value, float):
            out[key] = f"{value:g}"
        elif isinstance(value, list):
            out[key] = ", ".join(str(v) for v in value)
        else:
            out[key] = str(value).strip()
    return out


_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)")
_PLUS = re.compile(r"(\d+(?:\.\d+)?)\s*\+|(?:more than|over|above)\s*(\d+(?:\.\d+)?)")
_LESS = re.compile(r"(?:less than|under|below|up to|upto)\s*(\d+(?:\.\d+)?)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _number_fits(number: float, option: str) -> bool:
    text = option.lower()
    if m := _RANGE.search(text):
        return float(m.group(1)) <= number <= float(m.group(2))
    if m := _PLUS.search(text):
        return number >= float(m.group(1) or m.group(2))
    if m := _LESS.search(text):
        return number < float(m.group(1)) or (
            "up to" in text or "upto" in text
        ) and number <= float(m.group(1))
    return False


def fit_option(value: Optional[str], options: List[str]) -> Optional[str]:
    """`value` expressed as one of `options`, or None when none fits.

    A free-text field takes the value as is. For a choice field the value has
    to become one of the form's own strings, or the browser cannot select it.
    """
    if value is None or not str(value).strip():
        return None
    value = str(value).strip()
    options = [o for o in options if o and o.strip()]
    if not options:
        return value
    v = value.lower()

    for option in options:
        if option.strip().lower() == v:
            return option

    if v in ("yes", "true", "y"):
        hits = [o for o in options if o.strip().lower().startswith("yes")]
        return hits[0] if len(hits) == 1 else None
    if v in ("no", "false", "n"):
        hits = [o for o in options if o.strip().lower().startswith("no")]
        return hits[0] if len(hits) == 1 else None

    contained = [
        o
        for o in options
        if len(o.strip()) > 1 and (o.strip().lower() in v or v in o.strip().lower())
    ]
    if len(contained) == 1:
        return contained[0]

    # "30 days" against "15-30 days" / "More than 30 days"; "3.5" against "3-5".
    numbers = _NUMBER.findall(v)
    if len(numbers) == 1:
        n = float(numbers[0])
        ranged = [o for o in options if _number_fits(n, o)]
        if len(ranged) == 1:
            return ranged[0]
    return None


def _supported(value: str, known: str) -> bool:
    """Whether a model's value for a free-text field is something the user
    actually told us. Every number in it must appear in what they told us, and
    so must the value itself or a word-level majority of it."""
    v = norm_label(value)
    k = norm_label(known)
    if not v:
        return False
    if v in k:
        return True
    if any(n not in k for n in _NUMBER.findall(v)):
        return False
    words = [w for w in v.split() if len(w) > 2]
    return bool(words) and sum(w in k for w in words) / len(words) >= 0.6


def _answer(field: FormField, value=None, source=None, needs_you=False, note=None):
    if value is not None and field.max_length:
        value = value[: field.max_length]
    return {
        "key": field.key,
        "label": field.label,
        "value": value,
        "source": source,
        "needs_you": needs_you,
        "note": note,
    }


def _shape(field: FormField, value: Optional[str]) -> Optional[str]:
    """A value adapted to the field's type, or None if it cannot be."""
    if value is None:
        return None
    if field.type == "checkbox" and not field.options:
        low = value.strip().lower()
        if low in ("yes", "true"):
            return "true"
        if low in ("no", "false"):
            return "false"
        return None
    if field.type == "number":
        numbers = _NUMBER.findall(value)
        return numbers[0] if len(numbers) == 1 else None
    if field.type in CHOICE_TYPES:
        return fit_option(value, field.options)
    return value


# ── pass 1: rules ────────────────────────────────────────────────────────────
def rule_answer(
    field: FormField, values: Dict[str, str], remembered: Dict[str, dict]
) -> Optional[dict]:
    """An answer the rules settle for free, or None to hand the field on."""
    label = norm_label(field.label)

    if SENSITIVE.search(label):
        return _answer(field, needs_you=True, note="Personal question — answer it yourself.")
    if CONSENT.search(label) and field.type in CHOICE_TYPES:
        return _answer(field, needs_you=True, note="Read and confirm this yourself.")
    if field.type == "file":
        if re.search(r"\bcover\b", label):
            return _answer(field, needs_you=True, note="Cover letter upload.")
        return _answer(field, source="resume_file")

    if label and label in remembered:
        shaped = _shape(field, remembered[label].get("value"))
        if shaped is not None:
            return _answer(field, shaped, "remembered")

    key = next((k for pattern, k in RULES if pattern.search(label)), None)
    if key is None and field.type == "email":
        key = "email"
    if key is None and field.type == "tel":
        key = "phone"
    if key is None:
        return None
    if key not in values:
        return _answer(
            field,
            needs_you=True,
            note=f"Not in your apply profile yet ({key.replace('_', ' ')}).",
        )
    shaped = _shape(field, values[key])
    # A profile value that fits none of a dropdown's options is not a dead end:
    # the model may know "Bengaluru" is the option "Bangalore". Hand it on.
    return _answer(field, shaped, "profile") if shaped is not None else None


# ── pass 2: map onto the profile ─────────────────────────────────────────────
MAP_SYSTEM = """\
You map the fields of a job application form onto what is already known about \
the applicant. For each field, decide its `kind`:

  profile     one of the PROFILE values answers it.
  remembered  one of the PREVIOUS ANSWERS is to the same question, even if \
worded differently.
  question    answering needs their resume or the job: skills, years with a \
specific technology, projects, motivation, "why us", cover-letter text.
  unknown     none of these — a referral's name, an interview date, anything \
the profile and previous answers do not contain.

For profile and remembered, put the value in `value`, adapted to the field: for \
a choice field copy one of its options exactly; for a number field digits only; \
otherwise keep the applicant's own wording. Do not convert units or currencies \
("12 LPA" stays "12 LPA", not a rupee amount) — if the field needs a different \
unit than the profile gives, it is `unknown`.

Never invent a value. If the profile does not contain what the field asks, it \
is `unknown` (or `question` if a resume could answer it) — even when a guess \
seems obvious. Return every field key exactly once.
"""


def _field_json(fields: List[FormField]) -> str:
    return json.dumps(
        [
            {
                "key": f.key,
                "label": f.label,
                "type": f.type,
                **({"options": f.options} if f.options else {}),
                **({"max_length": f.max_length} if f.max_length else {}),
            }
            for f in fields
        ],
        ensure_ascii=False,
    )


async def map_fields(
    fields: List[FormField], values: Dict[str, str], remembered: Dict[str, dict]
) -> Tuple[List[dict], List[FormField]]:
    """Answers the fast model can place, and the fields left for the resume."""
    if not fields:
        return [], []
    recent = sorted(
        remembered.values(), key=lambda a: a.get("updated_at") or "", reverse=True
    )[:MAX_REMEMBERED_IN_PROMPT]
    previous = "\n".join(f"- {a.get('label')}: {a.get('value')}" for a in recent)
    profile = "\n".join(f"- {k}: {v}" for k, v in values.items())

    chain = fast_llm.with_structured_output(FieldMappings)
    result: FieldMappings = await chain.ainvoke(
        [
            SystemMessage(content=MAP_SYSTEM),
            HumanMessage(
                content=(
                    f"PROFILE\n{profile or '(empty)'}\n\n"
                    f"PREVIOUS ANSWERS\n{previous or '(none)'}\n\n"
                    f"FIELDS\n{_field_json(fields)}"
                )
            ),
        ]
    )

    known = "\n".join([*values.values(), *(str(a.get("value")) for a in recent)])
    by_key = {m.key: m for m in result.fields}
    answers, questions = [], []
    for field in fields:
        mapping = by_key.get(field.key)
        if mapping is None or mapping.kind == "unknown":
            answers.append(_answer(field, needs_you=True, note="I don't know this one."))
            continue
        if mapping.kind == "question":
            questions.append(field)
            continue
        shaped = _shape(field, mapping.value)
        trusted = shaped is not None and (
            field.type in CHOICE_TYPES or _supported(shaped, known)
        )
        if trusted:
            answers.append(_answer(field, shaped, mapping.kind))
        else:
            answers.append(_answer(field, needs_you=True, note="I don't know this one."))
    return answers, questions


# ── pass 3: answer from the resume ───────────────────────────────────────────
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1
)}
_ONGOING = {"", "present", "current", "now", "till date", "today", "ongoing"}


def _month_index(text: Optional[str], today: date) -> Optional[int]:
    """'Jun 2023' / '06/2023' / '2023' / 'Present' as a month count, or None."""
    t = (text or "").strip().lower()
    if t in _ONGOING:
        return today.year * 12 + today.month
    year = re.search(r"(19|20)\d{2}", t)
    if not year:
        return None
    month = next((n for name, n in _MONTHS.items() if name in t), None)
    if month is None:
        numeric = re.search(r"\b(0?[1-9]|1[0-2])\s*[/\-.]\s*(19|20)\d{2}", t)
        # A bare year is counted from mid-year: neither end is known.
        month = int(numeric.group(1)) if numeric else 6
    return int(year.group(0)) * 12 + month


def role_durations(content: dict, today: Optional[date] = None) -> str:
    """How long each role lasted, computed here rather than by the model —
    which is unreliable at date arithmetic and was undercounting
    "Jun 2023 – Present" by years."""
    today = today or date.today()
    lines = []
    for job in (content or {}).get("experience") or []:
        # Only an END can be "ongoing"; a missing start is unknown, not today.
        if not (job.get("start") or "").strip():
            continue
        start = _month_index(job.get("start"), today)
        end = _month_index(job.get("end"), today)
        if start is None or end is None or end < start:
            continue
        years = (end - start) / 12
        who = " at ".join(b for b in (job.get("title"), job.get("company")) if b)
        lines.append(f"- {who or 'Role'}: {years:.1f} years")
    return "\n".join(lines)


ANSWER_SYSTEM = """\
You answer the open questions on a job application form for the applicant, \
using only their resume and the job posting.

Factual questions — "years of experience with Python", "have you worked with \
Kubernetes?", "your most recent project" — are answered from the resume, with \
basis "resume". Answer Yes only when the resume shows evidence. When the resume \
does not mention it, return value null and basis "unknown": the applicant may \
have experience their resume leaves out, so they decide. Count years from the \
ROLE DURATIONS given below — they are already computed, so use them rather than \
the raw dates — and never round up. Years with a specific tool are at most the \
years of the roles whose bullets show that tool; a tool listed only under \
skills gets no years.

Open questions — "why do you want to join", "describe a relevant project", \
"anything else we should know" — are written in the applicant's voice with \
basis "drafted": first person, plain, 2-4 sentences unless the field allows \
more. Lead with the concrete overlap: a specific thing from the resume (what \
they built, with its number if it has one) set against a specific thing the \
posting asks for. Say nothing about the company the posting does not say. Never \
use: "excited", "passionate", "thrilled", "opportunity to", "aligns with", \
"dynamic", "eager", "I am writing to", "I believe I would be a great fit".

The register to aim for — an example from an unrelated field, for tone only: \
"I spent the last two years on settlement systems at Northwind, where I cut \
daily reconciliation from six hours to forty minutes. Your posting is the same \
problem for cross-border payouts, which is the harder version of work I already \
know well."

For a choice field the value must be one of its options exactly. Respect \
max_length. Never claim a skill, employer, degree, date or number the resume \
does not contain. Return every field key exactly once.
"""


async def answer_questions(
    fields: List[FormField], resume: Optional[dict], job: str
) -> List[dict]:
    if not fields:
        return []
    if not resume:
        return [
            _answer(f, needs_you=True, note="Upload your resume so I can answer this.")
            for f in fields
        ]
    chain = llm.with_structured_output(QuestionAnswers)
    result: QuestionAnswers = await chain.ainvoke(
        [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(
                content=(
                    f"RESUME\n{_resume_digest(resume.get('content'))}\n\n"
                    f"ROLE DURATIONS\n{role_durations(resume.get('content') or {}) or '(no dates)'}\n\n"
                    f"JOB POSTING\n{job or '(not available)'}\n\n"
                    f"FIELDS\n{_field_json(fields)}"
                )
            ),
        ]
    )
    by_key = {a.key: a for a in result.answers}
    answers = []
    for field in fields:
        reply = by_key.get(field.key)
        shaped = _shape(field, reply.value) if reply and reply.basis != "unknown" else None
        if shaped is None:
            answers.append(
                _answer(field, needs_you=True, note="Your resume doesn't say — answer this yourself.")
            )
        elif reply.basis == "drafted":
            answers.append(
                _answer(field, shaped, "drafted", note="Drafted for you — read it before submitting.")
            )
        else:
            answers.append(_answer(field, shaped, "resume"))
    return answers


# ── profile ──────────────────────────────────────────────────────────────────
def _profile_from_resume(content: dict) -> dict:
    contact = content.get("contact") or {}
    out: dict = {
        "full_name": contact.get("name"),
        "email": contact.get("email"),
        "phone": contact.get("phone"),
    }
    if contact.get("location"):
        parts = [p.strip() for p in contact["location"].split(",") if p.strip()]
        out["city"] = parts[0] if parts else None
        if len(parts) > 1:
            out["country"] = parts[-1]
    for link in contact.get("links") or []:
        low = link.lower()
        if "linkedin.com" in low:
            out.setdefault("linkedin_url", link)
        elif "github.com" in low:
            out.setdefault("github_url", link)
        else:
            out.setdefault("portfolio_url", link)
    jobs = content.get("experience") or []
    if jobs and (jobs[0].get("end") or "present").strip().lower() in (
        "present", "current", "now", "till date", "",
    ):
        out["current_title"] = jobs[0].get("title")
        out["current_company"] = jobs[0].get("company")
    schools = content.get("education") or []
    if schools:
        out["highest_degree"] = schools[0].get("degree")
        out["university"] = schools[0].get("institution")
        out["graduation_year"] = schools[0].get("year")
    return {k: v for k, v in out.items() if v}


async def get_apply_profile(user_id: str) -> Tuple[ApplyProfile, bool, List[dict]]:
    """The profile, whether the user has saved it, and remembered answers.

    Unsaved, it is pre-filled from the resume and what the job helper has
    learned in conversation, for the user to check — never used silently as if
    confirmed: the fill response says `profile_saved: false`.
    """
    doc = await repo.get_apply_profile(user_id) or {}
    remembered = [
        {"key": k, "label": v.get("label"), "value": v.get("value")}
        for k, v in (doc.get("answers") or {}).items()
    ]
    if doc.get("profile"):
        return ApplyProfile.model_validate(doc["profile"]), True, remembered

    resume = await repo.latest_base_resume(user_id)
    seeded = _profile_from_resume((resume or {}).get("content") or {})
    memory = await get_profile(user_id)
    if memory.get("years_experience") is not None:
        seeded["total_experience_years"] = memory["years_experience"]
    if memory.get("work_authorization"):
        seeded["work_authorization"] = memory["work_authorization"]
    if memory.get("salary_expectation"):
        seeded["expected_ctc"] = memory["salary_expectation"]
    if memory.get("target_locations"):
        seeded["preferred_locations"] = list(memory["target_locations"])[:10]
    try:
        return ApplyProfile.model_validate(seeded), False, remembered
    except Exception:
        # A parsed resume with an odd value must not break the profile screen.
        return ApplyProfile(), False, remembered


async def remember(user_id: str, items: List[dict]) -> dict:
    """Keep answers the user chose to reuse. Personal questions are refused:
    they are never auto-filled, so storing them would only be a liability."""
    keep, skipped = {}, []
    for item in items:
        label = (item.get("label") or "").strip()
        value = (item.get("value") or "").strip()
        norm = norm_label(label)
        if not norm or not value:
            continue
        if SENSITIVE.search(norm) or CONSENT.search(norm):
            skipped.append(label)
            continue
        keep[norm] = {"label": label[:300], "value": value[:2000]}
    total = await repo.remember_answers(user_id, keep)
    return {"saved": len(keep), "skipped": skipped, "total": total}


# ── the whole fill ───────────────────────────────────────────────────────────
async def resume_for(user_id: str, application: Optional[dict]) -> Optional[dict]:
    """The resume an application goes out with: its approved tailored version
    if there is one, else the latest base resume."""
    if application and application.get("tailored_resume_id"):
        tailored = await repo.get_resume(user_id, application["tailored_resume_id"])
        if tailored:
            return tailored
    return await repo.latest_base_resume(user_id)


async def fill_form(
    user_id: str,
    fields: List[FormField],
    application: Optional[dict] = None,
    job_text: Optional[str] = None,
) -> dict:
    """A value or a `needs_you` for every field, in the order they were sent.

    Model failures degrade to `needs_you` for the fields they would have
    answered — the rule-based answers are still worth returning.
    """
    profile, saved, remembered_list = await get_apply_profile(user_id)
    values = profile_values(profile)
    remembered = {a["key"]: a for a in remembered_list}
    resume = await resume_for(user_id, application)

    results: Dict[str, dict] = {}
    pending: List[FormField] = []
    for field in fields:
        settled = rule_answer(field, values, remembered)
        if settled is not None:
            results[field.key] = settled
        else:
            pending.append(field)

    questions: List[FormField] = []
    try:
        mapped, questions = await map_fields(pending, values, remembered)
        results.update({a["key"]: a for a in mapped})
    except Exception as e:
        logger.warning("fill: mapping pass failed: %s", e)
        for field in pending:
            results[field.key] = _answer(field, needs_you=True, note="Couldn't work this one out.")

    if application:
        job = _jd_digest(application)
    else:
        job = (job_text or "")[:MAX_JOB_TEXT_CHARS]
    try:
        results.update({a["key"]: a for a in await answer_questions(questions, resume, job)})
    except Exception as e:
        logger.warning("fill: answer pass failed: %s", e)
        for field in questions:
            results[field.key] = _answer(field, needs_you=True, note="Couldn't draft this one.")

    answers = [results[f.key] for f in fields if f.key in results]
    return {
        "answers": answers,
        "filled": sum(1 for a in answers if a["value"] is not None),
        "needs_you": sum(1 for a in answers if a["needs_you"]),
        "profile_saved": saved,
        "resume": (
            {
                "id": resume.get("id"),
                "kind": resume.get("kind"),
                "label": resume.get("label"),
                "has_original_file": bool((resume.get("file") or {}).get("path")),
            }
            if resume
            else None
        ),
    }
