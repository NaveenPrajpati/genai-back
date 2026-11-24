"""Deduplicating and ranking job listings against one user.

THREE PASSES, CHEAPEST FIRST
────────────────────────────
A daily sweep returns a hundred-odd postings, most of them noise, and the same
posting several times. Spending a model call on each would make the feed the
most expensive thing in the app, so the work is ordered by cost:

  1. fingerprint()  — free. Collapses one posting seen on several boards (with
                      different URLs and slightly different titles) to one key.
  2. dealbreaker()  — free. Drops what the user said they never want: an
                      excluded company, a title keyword, onsite when they only
                      want remote.
     prefilter()    — free. Keyword overlap between the resume's skills and the
                      posting, used only to decide which listings are worth a
                      model call first.
  3. score_listing() — one fast-model call per surviving listing, capped per
                      run and paid once per posting (the score is stored).

The model never sees a listing the cheap passes already ruled out, and never
sees the same posting twice.
"""

import re
from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import fast_llm

from .state import JobPreferences, ListingScore

# How much of a posting the scorer reads. Requirements sit near the top; the
# tail is benefits and legal text that only costs tokens.
MAX_SCORE_DESCRIPTION_CHARS = 4_000

# City spellings that differ between boards. Without this, "Bengaluru, IN" from
# one board and "Bangalore, Karnataka" from another are two different jobs.
CITY_ALIASES = {
    "bengaluru": "bangalore",
    "gurugram": "gurgaon",
    "bombay": "mumbai",
    "new delhi": "delhi",
    "navi mumbai": "mumbai",
    "madras": "chennai",
}

# Words that boards bolt onto titles and that do not make it a different job.
TITLE_NOISE = re.compile(
    r"\b(remote|hybrid|onsite|on-site|wfh|urgent|hiring|immediate joiners?)\b"
)


def _norm(text: Optional[str]) -> str:
    text = (text or "").lower()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
    text = re.sub(r"[^a-z0-9+#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _city(location: Optional[str]) -> str:
    first = _norm((location or "").split(",")[0])
    return CITY_ALIASES.get(first, first)


def fingerprint(listing: dict) -> str:
    """One key per real-world posting, the same whichever board returned it."""
    company = _norm(listing.get("company"))
    title = TITLE_NOISE.sub(" ", _norm(listing.get("title")))
    title = re.sub(r"\s+", " ", title).strip()
    if not company:
        # No company to anchor on: the URL is the only stable identity left.
        return "url:" + (listing.get("url") or title)
    city = "remote" if listing.get("remote") else _city(listing.get("location"))
    return f"{company}|{title}|{city}"


def dealbreaker(listing: dict, prefs: JobPreferences) -> Optional[str]:
    """Why the user would never want this listing, or None if it may stay."""
    company = (listing.get("company") or "").lower()
    for excluded in prefs.exclude_companies:
        if excluded.strip() and excluded.strip().lower() in company:
            return f"excluded company: {excluded.strip()}"
    title = (listing.get("title") or "").lower()
    for word in prefs.exclude_keywords:
        if word.strip() and word.strip().lower() in title:
            return f"excluded keyword: {word.strip()}"
    if prefs.work_mode == "remote":
        is_remote = listing.get("remote") or "remote" in (
            listing.get("location") or ""
        ).lower()
        if not is_remote:
            return "not remote"
    return None


def _mentions(skill: str, text: str) -> bool:
    # Lookarounds instead of \b, which fails on skills that end in a symbol
    # ("C++", "C#") or start with one (".NET").
    pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def prefilter(
    listing: dict, skills: List[str], queries: List[str]
) -> Tuple[int, List[str]]:
    """A free 0-100 pre-score and the resume skills the posting mentions.

    Not shown as the match score — it cannot tell "requires Kubernetes" from
    "nice to have Kubernetes". It only decides which listings get a model call
    first when a run is capped.
    """
    text = f"{listing.get('title') or ''}\n{listing.get('description') or ''}".lower()
    skills = [s for s in skills if s and s.strip()]
    hits = [s for s in skills if _mentions(s, text)]
    # Ten matching skills is a strong posting whatever the resume's length; a
    # thirty-skill resume should not make every posting look weak.
    skill_part = min(len(hits) / max(1, min(len(skills), 10)), 1.0)

    title = _norm(listing.get("title"))
    title_part = 0.0
    for query in queries:
        words = [w for w in _norm(query).split() if len(w) > 1]
        if words:
            title_part = max(title_part, sum(w in title for w in words) / len(words))

    return round(70 * skill_part + 30 * title_part), hits


SCORE_SYSTEM = """\
You judge how well one job posting fits one candidate, from their resume.

Score `match_score` 0-100 on the posting's REQUIRED skills and seniority:
  85-100  they evidence nearly every requirement at the right level
  65-84   a solid fit with one or two gaps
  40-64   a stretch — several requirements unevidenced, or seniority off
  0-39    a different role, or far too senior/junior

Only count a skill as matched when the resume shows real evidence of it. Do not \
give credit for skills the candidate merely could learn. `matched` and \
`missing` are short skill names. `reason` is one plain sentence a candidate \
reads in a list ("Strong on LLM/RAG work; missing production Kubernetes"). \
Set `min_years_required` only when the posting states a number.

When the posting says "(No description — title only.)", judge from the title, \
company and location alone: score no higher than 70, leave `matched` and \
`missing` empty, and start `reason` with "From the title only:".
"""

# What the scorer is shown for a listing imported from a results page, where
# the card carries a title and no description (see feed.import_listings).
TITLE_ONLY = "(No description — title only.)"


def _prefs_context(prefs: JobPreferences, memory: dict) -> str:
    lines = []
    if prefs.queries:
        lines.append("Looking for: " + ", ".join(prefs.queries))
    if memory.get("years_experience") is not None:
        lines.append(f"Years of experience: {memory['years_experience']}")
    if prefs.min_salary:
        lines.append(f"Salary expectation: {prefs.min_salary}")
    if prefs.work_mode and prefs.work_mode != "any":
        lines.append(f"Wants: {prefs.work_mode}")
    return "\n".join(lines)


async def score_listing(
    listing: dict, resume_digest: str, prefs: JobPreferences, memory: dict
) -> ListingScore:
    """One fast-model call: how well this posting fits this resume."""
    posting = "\n".join(
        b
        for b in (
            f"Title: {listing.get('title') or '—'}",
            f"Company: {listing.get('company') or '—'}",
            f"Location: {listing.get('location') or '—'}"
            + (" (remote)" if listing.get("remote") else ""),
            f"Salary: {listing['salary']}" if listing.get("salary") else "",
            "",
            (listing.get("description") or "")[:MAX_SCORE_DESCRIPTION_CHARS]
            or TITLE_ONLY,
        )
        if b is not None
    )
    chain = fast_llm.with_structured_output(ListingScore)
    return await chain.ainvoke(
        [
            SystemMessage(content=SCORE_SYSTEM),
            HumanMessage(
                content=(
                    f"CANDIDATE\n{_prefs_context(prefs, memory)}\n\n"
                    f"RESUME\n{resume_digest}\n\n"
                    f"JOB POSTING\n{posting}"
                )
            ),
        ]
    )
