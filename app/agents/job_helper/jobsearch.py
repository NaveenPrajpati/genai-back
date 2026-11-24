"""Finding job postings, across four providers behind one shape.

WHY A PROVIDER LAYER AND NOT ONE API
────────────────────────────────────
Job boards differ in the two ways that matter here — whether they need a key,
and whether they return the *full* description. The agent's whole value comes
from reading requirements out of a posting, so a provider that returns a
200-character teaser is a provider whose results still need a second fetch.

    provider    key needed        coverage              description
    ─────────────────────────────────────────────────────────────────────
    jsearch     RapidAPI          global, aggregated    full
    adzuna      app id + key      global inc. India     truncated
    remotive    none              remote tech only      full (HTML)
    arbeitnow   none              mostly EU             full (HTML)

`auto` (the default) picks the best provider that is actually configured and
falls back to a keyless one, so the tool works on a fresh clone with no keys and
gets better when a key is added — the same degradation the Tavily-backed web
search and the OCR path already use.

Everything is normalised to `JobListing` (state.py) so the MCP tool has one
schema no matter which board answered, and `track_job_posting` can take a search
result straight through without knowing where it came from.
"""

import asyncio
import logging
import os
from typing import List, Optional

import httpx

from .state import JobListing

logger = logging.getLogger(__name__)

SEARCH_TIMEOUT = 20.0

# Descriptions are the point, but a board that returns twenty full postings can
# return 200 KB. This is per listing, before the model ever sees it.
MAX_DESCRIPTION_CHARS = 6_000

PROVIDERS = ("jsearch", "adzuna", "remotive", "arbeitnow")


# Tags that end a line. Everything else is inline and must not break the
# sentence it sits in: a naive get_text("\n") turns "we use <b>Python</b> and
# Postgres" into three lines, and a requirements list read that way loses the
# very structure that makes it readable.
BLOCK_TAGS = ("p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section")


def _clean_html(raw: str) -> str:
    """Remotive and Arbeitnow return HTML descriptions; the model wants text."""
    if not raw:
        return ""
    if "<" not in raw:
        return raw.strip()[:MAX_DESCRIPTION_CHARS]
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for block in soup.find_all(BLOCK_TAGS):
            block.append("\n")
        # Empty separator: the newlines above carry the structure, so inline
        # tags close up rather than adding whitespace mid-word.
        text = soup.get_text("")
    except Exception:
        text = raw
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)[:MAX_DESCRIPTION_CHARS]


def configured_providers() -> List[str]:
    """Which providers can actually run right now, best first."""
    available = []
    if os.getenv("RAPIDAPI_KEY"):
        available.append("jsearch")
    if os.getenv("ADZUNA_APP_ID") and os.getenv("ADZUNA_APP_KEY"):
        available.append("adzuna")
    # Keyless, so always available — but narrow, hence last.
    available += ["remotive", "arbeitnow"]
    return available


def _resolve(provider: Optional[str]) -> str:
    choice = (provider or os.getenv("JOB_SEARCH_PROVIDER") or "auto").lower()
    if choice != "auto" and choice in PROVIDERS:
        return choice
    return configured_providers()[0]


# ── providers ────────────────────────────────────────────────────────────────
async def _jsearch(query: str, location: Optional[str], remote: bool, limit: int):
    """RapidAPI JSearch — aggregates Google for Jobs. Full descriptions."""
    search = f"{query} in {location}" if location else query
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        resp = await client.get(
            "https://jsearch.p.rapidapi.com/search",
            params={
                "query": search,
                "page": "1",
                "num_pages": "1",
                **({"remote_jobs_only": "true"} if remote else {}),
            },
            headers={
                "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY", ""),
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
            },
        )
        resp.raise_for_status()
        rows = resp.json().get("data") or []

    return [
        JobListing(
            title=row.get("job_title") or "",
            company=row.get("employer_name"),
            location=", ".join(
                b for b in (row.get("job_city"), row.get("job_country")) if b
            )
            or None,
            remote=bool(row.get("job_is_remote")),
            url=row.get("job_apply_link") or row.get("job_google_link") or "",
            description=_clean_html(row.get("job_description") or ""),
            salary=(
                f"{row.get('job_min_salary')}–{row.get('job_max_salary')} "
                f"{row.get('job_salary_currency') or ''}".strip()
                if row.get("job_min_salary")
                else None
            ),
            posted_at=row.get("job_posted_at_datetime_utc"),
            source="jsearch",
        )
        for row in rows[:limit]
    ]


async def _adzuna(query: str, location: Optional[str], remote: bool, limit: int):
    """Adzuna — real listings including India. Descriptions arrive truncated, so
    a posting the user cares about still wants `track_job_posting` on its URL."""
    country = os.getenv("ADZUNA_COUNTRY", "in").lower()
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        resp = await client.get(
            f"https://api.adzuna.com/v1/api/jobs/{country}/search/1",
            params={
                "app_id": os.getenv("ADZUNA_APP_ID", ""),
                "app_key": os.getenv("ADZUNA_APP_KEY", ""),
                "what": query,
                "results_per_page": min(limit, 20),
                **({"where": location} if location else {}),
            },
        )
        resp.raise_for_status()
        rows = resp.json().get("results") or []

    return [
        JobListing(
            title=row.get("title") or "",
            company=(row.get("company") or {}).get("display_name"),
            location=(row.get("location") or {}).get("display_name"),
            remote=remote,
            url=row.get("redirect_url") or "",
            description=_clean_html(row.get("description") or ""),
            salary=(
                f"{int(row['salary_min'])}–{int(row['salary_max'])}"
                if row.get("salary_min") and row.get("salary_max")
                else None
            ),
            posted_at=row.get("created"),
            source="adzuna",
            description_truncated=True,
        )
        for row in rows[:limit]
    ]


async def _remotive(query: str, location: Optional[str], remote: bool, limit: int):
    """Remotive — keyless, remote tech roles only, full HTML descriptions.

    Its `limit` parameter is advisory (it returned 20 rows for limit=2), so the
    slice here is what actually bounds the result.
    """
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        resp = await client.get(
            "https://remotive.com/api/remote-jobs",
            params={"search": query, "limit": limit},
        )
        resp.raise_for_status()
        rows = resp.json().get("jobs") or []

    return [
        JobListing(
            title=row.get("title") or "",
            company=row.get("company_name"),
            location=row.get("candidate_required_location"),
            remote=True,
            url=row.get("url") or "",
            description=_clean_html(row.get("description") or ""),
            salary=row.get("salary") or None,
            posted_at=row.get("publication_date"),
            source="remotive",
        )
        for row in rows[:limit]
    ]


async def _arbeitnow(query: str, location: Optional[str], remote: bool, limit: int):
    """Arbeitnow — keyless, mostly European. The board has no search parameter,
    so the filtering is ours: match the query's words against title, company and
    tags over one page."""
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        resp = await client.get("https://www.arbeitnow.com/api/job-board-api")
        resp.raise_for_status()
        rows = resp.json().get("data") or []

    terms = [t for t in query.lower().split() if len(t) > 2]

    def matches(row: dict) -> bool:
        haystack = " ".join(
            [
                row.get("title") or "",
                row.get("company_name") or "",
                " ".join(row.get("tags") or []),
            ]
        ).lower()
        return all(t in haystack for t in terms) if terms else True

    hits = [r for r in rows if matches(r)]
    if remote:
        hits = [r for r in hits if r.get("remote")]
    if location:
        needle = location.lower()
        hits = [r for r in hits if needle in (r.get("location") or "").lower()]

    return [
        JobListing(
            title=row.get("title") or "",
            company=row.get("company_name"),
            location=row.get("location"),
            remote=bool(row.get("remote")),
            url=row.get("url") or "",
            description=_clean_html(row.get("description") or ""),
            posted_at=str(row.get("created_at") or "") or None,
            source="arbeitnow",
        )
        for row in hits[:limit]
    ]


_IMPLEMENTATIONS = {
    "jsearch": _jsearch,
    "adzuna": _adzuna,
    "remotive": _remotive,
    "arbeitnow": _arbeitnow,
}


async def search_jobs(
    query: str,
    location: Optional[str] = None,
    remote: bool = False,
    limit: int = 8,
    provider: Optional[str] = None,
) -> dict:
    """Search one provider and return `{provider, listings, note}`.

    Never raises: a board that is down, rate-limited or has changed its shape
    comes back as an empty result with a `note` saying so, because the honest
    answer to "find me jobs" when the board is unreachable is "I could not
    reach it — paste one and I will analyse it", not a failed turn.
    """
    chosen = _resolve(provider)
    limit = max(1, min(limit, 20))
    try:
        listings = await _IMPLEMENTATIONS[chosen](query, location, remote, limit)
    except Exception as e:
        logger.warning("job search via %s failed: %s", chosen, e)
        return {
            "provider": chosen,
            "listings": [],
            "note": f"The {chosen} job board could not be reached just now.",
        }

    note = None
    if not listings:
        note = f"No matches on {chosen} for that search."
    elif chosen == "remotive":
        note = "Remotive lists remote tech roles only."
    elif chosen == "arbeitnow":
        note = "Arbeitnow listings are mostly European."
    elif chosen == "adzuna":
        note = (
            "Adzuna descriptions are truncated — track a posting by its URL to "
            "read the full requirements."
        )
    return {
        "provider": chosen,
        "listings": [listing.model_dump() for listing in listings],
        "note": note,
    }


# ── every board at once (the daily feed) ─────────────────────────────────────
# Boards that ignore the location parameter. Asking them once per location would
# just return the same page several times.
LOCATIONLESS = {"remotive"}

# Hard ceiling on outbound calls per sweep. Queries x locations x boards grows
# fast, and JSearch/Adzuna bill per request.
MAX_CALLS_PER_SWEEP = 24


async def search_all(
    queries: List[str],
    locations: Optional[List[str]] = None,
    remote: bool = False,
    limit: int = 10,
) -> dict:
    """Run every configured board for every query/location, concurrently.

    Returns `{listings, providers, notes}`. Never raises: one board being down
    costs its listings and adds a note, nothing more — the feed is the union of
    whatever answered. Deduplication is the caller's job (ranking.fingerprint),
    because the same posting arrives from several boards with different URLs.
    """
    providers = configured_providers()
    places: List[Optional[str]] = list(locations or []) or [None]
    limit = max(1, min(limit, 20))

    calls = []
    for provider in providers:
        for query in queries:
            for place in [None] if provider in LOCATIONLESS else places:
                calls.append((provider, query, place))
    calls = calls[:MAX_CALLS_PER_SWEEP]

    async def run(provider: str, query: str, place: Optional[str]):
        return await _IMPLEMENTATIONS[provider](query, place, remote, limit)

    results = await asyncio.gather(
        *(run(*call) for call in calls), return_exceptions=True
    )

    listings: List[dict] = []
    failed: set = set()
    answered: set = set()
    for (provider, query, place), result in zip(calls, results):
        if isinstance(result, BaseException):
            logger.warning(
                "feed search via %s (%r, %r) failed: %s", provider, query, place, result
            )
            failed.add(provider)
            continue
        answered.add(provider)
        listings.extend(listing.model_dump() for listing in result)

    notes = [
        f"The {p} job board could not be reached just now."
        for p in providers
        if p in failed and p not in answered
    ]
    return {
        "listings": listings,
        "providers": [p for p in providers if p in answered],
        "notes": notes,
    }
