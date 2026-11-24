"""The daily job feed: every board searched, deduplicated, and ranked against
the user's resume, so they open one list instead of five portals.

    preferences ─→ search_all (every board) ─┐
    extension "Import this page" ───────────┴→ _ingest ─→ _score_pending
                                               fingerprint  fast model, capped,
                                               dealbreaker  once per posting
                                               prefilter

Two ways in, one pipeline. The boards the backend can query directly arrive via
`refresh_feed`; the portals it cannot (Naukri, LinkedIn, Instahyre…) arrive via
`import_listings`, from the results page the user has open in their own browser.
Either way a posting lands once, gets scored once, and keeps its state —
dismissed stays dismissed — however many times it is seen again.

The feed is deliberately separate from the tracker. Nothing here is an
application until the user picks it (`track_item`), which is when the full
posting is analysed by the main model — the expensive pass is spent only on
jobs they actually want.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from app.agents.memory_store import get_profile

from . import repository as repo
from .jobsearch import search_all
from .ranking import dealbreaker, fingerprint, prefilter, score_listing
from .service import _resume_digest, fetch_job_posting, track_application
from .state import JobPreferences

logger = logging.getLogger(__name__)

# Model calls per refresh/import. Unscored leftovers are picked up, best
# pre-score first, by the next run — so a big first sweep costs the same as a
# quiet day, it just takes a couple of days to score the long tail.
MAX_SCORED_PER_RUN = 20
SCORE_CONCURRENCY = 5

# Results per board per query. Twenty is the most most boards return per page.
RESULTS_PER_CALL = 15

# A posting no board has returned for this long has almost certainly closed.
PRUNE_AFTER_DAYS = 30

# A description shorter than this is a teaser (a results-page card, an Adzuna
# snippet), so tracking it fetches the full posting first.
SHORT_DESCRIPTION_CHARS = 400


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── preferences ──────────────────────────────────────────────────────────────
async def get_preferences(user_id: str) -> tuple[JobPreferences, bool]:
    """The user's feed preferences and whether they have saved any.

    Unsaved, they are seeded from what the job helper has already learned in
    conversation (target roles, locations, work mode), so the first screen is
    a pre-filled form to confirm rather than a blank one.
    """
    stored = await repo.get_preferences(user_id)
    if stored:
        fields = {k: v for k, v in stored.items() if k in JobPreferences.model_fields}
        return JobPreferences.model_validate(fields), True

    memory = await get_profile(user_id)
    mode = (memory.get("preferred_work_mode") or "").lower()
    seeded = JobPreferences(
        queries=list(memory.get("target_roles") or [])[:5],
        locations=list(memory.get("target_locations") or [])[:5],
        work_mode=mode if mode in ("remote", "hybrid", "onsite") else "any",
        min_salary=memory.get("salary_expectation"),
    )
    return seeded, False


async def save_preferences(user_id: str, prefs: JobPreferences) -> JobPreferences:
    await repo.save_preferences(user_id, prefs.model_dump())
    # A posting hidden by the old dealbreakers may be wanted under the new ones.
    await repo.clear_filtered_feed(user_id)
    return prefs


# ── ingest ───────────────────────────────────────────────────────────────────
def _better(a: dict, b: dict) -> dict:
    """Of two copies of one posting, keep the one the scorer can read best."""
    def quality(listing: dict) -> tuple:
        return (
            not listing.get("description_truncated"),
            len(listing.get("description") or ""),
        )

    return a if quality(a) >= quality(b) else b


def _prepare(
    listings: List[dict],
    prefs: JobPreferences,
    skills: List[str],
    source: Optional[str] = None,
) -> List[dict]:
    """Dedupe a batch, apply dealbreakers and the free pre-score."""
    unique: dict = {}
    for listing in listings:
        if not (listing.get("title") or "").strip():
            continue
        key = fingerprint(listing)
        unique[key] = _better(listing, unique[key]) if key in unique else listing

    items = []
    for key, listing in unique.items():
        pre, hits = prefilter(listing, skills, prefs.queries)
        blocked = dealbreaker(listing, prefs)
        items.append(
            {
                "fingerprint": key,
                "title": listing.get("title"),
                "company": listing.get("company"),
                "location": listing.get("location"),
                "remote": bool(listing.get("remote")),
                "url": listing.get("url") or "",
                "description": listing.get("description") or "",
                "description_truncated": bool(listing.get("description_truncated")),
                "salary": listing.get("salary"),
                "posted_at": listing.get("posted_at"),
                "source": source or listing.get("source"),
                "prefilter_score": pre,
                "skill_hits": hits,
                **(
                    {"state": "filtered", "filtered_reason": blocked}
                    if blocked
                    else {"state": "new"}
                ),
            }
        )
    return items


async def _score_pending(
    user_id: str, prefs: JobPreferences, resume: Optional[dict]
) -> List[dict]:
    """Score up to MAX_SCORED_PER_RUN unscored postings. Returns them scored.

    A failed call leaves its posting unscored for the next run rather than
    failing the whole refresh.
    """
    if not resume:
        return []
    pending = await repo.unscored_feed_items(user_id, MAX_SCORED_PER_RUN)
    if not pending:
        return []

    digest = _resume_digest(resume.get("content"))
    memory = await get_profile(user_id)
    gate = asyncio.Semaphore(SCORE_CONCURRENCY)

    async def one(item: dict) -> Optional[dict]:
        async with gate:
            try:
                score = await score_listing(item, digest, prefs, memory)
            except Exception as e:
                logger.warning("feed: scoring %s failed: %s", item.get("id"), e)
                return None
        updates = {
            "score": score.match_score,
            "matched": score.matched,
            "missing": score.missing,
            "reason": score.reason,
            "min_years_required": score.min_years_required,
            "scored_at": _now().isoformat(),
            "resume_id": resume.get("id"),
        }
        too_senior = (
            prefs.max_years_required is not None
            and score.min_years_required is not None
            and score.min_years_required > prefs.max_years_required
        )
        if too_senior:
            updates["state"] = "filtered"
            updates["filtered_reason"] = (
                f"asks for {score.min_years_required}+ years"
            )
        return await repo.update_feed_item(user_id, item["id"], updates)

    results = await asyncio.gather(*(one(item) for item in pending))
    return [r for r in results if r]


# ── the two ways in ──────────────────────────────────────────────────────────
async def refresh_feed(user_id: str) -> dict:
    """Search every board for the user's preferences and rank what is new.

    Raises ValueError when there is nothing to search for yet.
    """
    prefs, _ = await get_preferences(user_id)
    if not prefs.queries:
        raise ValueError(
            "Tell me which roles to look for first — for example 'AI Engineer'."
        )

    search = await search_all(
        prefs.queries,
        prefs.locations,
        remote=prefs.work_mode == "remote",
        limit=RESULTS_PER_CALL,
    )
    resume = await repo.latest_base_resume(user_id)
    skills = ((resume or {}).get("content") or {}).get("skills") or []

    items = _prepare(search["listings"], prefs, skills)
    new = await repo.upsert_feed_items(user_id, items)
    scored = await _score_pending(user_id, prefs, resume)

    cutoff = (_now() - timedelta(days=PRUNE_AFTER_DAYS)).isoformat()
    try:
        await repo.prune_feed(user_id, cutoff)
    except Exception as e:  # housekeeping must not fail the refresh
        logger.warning("feed: prune failed for user=%s: %s", user_id, e)

    logger.info(
        "feed refresh user=%s: %d listings, %d new, %d scored via %s",
        user_id,
        len(search["listings"]),
        new,
        len(scored),
        search["providers"],
    )
    return {
        "found": len(items),
        "new": new,
        # Without descriptions: this is a summary, and GET /feed has the rest.
        "scored": [
            {k: v for k, v in item.items() if k != "description"} for item in scored
        ],
        "providers": search["providers"],
        "notes": search["notes"],
        "needs_resume": resume is None,
    }


async def import_listings(user_id: str, listings: List[dict], source: str) -> dict:
    """Listings read from a portal page in the user's own browser.

    Same pipeline as a refresh. Returns the imported postings as they now
    stand in the feed — including ones already there — so the extension can
    show the ranking for exactly the page the user is looking at.
    """
    prefs, _ = await get_preferences(user_id)
    resume = await repo.latest_base_resume(user_id)
    skills = ((resume or {}).get("content") or {}).get("skills") or []

    items = _prepare(listings, prefs, skills, source=source)
    new = await repo.upsert_feed_items(user_id, items)
    await _score_pending(user_id, prefs, resume)
    current = await repo.feed_items_by_fingerprint(
        user_id, [item["fingerprint"] for item in items]
    )
    return {"new": new, "items": current, "needs_resume": resume is None}


# ── leaving the feed ─────────────────────────────────────────────────────────
async def track_item(user_id: str, item_id: str) -> Optional[dict]:
    """Turn a feed posting into a tracked application. Idempotent: tracking it
    twice returns the same application. None when the item does not exist."""
    item = await repo.get_feed_item(user_id, item_id)
    if not item:
        return None
    if item.get("state") == "tracked" and item.get("application_id"):
        existing = await repo.get_application(user_id, item["application_id"])
        if existing:
            return existing

    text = item.get("description") or ""
    url = item.get("url") or None
    if url and (item.get("description_truncated") or len(text) < SHORT_DESCRIPTION_CHARS):
        # A teaser makes a poor analysis; try for the full posting, and fall
        # back to the teaser when the site blocks the fetch.
        fetched = await fetch_job_posting(url)
        if len(fetched) > len(text):
            text = fetched
    if not text.strip():
        raise ValueError(
            "That posting has no description to analyse — open it and use Save "
            "job, or paste the description."
        )

    application = await track_application(user_id, jd_text=text, source_url=url)
    linked = await repo.update_application(
        user_id,
        application["id"],
        {
            "feed_item_id": item_id,
            "source": item.get("source"),
            "feed_score": item.get("score"),
        },
    )
    await repo.update_feed_item(
        user_id,
        item_id,
        {"state": "tracked", "application_id": application["id"]},
    )
    return linked or application
