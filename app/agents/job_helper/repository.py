"""MongoDB persistence for the job helper: tracked applications and resumes.

TWO COLLECTIONS, ONE OF THEM POLYMORPHIC
────────────────────────────────────────
`job_applications` is one document per posting the user is tracking, carrying
the raw JD text alongside the structured analysis extracted from it. The raw
text is kept deliberately: a re-analysis with a better prompt should not require
the user to find and paste the posting again.

`resumes` holds both the user's base resume and every version tailored from it,
separated by `kind`. They share a schema (`state.ResumeContent`) and every read
path wants them together — "show me my resumes" means both — so splitting them
across two collections would buy nothing but a join.

A base resume carries `content` (the structure the agent reasons over) and,
when it came from an upload, `file` (where the untouched original lives — see
files.py). A resume typed in by hand has content and no file; that is a normal
state, not a missing one.

THE POINTER ON THE USER
───────────────────────
`users.resume` mirrors the active base resume — its id, filename and storage
path. The resumes themselves stay in their own collection, because a user
accumulates versions and a growing array on a document read by every
authenticated request is the wrong shape. The pointer exists so anything holding
a user (the profile screen, another agent) can say "they have a resume, here it
is" without a second query.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from bson import ObjectId
from pymongo import ReturnDocument

from app.database import get_db

logger = logging.getLogger(__name__)

APPLICATIONS = "job_applications"
RESUMES = "resumes"

# How much JD text is stored per application. Postings run long and the whole
# thing is re-sent to the model on every re-analysis; past this the tail is
# boilerplate (equal-opportunity statements, benefits, legal).
MAX_JD_CHARS = 20_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize(doc: Optional[dict]) -> Optional[dict]:
    """Make a Mongo doc JSON-safe (ObjectId -> str), as the PA repository does."""
    if not doc:
        return None
    out = dict(doc)
    if "_id" in out:
        out["id"] = str(out.pop("_id"))
    return out


def _oid(value: str) -> Optional[ObjectId]:
    """A guessed or malformed id is a 404, not a 500."""
    return ObjectId(value) if ObjectId.is_valid(value) else None


# ── applications ─────────────────────────────────────────────────────────────
async def insert_application(user_id: str, data: dict) -> dict:
    now = _now()
    analysis = data.get("analysis") or {}
    doc = {
        "user_id": user_id,
        # Denormalised from the analysis so the list screen and the reminder
        # sweep never have to reach into the nested blob to say what a row is.
        "company": data.get("company") or analysis.get("company"),
        "role_title": data.get("role_title") or analysis.get("role_title"),
        "location": data.get("location") or analysis.get("location"),
        "source_url": data.get("source_url"),
        "jd_text": (data.get("jd_text") or "")[:MAX_JD_CHARS],
        "analysis": analysis,
        "gap": data.get("gap"),
        "status": data.get("status", "saved"),
        "applied_at": data.get("applied_at"),
        "deadline": data.get("deadline") or analysis.get("deadline"),
        "notes": [],
        "created_at": now,
        "updated_at": now,
    }
    res = await get_db()[APPLICATIONS].insert_one(doc)
    doc["_id"] = res.inserted_id
    return _serialize(doc)


async def list_applications(
    user_id: str, status: Optional[str] = None, limit: int = 100
) -> List[dict]:
    query: dict = {"user_id": user_id}
    if status:
        query["status"] = status
    cursor = (
        get_db()[APPLICATIONS]
        # jd_text is the bulk of a document and no list screen renders it.
        .find(query, {"jd_text": 0})
        .sort("updated_at", -1)
    )
    return [_serialize(d) for d in await cursor.to_list(length=limit)]


async def get_application(user_id: str, application_id: str) -> Optional[dict]:
    oid = _oid(application_id)
    if not oid:
        return None
    doc = await get_db()[APPLICATIONS].find_one({"_id": oid, "user_id": user_id})
    return _serialize(doc)


async def verify_application_ownership(application_id: str, user_id: str) -> bool:
    oid = _oid(application_id)
    if not oid:
        return False
    return await get_db()[APPLICATIONS].count_documents(
        {"_id": oid, "user_id": user_id}, limit=1
    ) > 0


async def update_application(
    user_id: str, application_id: str, updates: dict
) -> Optional[dict]:
    oid = _oid(application_id)
    if not oid:
        return None
    updates = {k: v for k, v in updates.items() if v is not None}
    if not updates:
        return await get_application(user_id, application_id)
    updates["updated_at"] = _now()
    doc = await get_db()[APPLICATIONS].find_one_and_update(
        {"_id": oid, "user_id": user_id},
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    return _serialize(doc)


async def add_application_note(
    user_id: str, application_id: str, note: str
) -> Optional[dict]:
    """Append a dated note — an interview outcome, a recruiter reply."""
    oid = _oid(application_id)
    if not oid:
        return None
    doc = await get_db()[APPLICATIONS].find_one_and_update(
        {"_id": oid, "user_id": user_id},
        {
            "$push": {"notes": {"content": note, "created_at": _now()}},
            "$set": {"updated_at": _now()},
        },
        return_document=ReturnDocument.AFTER,
    )
    return _serialize(doc)


async def delete_application(user_id: str, application_id: str) -> bool:
    oid = _oid(application_id)
    if not oid:
        return False
    res = await get_db()[APPLICATIONS].delete_one({"_id": oid, "user_id": user_id})
    return res.deleted_count > 0


async def stale_applications(user_id: str, since_iso: str) -> List[dict]:
    """Applications sent before `since_iso` and still sitting at `applied`.

    The follow-up sweep's query: an application that has moved to interviewing
    or been rejected has an answer already, and one still at `saved` was never
    sent, so neither is waiting on anything.
    """
    cursor = get_db()[APPLICATIONS].find(
        {
            "user_id": user_id,
            "status": "applied",
            "applied_at": {"$lt": since_iso, "$ne": None},
        },
        {"jd_text": 0, "analysis": 0},
    )
    return [_serialize(d) for d in await cursor.to_list(length=50)]


# ── resumes ──────────────────────────────────────────────────────────────────
async def insert_resume(user_id: str, data: dict) -> dict:
    """Store a resume. `kind` is "base" (the user's own) or "tailored"."""
    now = _now()
    doc = {
        "user_id": user_id,
        "kind": data.get("kind", "base"),
        "label": data.get("label") or "Resume",
        "content": data.get("content") or {},
        # "upload" (parsed from a file, `file` is set) or "manual" (typed in).
        "source": data.get("source", "upload"),
        "file": data.get("file"),
        # Set on tailored versions only: which posting it targets and what it
        # was derived from, so a version is never orphaned from its context.
        "application_id": data.get("application_id"),
        "base_resume_id": data.get("base_resume_id"),
        "edits": data.get("edits") or [],
        "changes": data.get("changes") or [],
        "keywords_added": data.get("keywords_added") or [],
        "created_at": now,
        "updated_at": now,
    }
    res = await get_db()[RESUMES].insert_one(doc)
    doc["_id"] = res.inserted_id
    resume = _serialize(doc)
    if doc["kind"] == "base":
        await set_active_resume(user_id, resume)
    return resume


async def set_active_resume(user_id: str, resume: dict) -> None:
    """Mirror the active base resume onto the user document (see the module
    docstring). Best-effort: the resume is already saved, and failing the
    request over a denormalised copy would be the wrong trade."""
    stored = resume.get("file") or {}
    try:
        await get_db()["users"].update_one(
            {"uid": user_id},
            {
                "$set": {
                    "resume": {
                        "resume_id": resume["id"],
                        "label": resume.get("label"),
                        "source": resume.get("source"),
                        "filename": stored.get("filename"),
                        # The path, not a URL: links are signed per read and
                        # a stored one is either expired or permanent, and
                        # both are wrong for a file this private.
                        "file_path": stored.get("path"),
                        "updated_at": resume.get("updated_at"),
                    }
                }
            },
        )
    except Exception as e:
        logger.error("active resume pointer failed user=%s: %s", user_id, e)


async def clear_active_resume(user_id: str, resume_id: str) -> None:
    """Drop the pointer if it names this resume — called on delete, so the user
    document never advertises a resume that is gone."""
    try:
        await get_db()["users"].update_one(
            {"uid": user_id, "resume.resume_id": resume_id},
            {"$unset": {"resume": ""}},
        )
    except Exception as e:
        logger.error("active resume clear failed user=%s: %s", user_id, e)


async def list_resumes(
    user_id: str,
    kind: Optional[str] = None,
    application_id: Optional[str] = None,
    limit: int = 100,
) -> List[dict]:
    query: dict = {"user_id": user_id}
    if kind:
        query["kind"] = kind
    if application_id:
        query["application_id"] = application_id
    cursor = (
        get_db()[RESUMES]
        # `content` is the whole resume; the list screen shows labels and dates.
        .find(query, {"content": 0})
        .sort("created_at", -1)
    )
    return [_serialize(d) for d in await cursor.to_list(length=limit)]


async def get_resume(user_id: str, resume_id: str) -> Optional[dict]:
    oid = _oid(resume_id)
    if not oid:
        return None
    doc = await get_db()[RESUMES].find_one({"_id": oid, "user_id": user_id})
    return _serialize(doc)


async def verify_resume_ownership(resume_id: str, user_id: str) -> bool:
    oid = _oid(resume_id)
    if not oid:
        return False
    return await get_db()[RESUMES].count_documents(
        {"_id": oid, "user_id": user_id}, limit=1
    ) > 0


async def latest_base_resume(user_id: str) -> Optional[dict]:
    """The resume tailoring starts from when the user names none: their most
    recently uploaded base. Returns None when they have never uploaded one —
    every caller has to handle that, since it is the state a new user is in."""
    doc = await (
        get_db()[RESUMES]
        .find({"user_id": user_id, "kind": "base"})
        .sort("created_at", -1)
        .limit(1)
        .to_list(length=1)
    )
    return _serialize(doc[0]) if doc else None


async def update_resume(user_id: str, resume_id: str, updates: dict) -> Optional[dict]:
    oid = _oid(resume_id)
    if not oid:
        return None
    updates["updated_at"] = _now()
    doc = await get_db()[RESUMES].find_one_and_update(
        {"_id": oid, "user_id": user_id},
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    resume = _serialize(doc)
    if resume and resume.get("kind") == "base":
        await set_active_resume(user_id, resume)
    return resume


async def delete_resume(user_id: str, resume_id: str) -> Optional[dict]:
    """Delete a resume and return it, so the caller can clean up its stored
    file. None when nothing was deleted."""
    oid = _oid(resume_id)
    if not oid:
        return None
    doc = await get_db()[RESUMES].find_one_and_delete({"_id": oid, "user_id": user_id})
    resume = _serialize(doc)
    if resume:
        await clear_active_resume(user_id, resume_id)
    return resume


# ── indexes ──────────────────────────────────────────────────────────────────
# Every read here is scoped by user_id and then sorted or narrowed by status;
# unindexed that is a collection scan per screen. Same reasoning, and the same
# explicit names for idempotence, as learning_tracker/indexes.py.
INDEXES: tuple[tuple[str, list[tuple[str, int]], str], ...] = (
    (APPLICATIONS, [("user_id", 1), ("updated_at", -1)], "job_apps_by_owner"),
    (
        APPLICATIONS,
        [("user_id", 1), ("status", 1), ("applied_at", 1)],
        "job_apps_followup_sweep",
    ),
    (RESUMES, [("user_id", 1), ("kind", 1), ("created_at", -1)], "job_resumes_by_owner"),
    (
        RESUMES,
        [("user_id", 1), ("application_id", 1)],
        "job_resumes_by_application",
    ),
)


async def ensure_indexes() -> None:
    """Create the job-helper indexes. Idempotent, and best-effort: a failure here
    degrades performance, so it must not stop the app from booting."""
    db = get_db()
    for collection, keys, name in INDEXES:
        try:
            await db[collection].create_index(keys, name=name, background=True)
        except Exception as e:
            logger.error("job helper index %s failed: %s", name, e)
