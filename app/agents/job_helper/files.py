"""Storing the user's resume file, unmodified, in Supabase Storage.

WHY THE FILE IS KEPT AT ALL
───────────────────────────
The parsed structure is what the agent reasons over, but it is not what the user
sends to an employer. Their .docx carries a layout and typography we cannot
reproduce and should not try to — so the upload is stored byte-for-byte and
stays downloadable exactly as it arrived. Tailoring then returns *edits* they can
apply inside that document, rather than a replacement for it.

PRIVATE BUCKET, SIGNED LINKS
────────────────────────────
A resume is a home address, a phone number and an employment history in one
file, so the bucket is private and reads go through short-lived signed URLs
minted per request. A public URL would be a permanent, guessable, unauthenticated
link to a stranger's PII — and it would outlive the account that owns it.

The path is `{user_id}/{uuid}{ext}`, so ownership is visible in the key itself
and two users can never collide.

DEGRADING
─────────
Storage is best-effort: if the bucket is missing or Supabase is unreachable, the
upload still parses and the resume is still usable — the response just reports
that the original could not be kept. Losing the convenience of the original file
must not lose the resume.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.core.config import supabase

logger = logging.getLogger(__name__)

BUCKET = os.getenv("RESUME_BUCKET", "resumes")

# Signed links are minted per request and handed straight to the browser, so
# this only has to outlive one click.
SIGNED_URL_TTL = int(os.getenv("RESUME_URL_TTL", "3600"))

CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    ".txt": "text/plain",
    ".md": "text/markdown",
}


def content_type_for(filename: str) -> str:
    return CONTENT_TYPES.get(
        os.path.splitext(filename or "")[1].lower(), "application/octet-stream"
    )


def store(user_id: str, payload: bytes, filename: str) -> Optional[dict]:
    """Upload a resume and return its stored metadata, or None on failure.

    Synchronous: the Supabase client is, so callers run this in a thread the way
    the RAG routes already do for `storage.create_ingestion_log`.
    """
    extension = os.path.splitext(filename or "")[1].lower()
    path = f"{user_id}/{uuid.uuid4().hex}{extension}"
    try:
        supabase.storage.from_(BUCKET).upload(
            path,
            payload,
            {"content-type": content_type_for(filename), "upsert": "false"},
        )
    except Exception as e:
        logger.error("resume upload failed user=%s: %s", user_id, e)
        return None

    return {
        "path": path,
        "filename": filename,
        "content_type": content_type_for(filename),
        "size": len(payload),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }


def signed_url(path: Optional[str]) -> Optional[str]:
    """A short-lived download link for a stored resume, or None."""
    if not path:
        return None
    try:
        result = supabase.storage.from_(BUCKET).create_signed_url(path, SIGNED_URL_TTL)
    except Exception as e:
        logger.error("resume signed url failed path=%s: %s", path, e)
        return None
    if isinstance(result, dict):
        return result.get("signedURL") or result.get("signed_url")
    return getattr(result, "signed_url", None)


def fetch(path: Optional[str]) -> Optional[bytes]:
    """The stored file itself, for serving a download through our own auth
    rather than handing out a link."""
    if not path:
        return None
    try:
        return supabase.storage.from_(BUCKET).download(path)
    except Exception as e:
        logger.error("resume download failed path=%s: %s", path, e)
        return None


def remove(path: Optional[str]) -> None:
    """Delete a stored resume. Best-effort: a failure here leaves an orphaned
    object, which is a cleanup problem, not a request failure."""
    if not path:
        return
    try:
        supabase.storage.from_(BUCKET).remove([path])
    except Exception as e:
        logger.error("resume delete failed path=%s: %s", path, e)


def with_link(resume: Optional[dict]) -> Optional[dict]:
    """Attach a fresh signed URL to a resume document on its way out.

    The link is minted on read rather than stored, because a stored one expires
    and a non-expiring one is a permanent public handle on someone's PII.
    """
    if not resume:
        return resume
    stored = resume.get("file")
    if stored and stored.get("path"):
        resume["file"] = {**stored, "url": signed_url(stored["path"])}
    return resume
