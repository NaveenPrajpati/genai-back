"""Shared MongoDB-backed human-in-the-loop approval store.

Every agent that pauses for human review (roadmap save, meal plan, task delete)
funnels through one `approvals` collection and this small API, instead of each
workflow hand-rolling its own lookup/insert/resolve against a different store.

Canonical document shape:
    {
        _id, userId, threadId, action_type, payload,
        status: "pending" | "approved" | "rejected",
        createdAt, resolvedAt
    }

Workflow nodes are re-run safe (LangGraph replays the node on resume): they call
`get_pending` to reuse this thread's proposal, `create_pending` on first run, then
`resolve` after the side effect. Schedulers that pre-create a notification also use
`create_pending`. Routers use `get_pending` (ownership/resume) and `list_pending`
(the GET list). Each agent keeps its own interrupt/response shape on top of this,
so the client contract is unchanged.
"""

import logging
from datetime import datetime, timezone

from bson import ObjectId

from app.database import get_db

logger = logging.getLogger(__name__)


def _col():
    return get_db()["approvals"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_pending(
    userId: str, thread_id: str, action_type: str, payload: dict
) -> str | None:
    """Insert a pending approval and return its id (used by schedulers)."""
    try:
        res = await _col().insert_one(
            {
                "userId": userId,
                "threadId": thread_id,
                "action_type": action_type,
                "payload": payload,
                "status": "pending",
                "createdAt": _now(),
            }
        )
        return str(res.inserted_id)
    except Exception as e:
        logger.error("approval insert error thread=%s: %s", thread_id, e)
        return None


async def resolve(approval_id: str | None, status: str) -> None:
    """Mark an approval approved/rejected. No-op if approval_id is missing."""
    if not approval_id:
        return
    try:
        await _col().update_one(
            {"_id": ObjectId(approval_id)},
            {"$set": {"status": status, "resolvedAt": _now()}},
        )
    except Exception as e:
        logger.error("approval resolve error id=%s: %s", approval_id, e)


async def get_pending(thread_id: str) -> dict | None:
    """The pending approval for a thread (for router ownership checks/resume)."""
    return await _col().find_one({"threadId": thread_id, "status": "pending"})


async def list_pending(
    userId: str, action_types: list[str] | None = None
) -> list[dict]:
    """A user's pending approvals, optionally filtered to specific action_types."""
    query: dict = {"userId": userId, "status": "pending"}
    if action_types:
        query["action_type"] = {"$in": action_types}
    cursor = _col().find(query)
    docs = await cursor.to_list(None)
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return docs


def to_legacy(doc: dict) -> dict:
    """Project a canonical approval doc to the snake_case shape the meal/PA clients
    historically received from Supabase, so those list endpoints stay non-breaking."""
    return {
        "id": str(doc.get("_id")),
        "user_id": doc.get("userId"),
        "thread_id": doc.get("threadId"),
        "action_type": doc.get("action_type"),
        "payload": doc.get("payload"),
        "status": doc.get("status"),
        "created_at": doc.get("createdAt"),
        "resolved_at": doc.get("resolvedAt"),
    }
