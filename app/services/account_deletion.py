"""Erasing everything one account owns.

A user's data is spread across four systems — MongoDB, Supabase (Postgres and
object storage), Pinecone and Redis — because each agent stores what suits it.
That is fine while data is being written and a problem exactly once: when
someone asks for all of it to go.

So this module is deliberately the single place that knows the full inventory.
Adding a store elsewhere in the app without adding it to `MONGO_COLLECTIONS` (or
to `purge_user_data`) is the way an account deletion silently starts leaving
things behind, and there is a test asserting the Mongo list stays in step with
the collections the code actually writes.

Two rules shape the implementation:

1. **The account record goes last.** Everything else is deleted while the user
   still exists, so a failure part-way through leaves an account that can retry.
   Deleting the user first and then failing would strand data addressed by an id
   nothing points to any more.

2. **A failing store does not block the deletion.** A Pinecone outage must not
   be the reason someone cannot close their account. Each store is attempted
   independently, failures are collected and logged with the user id so orphans
   can be swept up later, and the account is removed regardless.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from bson import ObjectId

from app.database import get_db

logger = logging.getLogger(__name__)

# Every Mongo collection keyed by `user_id`. The users collection is not here:
# it is deleted last, by id, after this list has been swept.
MONGO_COLLECTIONS = (
    "todos",
    "memories",
    "approvals",
    "triggers",
    "roadmaps",
    "quizzes",
    "quiz_attempts",
    "learning_digests",
    "learning_misconceptions",
    "learning_notes",
    "job_applications",
    "resumes",
)

# Supabase tables owned by a user. `rag_messages` is absent on purpose — it is
# keyed by chat, not by user, and is deleted through its chats.
SUPABASE_TABLES = ("rag_ingestion_logs", "rag_chats")


@dataclass
class PurgeReport:
    """What went, and what refused to.

    `failures` being non-empty does not mean the account survived — it means
    some store still holds rows that need sweeping, which is an operational
    problem rather than a user-facing one.
    """

    deleted: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def record(self, store: str, count: int) -> None:
        if count:
            self.deleted[store] = self.deleted.get(store, 0) + count

    def fail(self, store: str, error: Exception) -> None:
        self.failures.append(store)
        logger.error("account purge failed store=%s: %s", store, error)

    @property
    def total(self) -> int:
        return sum(self.deleted.values())


async def _purge_mongo(user_id: str, report: PurgeReport) -> None:
    db = get_db()
    for name in MONGO_COLLECTIONS:
        try:
            result = await db[name].delete_many({"user_id": user_id})
            report.record(name, result.deleted_count)
        except Exception as e:
            report.fail(f"mongo:{name}", e)


def _purge_resume_files_sync(paths: list[str]) -> int:
    """Remove stored resume objects. Synchronous — the Supabase client is."""
    from app.agents.job_helper import files

    removed = 0
    for path in paths:
        files.remove(path)
        removed += 1
    return removed


async def _purge_resume_files(user_id: str, report: PurgeReport) -> None:
    """Delete uploaded resumes from object storage.

    Runs *before* the Mongo sweep, because the storage paths only exist on the
    `resumes` documents — once those rows are gone the files are unreachable
    and become permanent orphans in the bucket.
    """
    try:
        cursor = get_db()["resumes"].find({"user_id": user_id}, {"file.path": 1})
        paths = [
            path
            async for doc in cursor
            if (path := (doc.get("file") or {}).get("path"))
        ]
    except Exception as e:
        report.fail("storage:resumes", e)
        return

    if not paths:
        return
    try:
        removed = await asyncio.to_thread(_purge_resume_files_sync, paths)
        report.record("resume_files", removed)
    except Exception as e:
        report.fail("storage:resumes", e)


def _purge_supabase_sync(user_id: str) -> dict[str, int]:
    """Delete this user's Supabase rows, and the Pinecone vectors behind them.

    Chats cascade by hand — `rag_messages` has no user column, so its rows are
    only reachable through the chat ids collected here. Same for vectors: they
    are addressed by `doc_id`, which lives on the ingestion log.
    """
    from app.core.config import supabase
    from app.services.rag.step4_retrieval import delete_doc_vectors

    counts: dict[str, int] = {}

    logs = (
        supabase.table("rag_ingestion_logs")
        .select("doc_id")
        .eq("user_id", user_id)
        .execute()
    )
    vectors = 0
    for row in logs.data or []:
        doc_id = row.get("doc_id")
        if not doc_id:
            continue
        try:
            vectors += delete_doc_vectors(doc_id)
        except Exception as e:
            # One unreachable document must not strand the rest of the purge.
            logger.error("vector delete failed doc=%s: %s", doc_id, e)
    if vectors:
        counts["pinecone_vectors"] = vectors

    chats = supabase.table("rag_chats").select("id").eq("user_id", user_id).execute()
    chat_ids = [row["id"] for row in (chats.data or []) if row.get("id")]
    if chat_ids:
        supabase.table("rag_messages").delete().in_("chat_id", chat_ids).execute()
        counts["rag_messages"] = len(chat_ids)

    for table in SUPABASE_TABLES:
        result = supabase.table(table).delete().eq("user_id", user_id).execute()
        counts[table] = len(result.data or [])

    return counts


async def _purge_supabase(user_id: str, report: PurgeReport) -> None:
    try:
        counts = await asyncio.to_thread(_purge_supabase_sync, user_id)
        for store, count in counts.items():
            report.record(store, count)
    except Exception as e:
        report.fail("supabase", e)


async def _purge_cache(user_id: str, report: PurgeReport) -> None:
    try:
        from app.services import cache

        await cache.invalidate_user(user_id)
    except Exception as e:
        report.fail("cache", e)


async def purge_user_data(user_id: str) -> PurgeReport:
    """Erase everything `user_id` owns, except the account record itself.

    Separated from account deletion so it can also serve a future "delete my
    data but keep my account" flow, and so the account row's removal stays a
    single obvious statement in `delete_account`.
    """
    report = PurgeReport()

    # Files first: their paths live on rows the Mongo sweep is about to delete.
    await _purge_resume_files(user_id, report)
    await _purge_mongo(user_id, report)
    await _purge_supabase(user_id, report)
    await _purge_cache(user_id, report)

    return report


async def delete_account(user_id: str) -> PurgeReport:
    """Purge the user's data, then the user.

    The account record is removed even when a store failed, because refusing to
    close an account over a transient backend error is not a decision this
    service gets to make on the user's behalf. Failures are logged against the
    id so the leftovers can be found.
    """
    report = await purge_user_data(user_id)

    try:
        result = await get_db()["users"].delete_one({"_id": ObjectId(user_id)})
        report.record("users", result.deleted_count)
    except Exception as e:
        # This one *is* fatal: the whole point of the request was the account.
        report.fail("mongo:users", e)
        raise

    if report.failures:
        logger.error(
            "account %s deleted with incomplete purge; stores still holding data: %s",
            user_id,
            ", ".join(report.failures),
        )
    else:
        logger.info("account %s deleted; %d records removed", user_id, report.total)

    return report
