"""
services/rag/storage.py
===================
All Supabase reads/writes in one place: ingestion logs, chats, and messages.

Keeping persistence here (instead of inline in routes) means the routes stay
readable and you could swap Supabase for Postgres/Mongo by editing one file.
"""

from datetime import datetime, timezone
from typing import List, Optional

from app.core.config import supabase


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Ingestion logs ───────────────────────────────────────────────────────────


def create_ingestion_log(
    doc_id: str, source: str, file_type: str, ingested_at: str, user_id: str
) -> None:
    supabase.table("rag_ingestion_logs").insert(
        {
            "doc_id": doc_id,
            "source": source,
            "file_type": file_type,
            "status": "queued",
            "ingested_at": ingested_at,
            "user_id": user_id,
        }
    ).execute()


def update_ingestion_log(doc_id: str, updates: dict) -> None:
    supabase.table("rag_ingestion_logs").update(updates).eq("doc_id", doc_id).execute()


def get_ingestion_log(doc_id: str):
    """Return the ingestion log row for `doc_id`, or None if it doesn't exist."""
    rows = (
        supabase.table("rag_ingestion_logs")
        .select("*")
        .eq("doc_id", doc_id)
        .execute()
        .data
    )
    return rows[0] if rows else None


def delete_ingestion_log(doc_id: str) -> None:
    supabase.table("rag_ingestion_logs").delete().eq("doc_id", doc_id).execute()


def list_ingestion_logs(user_id) -> list:
    return (
        supabase.table("rag_ingestion_logs")
        .select("*")
        .order("ingested_at", desc=True)
        .eq("user_id", user_id)
        .execute()
        .data
    )


def create_chat(title: str, user_id: Optional[str] = None) -> dict:
    """Insert a new chat and return the full inserted row.

    Returns the whole record (not just the id) so callers can hand the client a
    complete chat object without a follow-up read.

    `user_id` is optional only for backwards compatibility with the bare
    POST /chat route; without it the row is invisible to `list_chats`, which
    filters by user_id.
    """
    now = _now()
    record = {"title": title[:200], "created_at": now, "updated_at": now}
    if user_id:
        record["user_id"] = user_id
    row = supabase.table("rag_chats").insert(record).execute()
    return row.data[0]


def list_chats(user_id) -> list:
    try:
        return (
            supabase.table("rag_chats")
            .select("*")
            .order("updated_at", desc=True)
            .eq("user_id", user_id)
            .execute()
            .data
        )
    except Exception as e:
        print(f"Query failed: {e}")


def get_chat(chat_id: str):
    return (
        supabase.table("rag_chats")
        .select("id, title, created_at, updated_at")
        .eq("id", chat_id)
        .single()
        .execute()
        .data
    )


def delete_chat(chat_id: str) -> None:
    supabase.table("rag_messages").delete().eq("chat_id", chat_id).execute()
    supabase.table("rag_chats").delete().eq("id", chat_id).execute()


def get_messages(chat_id: str) -> list:
    return (
        supabase.table("rag_messages")
        .select("*")
        .eq("chat_id", chat_id)
        .order("created_at", desc=False)
        .execute()
        .data
    )


def save_messages(
    chat_id: str,
    question: str,
    answer: str,
    user_id: str,
    sources: list,
    ingestions: List[str],
) -> str:
    """Insert the user + assistant message pair and bump the chat's updated_at.

    Returns the chat id the turn was written to — callers MUST report this back
    to the client when they passed no chat_id, or the client can never continue
    the chat it just started and every question opens a new one.
    """
    now = _now()
    if chat_id:
        # Only bump the timestamp: the title is set from the FIRST question when
        # the chat is created. Rewriting it here made an existing chat's title
        # change to whatever was asked last, so it never looked like one thread.
        supabase.table("rag_chats").update({"updated_at": now}).eq(
            "id", chat_id
        ).execute()
    else:
        row = (
            supabase.table("rag_chats")
            .insert(
                {
                    "user_id": user_id,
                    "updated_at": now,
                    "created_at": now,
                    "title": question[:50],
                }
            )
            .execute()
        )

        chat_id = row.data[0]["id"]

    supabase.table("rag_messages").insert(
        [
            {
                "chat_id": chat_id,
                "role": "user",
                "content": question,
                "ingestions": ingestions,
                "created_at": now,
            },
            {
                "chat_id": chat_id,
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "created_at": now,
            },
        ]
    ).execute()
    return chat_id
