"""
services/storage.py
===================
All Supabase reads/writes in one place: ingestion logs, chats, and messages.

Keeping persistence here (instead of inline in routes) means the routes stay
readable and you could swap Supabase for Postgres/Mongo by editing one file.
"""

from datetime import datetime, timezone
from typing import List

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


def create_chat(title: str) -> str:
    """Insert a new chat and return its id."""
    now = _now()
    row = (
        supabase.table("rag_chats")
        .insert({"title": title[:200], "created_at": now, "updated_at": now})
        .execute()
    )
    return row.data[0]["id"]


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
) -> None:
    """Insert the user + assistant message pair and bump the chat's updated_at."""
    now = _now()
    if chat_id:
        supabase.table("rag_chats").update(
            {"updated_at": now, "title": question[:50]}
        ).eq("id", chat_id).execute()
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
