"""Shared long-term user-memory store (MongoDB).

Durable per-user facts that personalize all three agents, learned automatically
from conversation. Generalizes the learning-tracker's original `write_memory`.

Design:
- One Mongo doc per user in the `memories` collection: `{user_id, data: {...},
  createdAt, updatedAt}`. `data` is a flat dict of learned facts; each agent
  contributes its own (distinctly-named) fields via a Pydantic extraction schema.
- Read (`get_profile`) is merged into each agent's existing `memory` dict — it
  layers on top of the per-agent Supabase prefs rather than replacing them, so
  nothing currently working breaks. (Full Supabase→Mongo consolidation is a
  separate future step.)
- Write (`extract_and_save`) runs as a fire-and-forget background task after the
  response is sent, so it adds no latency to /query.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Type

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from app.core.llm import fast_llm
from app.database import get_db

logger = logging.getLogger(__name__)

MEMORIES = "memories"


async def get_profile(user_id: str) -> dict:
    """Return the user's learned-facts dict (the `data` field), or {} if none."""
    try:
        doc = await get_db()[MEMORIES].find_one({"user_id": user_id})
        if doc:
            return doc.get("data", {}) or {}
    except Exception as e:
        logger.error("get_profile error user=%s: %s", user_id, e)
    return {}


async def extract_and_save(
    user_id: str,
    text: str,
    schema: Type[BaseModel],
    instructions: str,
    current: Optional[dict] = None,
) -> None:
    """Pull durable facts out of `text` using `schema`, then merge them into the
    user's memory doc. `instructions` is the schema-specific extraction guidance;
    a shared rubric (only fill on clear evidence, don't invent/restate) is
    appended. Designed to run via FastAPI BackgroundTasks — never raises.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                instructions
                + "\nOnly fill a field when the message gives clear evidence; "
                "otherwise leave it null. Do not invent or restate the existing "
                "profile.\nKnown so far:\n{current}",
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | fast_llm.with_structured_output(schema)
    try:
        extracted: BaseModel = await chain.ainvoke(
            {"text": text, "current": current or "none"}
        )
    except Exception as e:
        logger.error("extract_and_save extract error user=%s: %s", user_id, e)
        return

    # Keep only the fields the model actually filled in.
    updates = {
        k: v for k, v in extracted.model_dump().items() if v not in (None, [], "")
    }
    if not updates:
        return

    now = datetime.now(timezone.utc).isoformat()
    set_doc = {f"data.{k}": v for k, v in updates.items()}
    set_doc["updatedAt"] = now
    try:
        await get_db()[MEMORIES].update_one(
            {"user_id": user_id},
            {"$set": set_doc, "$setOnInsert": {"createdAt": now}},
            upsert=True,
        )
        logger.info("memory updated user=%s fields=%s", user_id, list(updates))
    except Exception as e:
        logger.error("extract_and_save upsert error user=%s: %s", user_id, e)
