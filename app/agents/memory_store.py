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

# Sub-document holding an agent's learned facts. Each agent passes its own so a
# richer per-agent schema can't overwrite another agent's fields; agents that
# never opted into namespacing stay on the original shared `data` field.
DEFAULT_NS = "data"


async def get_profile(user_id: str, namespace: str = DEFAULT_NS) -> dict:
    """Return the user's learned-facts dict for `namespace`, or {} if none."""
    try:
        doc = await get_db()[MEMORIES].find_one({"user_id": user_id})
        if doc:
            return doc.get(namespace, {}) or {}
    except Exception as e:
        logger.error("get_profile error user=%s: %s", user_id, e)
    return {}


async def save_profile(
    user_id: str, updates: dict, namespace: str = DEFAULT_NS
) -> None:
    """Merge known-good facts straight into the profile, no extraction step.
    For values the user stated outright — an onboarding answer, a settings edit —
    rather than something a model inferred from prose."""
    if not updates:
        return
    now = datetime.now(timezone.utc).isoformat()
    try:
        await get_db()[MEMORIES].update_one(
            {"user_id": user_id},
            {
                "$set": {
                    **{f"{namespace}.{k}": v for k, v in updates.items()},
                    "updatedAt": now,
                },
                "$setOnInsert": {"createdAt": now},
            },
            upsert=True,
        )
    except Exception as e:
        logger.error("save_profile error user=%s: %s", user_id, e)


async def extract_and_save(
    user_id: str,
    text: str,
    schema: Type[BaseModel],
    instructions: str,
    current: Optional[dict] = None,
    namespace: str = DEFAULT_NS,
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

    await save_profile(user_id, updates, namespace)
    logger.info("memory updated user=%s fields=%s", user_id, list(updates))
