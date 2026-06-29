"""MongoDB persistence and memory helpers for the learning-tracker agent."""

import logging
from datetime import date, timedelta, datetime, timezone
from typing import Optional

from bson import ObjectId

from app.database import get_db
from app.agents.memory_store import extract_and_save
from .state import RoadmapOutput, TopicNode, MemoryExtract

logger = logging.getLogger(__name__)


def get_monday(today: Optional[date] = None) -> str:
    # weekday(): Monday=0, Sunday=6
    if today is None:
        today = date.today()
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


def active_topic(roadmap: dict) -> Optional[dict]:
    """The next uncovered topic (lowest order) the user is working towards."""
    topics = sorted(roadmap.get("topics", []), key=lambda t: t.get("order", 0))
    for t in topics:
        if not t.get("covered"):
            return t
    return None


async def insertRoadmapToDb(
    roadmap: RoadmapOutput, user_id: Optional[str] = None
) -> Optional[str]:
    try:
        doc = roadmap.model_dump()
        doc["user_id"] = user_id
        doc["createdAt"] = datetime.now(timezone.utc).isoformat()
        res = await get_db()["roadmaps"].insert_one(doc)
        logger.info("insertRoadmapToDb inserted: %s", res.inserted_id)
        return str(res.inserted_id)
    except Exception as e:
        logger.error("insertRoadmapToDb error: %s", e)
        return None


async def fetch_roadmap(roadmapId: Optional[str]) -> Optional[dict]:
    if not roadmapId:
        return None
    try:
        doc = await get_db()["roadmaps"].find_one({"_id": ObjectId(roadmapId)})
        if doc:
            doc["_id"] = str(doc["_id"])
            return doc
    except Exception as e:
        logger.error("roadmap fetch error: %s", e)
    return None


async def update_topic(roadmapId: str, topic: TopicNode) -> bool:
    try:
        res = await get_db()["roadmaps"].update_one(
            {"_id": ObjectId(roadmapId), "topics.id": topic.id},
            {"$set": {"topics.$": topic.model_dump()}},
        )
        logger.info(
            "update_topic matched=%s modified=%s",
            res.matched_count,
            res.modified_count,
        )
        return res.modified_count > 0
    except Exception as e:
        logger.error("update_topic error: %s", e)
        return False


async def set_topic_covered(
    roadmapId: str, topicId: str, covered: bool, user_id: Optional[str] = None
) -> bool:
    """Flip a single topic's `covered` flag via a targeted positional update — no
    LLM, no full-document rewrite. Optionally scope to user_id for ownership."""
    query = {"_id": ObjectId(roadmapId), "topics.id": topicId}
    if user_id:
        query["user_id"] = user_id
    try:
        res = await get_db()["roadmaps"].update_one(
            query, {"$set": {"topics.$.covered": covered}}
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error("set_topic_covered error: %s", e)
        return False


_LEARNER_MEMORY_INSTRUCTIONS = (
    "Extract durable facts about the learner from their message — skill level "
    "(beginner/intermediate/advanced), preferred resource types "
    "(video/text/interactive), learning goals, weekly availability, and topics "
    "they already know."
)


async def write_memory(user_id: str, query: str, current: Optional[dict] = None):
    """Background task: pull durable learner facts out of the latest message and
    merge them into the user's memory doc. Runs after the response is sent, so it
    adds no latency to /query. Delegates to the shared memory store."""
    await extract_and_save(
        user_id, query, MemoryExtract, _LEARNER_MEMORY_INSTRUCTIONS, current
    )
