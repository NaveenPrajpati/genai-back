"""Daily learning-digest job for the learning-tracker agent."""

import logging
from datetime import datetime, timezone

from langchain_core.prompts import ChatPromptTemplate

from app.core.llm import llm
from app.database import get_db
from app.agents.trigger_store import due_triggers, mark_ran
from app.services.push_service import send_push_notification
from .state import TopicTipsOutput
from .repository import active_topic
from .tools import tavily_search_tool

logger = logging.getLogger(__name__)


async def run_triggers(agent=None):
    """Hourly sweep: for every user who opted in via /toggle-trigger, fire only
    when the current hour matches their chosen schedule_hour in their timezone,
    then generate bullet-point tips (grounded in live Tavily search results)
    about the user's current (next uncovered) topic and store them as a digest."""
    logger.info("learning digest job running")
    now = datetime.now(timezone.utc)

    tipsPrompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a study coach. Using the topic and the web search results, "
                "generate 3-5 short, punchy bullet points that teach or reinforce a key "
                "idea about the topic and point to a useful resource where relevant. "
                "Keep each bullet to a single sentence.",
            ),
            (
                "human",
                "Topic: {topic}\nRoadmap context: {summary}\nSearch results:\n{results}",
            ),
        ]
    )
    chain = tipsPrompt | llm.with_structured_output(TopicTipsOutput)

    try:
        triggers = await due_triggers("learning_digest", now)
    except Exception as e:
        logger.error("run_triggers trigger fetch error: %s", e)
        return

    for trig in triggers:
        userId = trig.get("userId")
        try:
            cursor = get_db()["roadmaps"].find({"userId": userId})
            roadmaps = await cursor.to_list(None)
        except Exception as e:
            logger.error("run_triggers roadmap fetch error user=%s: %s", userId, e)
            continue

        for roadmap in roadmaps:
            try:
                topic = active_topic(roadmap)
                if not topic:
                    continue

                topic_title = topic.get("title", "")

                results = []
                try:
                    search = await tavily_search_tool.ainvoke(
                        {"query": f"best free learning resources for {topic_title}"}
                    )
                    results = (
                        search.get("results", [])
                        if isinstance(search, dict)
                        else search
                    )
                except Exception as e:
                    logger.error("tavily digest search error: %s", e)

                result: TopicTipsOutput = await chain.ainvoke(
                    {
                        "topic": topic_title,
                        "summary": roadmap.get("summary", ""),
                        "results": results,
                    }
                )

                resources = [
                    {"title": r.get("title"), "url": r.get("url")}
                    for r in results
                    if isinstance(r, dict)
                ]

                await get_db()["learning_digests"].insert_one(
                    {
                        "userId": userId,
                        "roadmapId": str(roadmap["_id"]),
                        "topicId": topic.get("id"),
                        "topicTitle": topic_title,
                        "bullets": result.bullets,
                        "resources": resources,
                        "createdAt": now.isoformat(),
                    }
                )
                logger.info(
                    "learning digest created user=%s topic=%s", userId, topic_title
                )

                await send_push_notification(
                    userId,
                    title=f"Today's tips: {topic_title}",
                    body=result.bullets[0] if result.bullets else "Your daily learning digest is ready.",
                    data={"type": "learning_digest", "topicId": topic.get("id")},
                )
            except Exception as e:
                logger.error(
                    "learning digest error roadmap=%s: %s", roadmap.get("_id"), e
                )

        # Record when this user's digest last ran.
        try:
            await mark_ran(trig, now)
        except Exception as e:
            logger.error("trigger last_run update error user=%s: %s", userId, e)
