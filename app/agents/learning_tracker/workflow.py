"""LangGraph nodes and graph wiring for the learning-tracker agent."""

import logging
from datetime import datetime, timezone

from bson import ObjectId
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langgraph.types import interrupt

from app.core.llm import llm
from app.database import get_db
from app.agents.approval_store import get_pending, create_pending, resolve
from .state import (
    LearningState,
    RoadmapOutput,
    IntentOutput,
    QuizOutput,
    UpdateProgressOutput,
    ResearchOutput,
)
from .repository import (
    fetch_roadmap,
    insertRoadmapToDb,
    update_topic,
    set_topic_covered,
    active_topic,
)
from .tools import tavily_search_tool

logger = logging.getLogger(__name__)


async def classify_intent(state: LearningState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the user message into one intent:\n"
                "- create_roadmap: user asks what to study \n"
                "- explain: user asks for an explanation of a concept, topic, or step\n"
                "- quiz: user requests a quiz or test on a topic\n"
                "- submit_quiz: user is submitting answers to a quiz for evaluation\n"
                "- find_resources: user asks for resources, links, books, or materials on a topic\n"
                "- update_progress: user marks progress, completes a step, or logs learning activity\n"
                "- query_roadmap: user asks what to do next in their learning roadmap or user wants to view or check the current state of their roadmap\n"
                "- modify_roadmap: user wants to change, restructure, or regenerate their roadmap\n",
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | llm.with_structured_output(IntentOutput)
    result: IntentOutput = await chain.ainvoke({"text": state.get("query", "")})
    logger.info("%s", result)
    return {
        "intent": result.intent,
    }


async def roadmap_agent(state: LearningState):
    is_modify = state.get("intent") == "modify_roadmap"
    action_type = "update_roadmap" if is_modify else "save_roadmap"
    existingApproval = await get_pending(state.get("thread_id"))
    if existingApproval:
        approval_id = str(existingApproval["_id"])
        result = RoadmapOutput(**existingApproval["payload"])
        logger.info("roadmap approval already exists: %s", approval_id)
    else:
        if is_modify:
            # Fetch the existing roadmap so the LLM can operate on it
            existing_roadmap = await fetch_roadmap(state.get("roadmapId"))

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert curriculum designer. The user wants to modify an existing learning roadmap.\n"
                        "Apply the requested change (add topic, remove topic, reorder, adjust hours, update resources, etc.).\n"
                        "Return the full updated roadmap — keep all unchanged topics intact.\n"
                        "Maintain correct order values and prerequisite links after any structural change.\n"
                        "Existing roadmap:\n{existing_roadmap}\n"
                        "Learner profile (use to tailor depth, pacing, and resources):\n{memory}",
                    ),
                    ("human", "{text}"),
                ]
            )
            chain = prompt | llm.with_structured_output(RoadmapOutput)
            result: RoadmapOutput = await chain.ainvoke(
                {
                    "text": state["query"],
                    "existing_roadmap": existing_roadmap or "none",
                    "memory": state.get("memory") or "none",
                }
            )
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert curriculum designer and learning path architect.\n"
                        "Given a topic the user wants to learn, produce a complete, sequenced roadmap:\n"
                        "1. Break the subject into ordered topics (order field starts at 1).\n"
                        "2. For each topic list its prerequisites by title — only topics that appear earlier in the list.\n"
                        "3. Group topics into broad stages (e.g. Foundations, Intermediate, Advanced).\n"
                        "4. Estimate realistic study hours per topic and a total.\n"
                        "5. Suggest 1-2 free learning resources (course names, docs, book titles) per topic.\n"
                        "Personalize based on the exact subject in the user query. Be specific and practical.\n"
                        "Learner profile (use to tailor depth, pacing, and resources):\n{memory}",
                    ),
                    ("human", "{text}"),
                ]
            )
            chain = prompt | llm.with_structured_output(RoadmapOutput)
            result: RoadmapOutput = await chain.ainvoke(
                {"text": state["query"], "memory": state.get("memory") or "none"}
            )
        logger.info("roadmap_agent result: %s", result)

        approval_id = await create_pending(
            state.get("userId"),
            state.get("thread_id"),
            action_type,
            result.model_dump(),
        )
        logger.info("roadmap pending approval created: %s", approval_id)

    # Pause — send roadmap + approval_id to client for review
    decision = interrupt(
        {"type": action_type, "approvalId": approval_id, "roadmap": result.model_dump()}
    )

    if decision != "approved":
        await resolve(approval_id, "rejected")
        return {"intent": state.get("intent"), "roadmap_status": "rejected"}

    # Approved — update approval status then persist roadmap
    await resolve(approval_id, "approved")

    if is_modify and state.get("roadmapId"):
        try:
            await get_db()["roadmaps"].replace_one(
                {"_id": ObjectId(state["roadmapId"])},
                {
                    **result.model_dump(),
                    "userId": state.get("userId"),
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                },
            )
            saved_roadmapId = state["roadmapId"]
        except Exception as e:
            logger.error("roadmap update error: %s", e)
            saved_roadmapId = None
    else:
        saved_roadmapId = await insertRoadmapToDb(result, state.get("userId"))

    return {
        "intent": state.get("intent"),
        "roadmap_status": "approved",
        "roadmapId": saved_roadmapId,
        "roadmap": result.model_dump(),
    }


async def tutor_agent(state: LearningState):
    roadmap = await fetch_roadmap(state.get("roadmapId"))
    roadmap_title = roadmap.get("title") if roadmap else "general"

    if state.get("intent") == "explain":
        findPrompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in: {existing_roadmap}\n"
                    "Briefly explain the topic the user wants to know — prefer bullet points.\n"
                    "Pitch the explanation to the learner's profile:\n{memory}",
                ),
                ("human", "{text}"),
            ]
        )

        # Plain text (no structured output) so the tokens can stream as they are
        # generated — see the /learning/query/stream route. The full text is also
        # returned, so the non-streaming /query route keeps working unchanged.
        chain = findPrompt | llm
        response = await chain.ainvoke(
            {
                "text": state["query"],
                "existing_roadmap": roadmap_title,
                "memory": state.get("memory") or "none",
            }
        )
        logger.info("query data %s", response.content)

        return {
            "topic_explaination": response.content,
        }
    elif state.get("intent") == "quiz":
        findPrompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in: {existing_roadmap}\n"
                    "Generate a short multiple-choice quiz on the topic the user asks about.",
                ),
                ("human", "{text}"),
            ]
        )

        chain = findPrompt | llm.with_structured_output(QuizOutput)
        result: QuizOutput = await chain.ainvoke(
            {"text": state["query"], "existing_roadmap": roadmap_title}
        )
        logger.info("query data %s", result)

        # Persist the full quiz (including correct answers) so we can grade a later
        # submission. Only the answer-free version is returned to the client.
        quizId = None
        try:
            res = await get_db()["quizzes"].insert_one(
                {
                    "userId": state.get("userId"),
                    "roadmapId": state.get("roadmapId"),
                    "questions": [q.model_dump() for q in result.quiz],
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                }
            )
            quizId = str(res.inserted_id)
        except Exception as e:
            logger.error("quiz insert error: %s", e)

        public_quiz = [
            {"question": q.question, "options": q.options} for q in result.quiz
        ]
        return {
            "quiz": public_quiz,
            "quizId": quizId,
        }


async def progress_agent(state: LearningState):
    existing_roadmap = await fetch_roadmap(state.get("roadmapId"))

    if state.get("intent") == "query_roadmap":
        # Pure data work — what's next and how far along is fully determined by the
        # `covered` / `order` fields. No LLM: it can't hallucinate or miscount.
        topics = existing_roadmap.get("topics", []) if existing_roadmap else []
        covered = [t for t in topics if t.get("covered")]
        nxt = active_topic(existing_roadmap) if existing_roadmap else None
        total = len(topics)
        progress = {
            "next_topic": nxt.get("title") if nxt else None,
            "next_topic_id": nxt.get("id") if nxt else None,
            "covered_count": len(covered),
            "remaining": total - len(covered),
            "total": total,
            "percent": round(len(covered) / total * 100) if total else 0,
        }
        return {
            "next_topic": progress["next_topic"] or "",
            "progress": progress,
        }

    elif state.get("intent") == "update_progress":
        roadmapId = state.get("roadmapId")
        topics = existing_roadmap.get("topics", []) if existing_roadmap else []
        if not roadmapId or not topics:
            return {"roadmap": existing_roadmap, "log_status": "not_found"}

        # LLM does only the fuzzy part: map the user's wording to one of the actual
        # topic ids. The state change itself is deterministic.
        topic_list = "\n".join(f"- {t.get('id')}: {t.get('title')}" for t in topics)
        findPrompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "The user is reporting which topic they have completed. Match their "
                    "message to exactly one topic from the list and return its id. "
                    "Return null if none clearly matches.\n"
                    "Topics (id: title):\n{topic_list}",
                ),
                ("human", "{text}"),
            ]
        )
        chain = findPrompt | llm.with_structured_output(UpdateProgressOutput)
        result: UpdateProgressOutput = await chain.ainvoke(
            {"text": state["query"], "topic_list": topic_list}
        )
        logger.info("update_progress matched id=%s", result.topicId)

        valid_ids = {t.get("id") for t in topics}
        if result.topicId not in valid_ids:
            return {"roadmap": existing_roadmap, "log_status": "not_found"}

        updated = await set_topic_covered(
            roadmapId, result.topicId, True, userId=state.get("userId")
        )
        if updated:
            for t in topics:
                if t.get("id") == result.topicId:
                    t["covered"] = True
                    break

        return {
            "roadmap": existing_roadmap,
            "log_status": "updated" if updated else "not_found",
        }


async def research_agent(state: LearningState):
    active = state.get("active_topic")
    topic = (active.title if active else None) or state["query"]

    results = []
    try:
        search = await tavily_search_tool.ainvoke(
            {"query": f"best free learning resources for {topic}"}
        )
        results = search.get("results", []) if isinstance(search, dict) else search
    except Exception as e:
        logger.error("tavily search error: %s", e)

    # Distill the raw hits into a clean resource list (single structured pass).
    findPrompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at curating study resources. From the search "
                "results, pick the most useful courses, docs, books, and articles "
                "for the topic. Return each as a short label followed by its URL.\n"
                "Topic: {topic}",
            ),
            ("human", "Search results:\n{results}"),
        ]
    )
    chain = findPrompt | llm.with_structured_output(ResearchOutput)
    result: ResearchOutput = await chain.ainvoke({"topic": topic, "results": results})
    logger.info("research data %s", result)

    if active and state.get("roadmapId"):
        active.resources = result.resources
        await update_topic(state["roadmapId"], active)

    return {
        "active_topic": active,
        "suggestions": result.resources,
    }


async def load_memory(state: LearningState):
    userId = state["userId"]
    memory = {}

    try:
        result = await get_db()["memories"].find_one({"userId": userId})
        if result:
            memory = result.get("data", {})
    except Exception as e:
        logger.error("load memory error: %s", e)

    return {"memory": memory}


def decide_agent(state: LearningState):
    intent = state.get("intent")
    if intent in ("create_roadmap", "modify_roadmap"):
        return "roadmap_agent"
    elif intent in ("explain", "quiz", "submit_quiz"):
        return "tutor_agent"
    elif intent == "find_resources":
        return "research_agent"
    elif intent in ("update_progress", "query_roadmap"):
        return "progress_agent"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(LearningState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("roadmap_agent", roadmap_agent)
    graph.add_node("progress_agent", progress_agent)
    graph.add_node("tutor_agent", tutor_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        decide_agent,
        ["roadmap_agent", "tutor_agent", "research_agent", "progress_agent", END],
    )
    return graph


graph = build_graph()
