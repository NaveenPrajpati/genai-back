from typing import TypedDict, Optional, List, Literal, Annotated
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from app.dependencies import get_current_user
from app.core.llm import llm
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from datetime import date, timedelta, datetime, timezone
from langgraph.types import interrupt, Command
import logging
import uuid
from bson import ObjectId
from app.database import get_db

load_dotenv()

logger = logging.getLogger(__name__)


mealRouter = APIRouter(
    prefix="/learning",
    tags=["learning"],
    responses={404: {"description": "Not found"}},
)


class QueryRequest(BaseModel):
    text: str
    roadmapId: Optional[str] = None
    thread_id: Optional[str] = None


class TopicNode(BaseModel):
    id: str
    order: int
    title: str
    description: str
    prerequisites: List[str]
    estimated_hours: Optional[int] = None
    resources: Optional[List[str]] = None
    covered: Optional[bool] = False


class RoadmapOutput(BaseModel):
    id: str
    title: str
    summary: str
    status: Literal["active", "archived", "completed"] = "archived"
    userid: str
    total_estimated_hours: Optional[int] = None
    stages: List[str]
    topics: List[TopicNode]


class Question(BaseModel):
    question: str
    options: list[str]
    answer: int  # index into `options` of the correct choice


class QuizOutput(BaseModel):
    quiz: list[Question]


class LearningState(TypedDict, total=False):
    query: str
    intent: str
    current_user: dict
    userId: str
    thread_id: str
    memory: dict
    plan_status: Optional[str]
    log_status: Optional[str]
    conflict: Optional[dict]
    roadmapId: Optional[str]
    suggestions: Optional[list]
    meal_slots: Optional[list]
    roadmap: Optional[RoadmapOutput]
    roadmap_status: Optional[str]
    next_topic: str
    topic_explaination: str
    quiz: list[dict]
    quizId: str
    active_topic: TopicNode


graph = StateGraph(LearningState)


class IntentOutput(BaseModel):
    intent: str


def get_monday(today: Optional[date] = None) -> str:
    # weekday(): Monday=0, Sunday=6
    if today is None:
        today = date.today()
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


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
                "- modify_roadmap: user wants to change, restructure, or regenerate their roadmap\n"
                "Reply with one word only: create_roadmap, explain, quiz, submit_quiz, find_resources, update_progress, query_roadmap, or modify_roadmap.",
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


class ProgressOutput(BaseModel):
    nextTopic: str
    covedTopic: list[str]
    remainingTopics: list[list]
    topicsToUpdate: list[str]


class TutorOutput(BaseModel):
    topic: str
    topic_explaination: str


class UpdateProgressOutput(BaseModel):
    topic: str
    covered: Optional[bool] = False


async def insertRoadmapToDb(
    roadmap: RoadmapOutput, userId: Optional[str] = None
) -> Optional[str]:
    try:
        doc = roadmap.model_dump()
        doc["userId"] = userId
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
            doc.pop("_id", None)
            return doc
    except Exception as e:
        logger.error("roadmap fetch error: %s", e)
    return None


async def roadmap_agent(state: LearningState):
    is_modify = state.get("intent") == "modify_roadmap"
    existingApproval = await get_db()["approvals"].find_one(
        {"threadId": state.get("thread_id"), "status": "pending"}
    )
    if existingApproval:
        approval_id = str(existingApproval["_id"])
        action_type = existingApproval.get(
            "action", "update_roadmap" if is_modify else "save_roadmap"
        )
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
                        "Existing roadmap:\n{existing_roadmap}",
                    ),
                    ("human", "{text}"),
                ]
            )
            chain = prompt | llm.with_structured_output(RoadmapOutput)
            result: RoadmapOutput = await chain.ainvoke(
                {"text": state["query"], "existing_roadmap": existing_roadmap or "none"}
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
                        "Personalize based on the exact subject in the user query. Be specific and practical.",
                    ),
                    ("human", "{text}"),
                ]
            )
            chain = prompt | llm.with_structured_output(RoadmapOutput)
            result: RoadmapOutput = await chain.ainvoke({"text": state["query"]})
        logger.info("roadmap_agent result: %s", result)

        action_type = "update_roadmap" if is_modify else "save_roadmap"

        approval_id = None
        try:

            res = await get_db()["approvals"].insert_one(
                {
                    "userId": state.get("userId"),
                    "threadId": state.get("thread_id"),
                    "action": action_type,
                    "payload": result.model_dump(),
                    "status": "pending",
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                }
            )
            approval_id = str(res.inserted_id)
            logger.info("roadmap pending approval created: %s", approval_id)
        except Exception as e:
            logger.error("roadmap approval insert error: %s", e)

    # Pause — send roadmap + approval_id to client for review
    decision = interrupt(
        {"type": action_type, "approvalId": approval_id, "roadmap": result.model_dump()}
    )

    if decision != "approved":
        if approval_id:
            try:
                await get_db()["approvals"].update_one(
                    {"_id": ObjectId(approval_id)},
                    {
                        "$set": {
                            "status": "rejected",
                            "resolvedAt": datetime.now(timezone.utc).isoformat(),
                        }
                    },
                )
            except Exception as e:
                logger.error("approval reject update error: %s", e)
        return {"intent": state.get("intent"), "roadmap_status": "rejected"}

    # Approved — update approval status then persist roadmap
    if approval_id:
        try:
            await get_db()["approvals"].update_one(
                {"_id": ObjectId(approval_id)},
                {
                    "$set": {
                        "status": "approved",
                        "resolvedAt": datetime.now(timezone.utc).isoformat(),
                    }
                },
            )
        except Exception as e:
            logger.error("approval approve update error: %s", e)

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
                    "Briefly explain the topic the user wants to know — prefer bullet points.",
                ),
                ("human", "{text}"),
            ]
        )

        chain = findPrompt | llm.with_structured_output(TutorOutput)
        result: TutorOutput = await chain.ainvoke(
            {"text": state["query"], "existing_roadmap": roadmap_title}
        )
        logger.info("query data %s", result)

        return {
            "topic_explaination": result.topic_explaination,
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


async def progress_agent(state: LearningState):
    existing_roadmap = await fetch_roadmap(state.get("roadmapId"))

    if state.get("intent") == "query_roadmap":
        findPrompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert curriculum designer. The user wants to know what to "
                    "study next. Check what the user has already covered in the existing "
                    "roadmap and suggest the next topic to learn, in order.\n"
                    "Existing roadmap:\n{existing_roadmap}",
                ),
                ("human", "{text}"),
            ]
        )

        chain = findPrompt | llm.with_structured_output(ProgressOutput)
        result: ProgressOutput = await chain.ainvoke(
            {"text": state["query"], "existing_roadmap": existing_roadmap or "none"}
        )
        logger.info("query data %s", result)

        return {
            "next_topic": result.nextTopic,
        }
    elif state.get("intent") == "update_progress":
        findPrompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert curriculum designer. Given the user's roadmap, find the "
                    "topic the user is talking about and whether they have completed it "
                    "(covered=True if done).\n"
                    "Existing roadmap:\n{existing_roadmap}",
                ),
                ("human", "{text}"),
            ]
        )

        chain = findPrompt | llm.with_structured_output(UpdateProgressOutput)
        result: UpdateProgressOutput = await chain.ainvoke(
            {"text": state["query"], "existing_roadmap": existing_roadmap or "none"}
        )
        logger.info("query data %s", result)

        updated = False
        roadmapId = state.get("roadmapId")
        if roadmapId and existing_roadmap:
            for topic in existing_roadmap.get("topics", []):
                if topic.get("title") == result.topic:
                    topic["covered"] = result.covered
                    updated = await update_topic(roadmapId, TopicNode(**topic))
                    break

        return {
            "roadmap": existing_roadmap,
            "log_status": "updated" if updated else "not_found",
        }


class ResearchOutput(BaseModel):
    resources: list[str]


async def research_agent(state: LearningState):
    existing_roadmap = await fetch_roadmap(state.get("roadmapId"))
    findPrompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at finding study resources (courses, docs, books, "
                "articles) on a given topic.\n"
                "Existing roadmap:\n{existing_roadmap}",
            ),
            ("human", "{text}"),
        ]
    )

    chain = findPrompt | llm.with_structured_output(ResearchOutput)
    result: ResearchOutput = await chain.ainvoke(
        {"text": state["query"], "existing_roadmap": existing_roadmap or "none"}
    )
    logger.info("research data %s", result)

    active_topic = state.get("active_topic")
    if active_topic and state.get("roadmapId"):
        active_topic.resources = result.resources
        await update_topic(state["roadmapId"], active_topic)

    return {
        "active_topic": active_topic,
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


graph.add_node("load_memory", load_memory)
graph.add_node("classify_intent", classify_intent)
graph.add_node("roadmap_agent", roadmap_agent)
graph.add_node("progress_agent", progress_agent)
graph.add_node("tutor_agent", tutor_agent)
graph.add_node("research_agent", research_agent)
graph.add_edge(START, "classify_intent")
# graph.add_edge("load_memory", "classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    decide_agent,
    ["roadmap_agent", "tutor_agent", "research_agent", "progress_agent", END],
)


@mealRouter.post("/query")
async def ask(
    body: QueryRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.agent

    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    _excluded = {"_id", "expires_at", "password_hash"}
    user_data = {k: v for k, v in current_user.items() if k not in _excluded}
    result = await agent.ainvoke(
        {
            "query": body.text,
            "userId": current_user["uid"],
            "thread_id": thread_id,
            "roadmapId": body.roadmapId,
            "current_user": user_data,
        },
        config=config,
    )
    logger.info("final -- %s", result)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return {
            "status": "needs_approval",
            "thread_id": thread_id,  # app sends this back to /approve
            "proposal": payload,
        }

    return {"status": "done", "result": result}


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["approved", "rejected"]


@mealRouter.post("/approvals")
async def approve(
    body: ApproveRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.agent
    config = {"configurable": {"thread_id": body.thread_id}}

    # The thread/approval must belong to the caller (prevents IDOR where a user
    # approves or rejects someone else's pending plan by guessing the thread_id).
    approval = None
    try:
        approval = await get_db()["approvals"].find_one(
            {"threadId": body.thread_id, "status": "pending"}
        )
        logger.info("approval found: %s", approval)
    except Exception as e:
        logger.error("approval ownership lookup error: %s", e)

    if not approval:
        raise HTTPException(
            status_code=404, detail="No pending approval for this thread."
        )
    if approval["userId"] != current_user["uid"]:
        raise HTTPException(
            status_code=403, detail="You do not have access to this approval."
        )

    snapshot = await agent.aget_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=404,
            detail="No pending approval for this thread. The server may have restarted — please re-submit your plan request.",
        )

    result = await agent.ainvoke(Command(resume=body.decision), config=config)
    return {"status": "done", "result": result}


class Answer(BaseModel):
    question: int  # index of the question in the quiz
    answer: int  # index of the option the user selected


class SubmitQuiz(BaseModel):
    quizId: str
    answers: list[Answer]


@mealRouter.post("/submit-quiz")
async def submit_quiz(
    body: SubmitQuiz, current_user: Annotated[dict, Depends(get_current_user)]
):
    userId = current_user["uid"]
    logger.info("--- %s", userId)
    try:
        quiz = await get_db()["quizzes"].find_one(
            {"_id": ObjectId(body.quizId), "userId": userId}
        )
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found.")

        questions = quiz.get("questions", [])
        selected = {a.question: a.answer for a in body.answers}

        correct = 0
        review = []  # only the questions the user got wrong
        for idx, q in enumerate(questions):
            chosen = selected.get(idx)
            if chosen == q.get("answer"):
                correct += 1
            else:
                review.append(
                    {
                        "question": idx,
                        "selected": chosen,
                        "correctAnswer": q.get("answer"),
                        "correctOption": q.get("options", [])[q.get("answer")]
                        if q.get("answer") is not None
                        else None,
                    }
                )

        return {
            "status": "done",
            "result": {
                "total": len(questions),
                "correct": correct,
                "review": review,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mealRouter.get("/roadmaps")
async def getPlans(current_user: Annotated[dict, Depends(get_current_user)]):
    userId = current_user["uid"]
    logger.info("--- %s", userId)
    try:
        cursor = get_db()["roadmaps"].find({"userId": userId})
        docs = await cursor.to_list(None)
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        logger.info("approvals found: %s", len(docs))

        if not docs:
            return {"status": "done", "message": "roadmaps not found", "result": []}

        return {"status": "done", "message": "roadmaps fetched", "result": docs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class Trigger(BaseModel):
    id: str
    name: str
    schedule: str
    action_type: str
    enabled: bool = True
    last_run_at: Optional[datetime] = None


@mealRouter.post("/toggle-trigger")
async def toggle_trigger(current_user: Annotated[dict, Depends(get_current_user)]):

    try:
        userId = current_user["uid"]

        return {"status": "done"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TopicTipsOutput(BaseModel):
    bullets: list[str]


def active_topic(roadmap: dict) -> Optional[dict]:
    """The next uncovered topic (lowest order) the user is working towards."""
    topics = sorted(roadmap.get("topics", []), key=lambda t: t.get("order", 0))
    for t in topics:
        if not t.get("covered"):
            return t
    return None


async def run_triggers(agent=None):
    """Daily 9am job: for every roadmap, generate a few bullet-point tips about the
    user's current (next uncovered) topic and store them as a learning digest the
    user can fetch later."""
    logger.info("learning digest job running")
    now = datetime.now(timezone.utc)

    tipsPrompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a study coach. Generate 3-5 short, punchy bullet points that "
                "teach or reinforce a key idea about the given topic. Keep each bullet "
                "to a single sentence.",
            ),
            ("human", "Topic: {topic}\nRoadmap context: {summary}"),
        ]
    )
    chain = tipsPrompt | llm.with_structured_output(TopicTipsOutput)

    try:
        cursor = get_db()["roadmaps"].find({})
        roadmaps = await cursor.to_list(None)
    except Exception as e:
        logger.error("run_triggers fetch error: %s", e)
        return

    for roadmap in roadmaps:
        try:
            topic = active_topic(roadmap)
            if not topic:
                continue  # roadmap fully covered — nothing to nudge

            result: TopicTipsOutput = await chain.ainvoke(
                {
                    "topic": topic.get("title", ""),
                    "summary": roadmap.get("summary", ""),
                }
            )

            await get_db()["learning_digests"].insert_one(
                {
                    "userId": roadmap.get("userId"),
                    "roadmapId": str(roadmap["_id"]),
                    "topicId": topic.get("id"),
                    "topicTitle": topic.get("title"),
                    "bullets": result.bullets,
                    "createdAt": now.isoformat(),
                }
            )
            logger.info(
                "learning digest created user=%s topic=%s",
                roadmap.get("userId"),
                topic.get("title"),
            )
        except Exception as e:
            logger.error(
                "learning digest error roadmap=%s: %s", roadmap.get("_id"), e
            )
