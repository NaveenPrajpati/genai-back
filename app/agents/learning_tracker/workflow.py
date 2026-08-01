"""LangGraph nodes and graph wiring for the learning-tracker agent."""

import logging
from datetime import datetime, timezone

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from app.core.llm import llm, fast_llm
from app.database import get_db
from app.agents.approval_store import get_pending, create_pending, resolve
from app.agents.react import run_tool_loop
from app.agents.memory_store import get_profile, save_profile
from .service import build_roadmap, revise_roadmap, sync_roadmap_to_pa
from .state import (
    LearningState,
    LearnerMemory,
    ONBOARDED_FLAG,
    ONBOARDING_KEYS,
    RoadmapDraft,
    IntentOutput,
    QuizOutput,
    QuizSubmission,
    UpdateProgressOutput,
    ResearchOutput,
)
from .repository import (
    LEARNING_NS,
    active_topic,
    fetch_quiz,
    fetch_roadmap,
    grade_quiz,
    insertRoadmapToDb,
    materialize_roadmap,
    merge_roadmap,
    record_quiz_attempt,
    replace_roadmap,
    resolve_roadmap_id,
    roadmap_progress,
    set_topic_progress,
    set_topic_resources,
)
from .tools import research_llm, research_tool_node
from app.services.cache import cached_value
from app.core.config import CACHE_CLASSIFY_THRESHOLD

logger = logging.getLogger(__name__)

# The approval kinds this agent raises. A supervisor thread carries approvals
# from every skill, so the roadmap node must not pick up the PA's pending task
# deletion and try to read it as a roadmap.
ROADMAP_ACTIONS = ["save_roadmap", "update_roadmap"]


async def classify_intent(state: LearningState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the user message into one intent:\n"
                "- create_roadmap: user asks what to study, or wants a new learning plan/roadmap built for a subject\n"
                "- explain: user asks for an explanation of a concept, topic, or step\n"
                "- quiz: user requests a quiz or test on a topic\n"
                "- submit_quiz: user is submitting answers to a quiz for evaluation\n"
                "- find_resources: user asks for resources, links, books, or materials on a topic\n"
                "- update_progress: user marks progress, completes a step, or logs learning activity\n"
                "- query_roadmap: user asks what to do next in their roadmap, or wants to view/check its current state\n"
                "- modify_roadmap: user wants to change, restructure, or regenerate their roadmap\n"
                "- chitchat: greetings, small talk, or meta questions about what this agent can do\n"
                "- fallback: the message is unclear, ambiguous, or outside the scope of learning/studying\n"
                "When unsure between a real action and fallback, prefer fallback only if no learning action clearly fits.\n",
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | fast_llm.with_structured_output(IntentOutput)
    query = state.get("query", "")

    async def produce():
        result: IntentOutput = await chain.ainvoke({"text": query})
        logger.info("%s", result)
        return result.model_dump()

    # Intent depends only on the message and is user-independent → global scope,
    # shared across users, with the loose classification threshold.
    data = await cached_value(
        query, "agent:learning:classify_intent", CACHE_CLASSIFY_THRESHOLD, produce
    )
    return {"intent": data["intent"]}


async def roadmap_agent(state: LearningState):
    user_id = state.get("user_id")
    is_modify = state.get("intent") == "modify_roadmap"
    action_type = "update_roadmap" if is_modify else "save_roadmap"

    existing_roadmap = None
    if is_modify:
        existing_roadmap = await fetch_roadmap(state.get("roadmapId"), user_id)
        if not existing_roadmap:
            # Nothing of the learner's to modify. Bail rather than quietly
            # creating a second roadmap they didn't ask for.
            return {"intent": state.get("intent"), "roadmap_status": "not_found"}

    # Re-run safe: LangGraph replays this node on resume, so reuse the proposal
    # already on this thread instead of paying for a second generation.
    existing_approval = await get_pending(state.get("thread_id"), ROADMAP_ACTIONS)
    if existing_approval:
        approval_id = str(existing_approval["_id"])
        draft = RoadmapDraft(**existing_approval["payload"])
        logger.info("roadmap approval already exists: %s", approval_id)
    else:
        if is_modify:
            draft = await revise_roadmap(
                state["query"], existing_roadmap, state.get("memory")
            )
        else:
            draft = await build_roadmap(state["query"], state.get("memory"))
        logger.info("roadmap_agent draft: %s", draft.title)

        approval_id = await create_pending(
            user_id, state.get("thread_id"), action_type, draft.model_dump()
        )
        logger.info("roadmap pending approval created: %s", approval_id)

    # Pause — send the proposal + approval_id to the client for review.
    decision = interrupt(
        {"type": action_type, "approvalId": approval_id, "roadmap": draft.model_dump()}
    )

    if decision != "approved":
        await resolve(approval_id, "rejected")
        return {"intent": state.get("intent"), "roadmap_status": "rejected"}

    if is_modify:
        # Merge onto the stored document so the learner keeps the progress they
        # already made on topics that survived the edit.
        roadmap = merge_roadmap(existing_roadmap, draft)
        saved = await replace_roadmap(state["roadmapId"], user_id, roadmap)
        saved_roadmapId = state["roadmapId"] if saved else None
    else:
        roadmap = materialize_roadmap(draft)
        saved_roadmapId = await insertRoadmapToDb(roadmap, user_id)

    if not saved_roadmapId:
        # Never report a save we didn't make. The approval stays pending, so a
        # retry reuses this proposal instead of regenerating (and re-charging).
        logger.error("roadmap persist failed thread=%s", state.get("thread_id"))
        return {"intent": state.get("intent"), "roadmap_status": "save_failed"}

    await resolve(approval_id, "approved")

    # Cross-agent collaboration: hand each roadmap topic to the personal
    # assistant as a tracked to-do, so study steps show up in the user's agenda.
    # Provenance + source_ref dedup makes this safe on roadmap modifications.
    pa_tasks_created = await sync_roadmap_to_pa(user_id, roadmap, saved_roadmapId)

    return {
        "intent": state.get("intent"),
        "roadmap_status": "approved",
        "roadmapId": saved_roadmapId,
        "roadmap": roadmap.model_dump(),
        "pa_tasks_created": pa_tasks_created,
    }


async def tutor_agent(state: LearningState):
    roadmap = await fetch_roadmap(state.get("roadmapId"), state.get("user_id"))
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
    # submission. Only the answer-free version is returned to the client. The
    # topic it was taken against is stamped on so a score can be attributed.
    current = active_topic(roadmap) if roadmap else None
    quizId = None
    try:
        res = await get_db()["quizzes"].insert_one(
            {
                "user_id": state.get("user_id"),
                "roadmapId": state.get("roadmapId"),
                "topicId": (current or {}).get("id"),
                "questions": [q.model_dump() for q in result.quiz],
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }
        )
        quizId = str(res.inserted_id)
    except Exception as e:
        logger.error("quiz insert error: %s", e)

    public_quiz = [{"question": q.question, "options": q.options} for q in result.quiz]
    return {
        "quiz": public_quiz,
        "quizId": quizId,
    }


async def quiz_grader_agent(state: LearningState):
    """Grades a quiz the learner answers in chat ("1-b, 2-c, 3-a").

    The stored quiz is the source of truth for correctness; the LLM only reads
    the learner's *choices* out of free text. Grading itself is arithmetic, so a
    model that misreads an answer cannot also mark it correct.
    """
    user_id = state.get("user_id")
    quiz = await fetch_quiz(user_id, state.get("quizId"))
    if not quiz:
        return {"quiz_result": None, "log_status": "quiz_not_found"}

    questions = quiz.get("questions") or []
    # Rendered WITHOUT the answer key: the model's only job is to read the
    # learner's choices, and showing it the answers invites helpful "corrections".
    rendered = "\n".join(
        f"{i}. {q.get('question')}\n"
        + "\n".join(f"   {j}) {opt}" for j, opt in enumerate(q.get("options") or []))
        for i, q in enumerate(questions)
    )
    chain = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "The learner is submitting answers to this quiz. Map their message "
                "to the questions below and return, for each answer you can "
                "identify, the 0-based question index and the 0-based index of the "
                "option they chose. Letters map to positions (a=0, b=1, …). Omit "
                "any question they did not answer — never guess.\n"
                "Quiz:\n{questions}",
            ),
            ("human", "{text}"),
        ]
    ) | fast_llm.with_structured_output(QuizSubmission)
    submission: QuizSubmission = await chain.ainvoke(
        {"text": state["query"], "questions": rendered}
    )

    # Drop anything out of range rather than letting it skew the score.
    selected = {
        a.question_index: a.selected_index
        for a in submission.answers
        if a.question_index < len(questions)
        and a.selected_index < len(questions[a.question_index].get("options") or [])
    }
    result = grade_quiz(questions, selected)
    logger.info(
        "quiz graded user=%s %s/%s", user_id, result["correct"], result["total"]
    )

    await record_quiz_attempt(
        user_id, str(quiz["_id"]), quiz.get("roadmapId"), quiz.get("topicId"), result
    )
    return {"quiz_result": result, "quizId": str(quiz["_id"])}


async def progress_agent(state: LearningState):
    user_id = state.get("user_id")
    existing_roadmap = await fetch_roadmap(state.get("roadmapId"), user_id)

    if state.get("intent") == "query_roadmap":
        # Pure data work — what's next and how far along is fully determined by
        # `progress_status` / `order`. No LLM: it can't hallucinate or miscount.
        progress = roadmap_progress(existing_roadmap)
        return {
            "next_topic": progress["next_topic"] or "",
            "progress": progress,
            "log_status": "listed" if existing_roadmap else "no_roadmap",
        }

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
    chain = findPrompt | fast_llm.with_structured_output(UpdateProgressOutput)
    result: UpdateProgressOutput = await chain.ainvoke(
        {"text": state["query"], "topic_list": topic_list}
    )
    logger.info("update_progress matched id=%s", result.topicId)

    if result.topicId not in {t.get("id") for t in topics}:
        return {"roadmap": existing_roadmap, "log_status": "not_found"}

    updated = await set_topic_progress(roadmapId, result.topicId, "completed", user_id)
    if updated:
        for t in topics:
            if t.get("id") == result.topicId:
                t["progress_status"] = "completed"
                break

    return {
        "roadmap": existing_roadmap,
        "log_status": "updated" if updated else "not_found",
    }


async def research_agent(state: LearningState):
    user_id = state.get("user_id")
    roadmapId = state.get("roadmapId")
    roadmap = await fetch_roadmap(roadmapId, user_id)
    # The topic the learner is actually on, read from the roadmap rather than
    # from state — nothing ever populated a state `active_topic`, so curated
    # resources used to be computed and then thrown away.
    current = active_topic(roadmap) if roadmap else None
    topic = (current or {}).get("title") or state["query"]

    # ReAct loop: the LLM issues its own tavily searches (refining queries as
    # needed) to gather the best free resources for the topic.
    messages = [
        SystemMessage(
            content=(
                "You are an expert at curating study resources. Use the search "
                "tool to find the most useful free courses, docs, books, and "
                "articles for the topic. Search more than once if a refined query "
                f"would help. Topic: {topic}"
            )
        ),
        HumanMessage(content=f"Find the best free learning resources for: {topic}"),
    ]
    messages = await run_tool_loop(research_llm, research_tool_node, messages)

    # Final pass: distill the gathered hits into a clean resource list.
    structured = llm.with_structured_output(ResearchOutput)
    result: ResearchOutput = await structured.ainvoke(
        messages
        + [
            HumanMessage(
                content=(
                    "Return the curated resources: a short title, the URL, and the "
                    "resource_type for each."
                )
            )
        ]
    )
    logger.info("research data %s", result)

    if current and roadmapId:
        await set_topic_resources(roadmapId, current["id"], result.resources, user_id)

    return {"suggestions": [r.model_dump() for r in result.resources]}


async def fallback_agent(state: LearningState):
    """Handles anything the router can't map to a real learning action:
    capability questions, greetings, ambiguous input, and off-scope requests.
    Intentionally cheap (fast_llm) and read-only — never mutates state."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the assistant for a personal learning-tracker agent. "
                "The user's message did not map to a specific action. Reply briefly "
                "and helpfully, then steer them back to what this agent can do.\n"
                "This agent can:\n"
                "- Create or modify a study roadmap\n"
                "- Explain a concept or topic\n"
                "- Quiz the user and grade answers\n"
                "- Find free learning resources\n"
                "- Track progress and say what to study next\n"
                "Rules:\n"
                "- If the user greets you or asks what you can do, briefly introduce "
                "these capabilities.\n"
                "- If the message is a vague or unclear learning request, ask one "
                "short clarifying question and give an example of how to phrase it.\n"
                "- If the message is clearly outside learning/study scope, say so "
                "politely in one line and redirect — do not attempt to answer it.\n"
                "- Keep it to 2-4 sentences. Prefer bullets only when listing "
                "capabilities.\n"
                "Learner profile (for tone/level only):\n{memory}",
            ),
            ("human", "{text}"),
        ]
    )

    chain = prompt | fast_llm
    response = await chain.ainvoke(
        {
            "text": state.get("query", ""),
            "memory": state.get("memory") or "none",
        }
    )
    logger.info("fallback_agent reply: %s", response.content)

    return {
        "intent": state.get("intent"),
        "topic_explaination": response.content,
    }


async def onboard(state: LearningState):
    """First-run profile capture. Pauses for two questions, then persists.

    The `onboarded` flag is written even when the learner skips, so declining
    once doesn't re-prompt on every subsequent message.
    """
    answers = interrupt(
        {
            "type": "onboarding",
            "questions": [
                {
                    "key": "skill_level",
                    "q": "Your current level?",
                    "options": ["beginner", "intermediate", "advanced"],
                },
                {
                    "key": "preferred_explanation_style",
                    "q": "How do you like things explained?",
                    "options": [
                        "concise",
                        "step_by_step",
                        "examples_first",
                        "socratic",
                    ],
                },
            ],
            "skippable": True,
        }
    )

    profile = {}
    if isinstance(answers, dict):
        # Validate through LearnerMemory and keep only the keys we asked for, so
        # a client can't write arbitrary fields into the profile by replying.
        try:
            answered = LearnerMemory(
                **{k: v for k, v in answers.items() if k in ONBOARDING_KEYS}
            )
            profile = answered.model_dump(exclude_none=True, exclude_defaults=True)
        except ValidationError as e:
            logger.warning("onboarding answers rejected: %s", e)

    profile[ONBOARDED_FLAG] = True
    await save_profile(state["user_id"], profile, LEARNING_NS)
    return {"memory": {**(state.get("memory") or {}), **profile}}


async def load_memory(state: LearningState):
    user_id = state["user_id"]
    memory = await get_profile(user_id, LEARNING_NS)
    # A bare "what should I study next?" carries no roadmapId, so resolve the
    # learner's current roadmap once here for every downstream node to use.
    roadmapId = await resolve_roadmap_id(user_id, state.get("roadmapId"))
    return {"memory": memory, "roadmapId": roadmapId}


def decide_onboarding(state: LearningState):
    memory = state.get("memory") or {}
    return "classify_intent" if memory.get(ONBOARDED_FLAG) else "onboard"


INTENT_ROUTES = {
    "create_roadmap": "roadmap_agent",
    "modify_roadmap": "roadmap_agent",
    "explain": "tutor_agent",
    "quiz": "tutor_agent",
    "submit_quiz": "quiz_grader_agent",
    "find_resources": "research_agent",
    "update_progress": "progress_agent",
    "query_roadmap": "progress_agent",
    "chitchat": "fallback_agent",
    "fallback": "fallback_agent",
}


def decide_agent(state: LearningState):
    return INTENT_ROUTES.get(state.get("intent"), "fallback_agent")


def build_graph() -> StateGraph:
    graph = StateGraph(LearningState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("onboard", onboard)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("roadmap_agent", roadmap_agent)
    graph.add_node("progress_agent", progress_agent)
    graph.add_node("tutor_agent", tutor_agent)
    graph.add_node("quiz_grader_agent", quiz_grader_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("fallback_agent", fallback_agent)
    graph.add_edge(START, "load_memory")
    # Conditional only — a second static load_memory → classify_intent edge would
    # run classification alongside onboarding and then again after it.
    graph.add_conditional_edges(
        "load_memory", decide_onboarding, ["onboard", "classify_intent"]
    )
    graph.add_edge("onboard", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent", decide_agent, sorted(set(INTENT_ROUTES.values()))
    )
    return graph


graph = build_graph()
