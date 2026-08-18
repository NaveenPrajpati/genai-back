"""LangGraph nodes and graph wiring for the learning-tracker agent."""

import asyncio
import logging
from datetime import datetime, timezone

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import ValidationError

from app.core.llm import llm, fast_llm
from app.database import get_db
from app.agents.approval_store import get_pending, create_pending, resolve
from app.agents.react import run_tool_loop
from app.agents.memory_store import get_profile, save_profile
from .service import build_roadmap, revise_roadmap, sync_roadmap_to_pa
from .context import learner_context, situation_text
from .actions import build_action_tools
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
    profile_snapshot,
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
                # The distinction that matters is asking-about versus asking-for.
                # "What should I do next?" wants an answer; "start the next topic"
                # wants it done. Spelled out with examples because the two are one
                # word apart and the wrong branch is a tutor describing a button.
                "- take_action: user asks the assistant to DO something to their learning setup: "
                "start/begin a topic, send or generate today's lesson now, pause or resume a "
                "roadmap, write a note down for them, or change what time lessons arrive. "
                "Examples: 'start the next topic', 'give me today's lesson now', 'pause Rust', "
                "'note that I keep mixing these up', 'send my lessons at 7am'. "
                "Choose query_roadmap instead when they are only asking what to do, not asking "
                "you to do it.\n"
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
    #
    # The scope carries a version, and it MUST be bumped whenever the label set
    # changes. Entries cached under the old scope hold verdicts from a classifier
    # that had never heard of the new label, so "start the next topic" would go on
    # resolving to `update_progress` for the rest of the TTL — a new intent that
    # silently does nothing for existing phrasings. v2 added `take_action`.
    data = await cached_value(
        query, "agent:learning:classify_intent:v2", CACHE_CLASSIFY_THRESHOLD, produce
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

    memory = state.get("memory")
    if is_modify:
        # Merge onto the stored document so the learner keeps the progress they
        # already made on topics that survived the edit.
        roadmap = merge_roadmap(existing_roadmap, draft, memory)
        saved = await replace_roadmap(state["roadmapId"], user_id, roadmap)
        saved_roadmapId = state["roadmapId"] if saved else None
    else:
        roadmap = materialize_roadmap(draft, personalization=profile_snapshot(memory))
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
                    "Pitch the explanation to the learner's profile:\n{memory}\n"
                    # The difference between an explanation and a *tutor's*
                    # explanation. A learner who has failed this topic's recall
                    # check twice on the same point needs that point attacked,
                    # not the same treatment a stranger would get.
                    "Where they currently stand:\n{situation}\n"
                    "Use that: if one of their recorded misunderstandings bears on "
                    "what they asked, teach directly against it. Do NOT say you are "
                    "doing so, do not mention their mistakes, scores or progress, and "
                    "do not tell them what is coming in a future question — teach the "
                    "point, and let the correction be the explanation.",
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
                "situation": situation_text(state.get("context")),
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
        user_id,
        str(quiz["_id"]),
        quiz.get("roadmapId"),
        quiz.get("topicId"),
        result,
        questions=questions,
        kind=quiz.get("kind") or "chat",
    )
    return {"quiz_result": result, "quizId": str(quiz["_id"])}


async def progress_agent(state: LearningState):
    user_id = state.get("user_id")
    existing_roadmap = await fetch_roadmap(state.get("roadmapId"), user_id)

    if state.get("intent") == "query_roadmap":
        # The numbers stay pure data work: what's next and how far along is fully
        # determined by `progress_status` / `order`, and a model asked to count
        # them would eventually miscount. What the model adds is the part a count
        # cannot be — which of it to do first, and why.
        #
        # This is the path behind "what should I work on next?", and answering
        # that with a progress bar was the most mechanical moment in the whole
        # app. The bar was never wrong; it just wasn't an answer.
        progress = roadmap_progress(existing_roadmap)
        guidance = ""
        if existing_roadmap:
            try:
                chain = (
                    ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are the learner's coach. They have asked what to "
                                "work on next.\n"
                                "Their situation:\n{situation}\n"
                                "Answer in one or two sentences: name the single thing "
                                "to do first, and why it is that one.\n"
                                "Rules:\n"
                                "- The screen beside you already shows their progress "
                                "and next topic. Do not restate counts, percentages or "
                                "the topic name as a heading — add the judgement they "
                                "cannot read off a bar.\n"
                                "- Prefer an overdue review, then anything blocking a "
                                "roadmap, then unread digests.\n"
                                "- Never state a clock time or countdown.\n"
                                "- Never use internal names like blocked_reason or "
                                "needs_revision; say what they mean.\n"
                                "- Speak to them as 'you'. No praise, no exclamation "
                                "marks.",
                            ),
                            ("human", "{text}"),
                        ]
                    )
                    | fast_llm
                )
                reply = await chain.ainvoke(
                    {
                        "text": state.get("query", "What should I do next?"),
                        "situation": situation_text(state.get("context")),
                    }
                )
                guidance = (reply.content or "").strip()
            except Exception as e:
                # The card renders from `progress` alone, exactly as it did
                # before this existed — the coaching is the addition, not the
                # substance, and losing it must not lose the answer.
                logger.error("query_roadmap guidance failed: %s", e)

        return {
            "next_topic": progress["next_topic"] or "",
            "progress": progress,
            "guidance": guidance,
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

    matched = next((t for t in topics if t.get("id") == result.topicId), None)
    if matched is None:
        return {"roadmap": existing_roadmap, "log_status": "not_found"}

    if matched.get("progress_status") == "completed":
        return {"roadmap": existing_roadmap, "log_status": "already_complete"}

    # Saying "I finished pointers" moves the topic along but does NOT complete
    # it: completion is earned by passing the checkpoint. Letting chat mark
    # things done would be a trivial way around the gate, which would make the
    # gate — and the mastery scores behind it — meaningless.
    updated = await set_topic_progress(
        roadmapId, result.topicId, "in_progress", user_id
    )
    if updated:
        matched["progress_status"] = "in_progress"

    return {
        "roadmap": existing_roadmap,
        "next_topic": matched.get("title", ""),
        "log_status": "checkpoint_required" if updated else "not_found",
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


_ACTION_SYSTEM = """You are the learner's tutor, and you can act on their behalf \
rather than telling them where to tap.

Their situation:
{situation}

Do what they asked, then say what you did in one or two sentences.

Rules:
- Resolve ids with list_topics before any action that needs a topicId. Never \
guess an id, and never invent one.
- Take the action. Do not ask permission for something they just asked for; if \
their request is genuinely ambiguous between two roadmaps or two topics, ask one \
short question instead of guessing.
- If a tool declines, relay its reason in your own words and say what would \
unblock it. A refusal is an answer, not an error — do not retry it.
- Never claim to have done something a tool did not confirm.
- You cannot mark a lesson as read, complete a topic, or answer a check for them. \
Those are earned by doing them. If they ask, say so plainly and without apology, \
and point at what would actually move it.
- Speak to them as "you". No clock times, no internal names, no exclamation marks.
"""


async def action_agent(state: LearningState):
    """Does the thing, instead of describing where the button is.

    The tools are built per turn and closed over this learner's id — see
    `build_action_tools`, which also carries the rules about what may never be
    exposed. The loop is the shared ReAct one: look ids up, act, report.

    On any failure this degrades to an explanation rather than a stack trace: the
    learner asked for something to happen, and "I couldn't" is a worse answer than
    a description of what to do, but both beat a broken turn.
    """
    user_id = state.get("user_id")
    tools = build_action_tools(user_id)

    messages = [
        SystemMessage(
            content=_ACTION_SYSTEM.format(situation=situation_text(state.get("context")))
        ),
        HumanMessage(content=state.get("query", "")),
    ]

    try:
        messages = await run_tool_loop(llm.bind_tools(tools), ToolNode(tools), messages)
    except Exception as e:
        logger.error("action_agent tool loop failed: %s", e)
        return {
            "topic_explaination": (
                "I couldn't do that just now — the action didn't go through. "
                "Nothing has changed, so it's safe to ask again."
            ),
            "actions_taken": [],
        }

    # Which tools actually ran. The client refreshes what they touched, and
    # returning the names rather than a bare flag means it can refresh only what
    # moved instead of re-reading the whole section on every chat turn.
    taken = [
        call["name"]
        for message in messages
        for call in (getattr(message, "tool_calls", None) or [])
    ]
    reply = messages[-1].content if messages else ""

    return {
        "topic_explaination": reply or "Done.",
        "actions_taken": taken,
    }


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
                "Learner profile (for tone/level only):\n{memory}\n"
                # A greeting is the single best moment to say the useful thing,
                # and answering "hi" with a feature list wastes it.
                "Where they currently stand:\n{situation}\n"
                "If they are greeting you or asking what you can do, lead with the "
                "one thing actually waiting for them rather than a list of features — "
                "name it, then keep the capabilities to a line. Never state a clock "
                "time or countdown, and never use internal names like blocked_reason.",
            ),
            ("human", "{text}"),
        ]
    )

    chain = prompt | fast_llm
    response = await chain.ainvoke(
        {
            "text": state.get("query", ""),
            "memory": state.get("memory") or "none",
            "situation": situation_text(state.get("context")),
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


async def load_context(state: LearningState):
    """Who the learner is, which roadmap they mean, and where they are standing.

    The third of those is what separates a tutor from a router. Until it existed,
    every node saw the learner's profile and their message and nothing else — so
    the agent could explain a topic but could not know the learner had failed its
    checkpoint that morning, and every judgement of that kind had to be made in
    the client instead.

    All three reads are independent, so they go out together: the situation costs
    about what one screen load costs, and it is paid once per turn rather than
    once per node.
    """
    user_id = state["user_id"]
    memory, roadmapId, context = await asyncio.gather(
        get_profile(user_id, LEARNING_NS),
        # A bare "what should I study next?" carries no roadmapId, so resolve the
        # learner's current roadmap once here for every downstream node to use.
        resolve_roadmap_id(user_id, state.get("roadmapId")),
        # Cheap half only — mastery sweeps every attempt ever recorded, which is
        # a briefing's cost to pay, not a chat turn's. See `learner_context`.
        learner_context(user_id),
    )
    return {"memory": memory, "roadmapId": roadmapId, "context": context}


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
    "take_action": "action_agent",
    "chitchat": "fallback_agent",
    "fallback": "fallback_agent",
}


def decide_agent(state: LearningState):
    return INTENT_ROUTES.get(state.get("intent"), "fallback_agent")


def build_graph() -> StateGraph:
    graph = StateGraph(LearningState)
    graph.add_node("load_context", load_context)
    graph.add_node("onboard", onboard)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("roadmap_agent", roadmap_agent)
    graph.add_node("progress_agent", progress_agent)
    graph.add_node("tutor_agent", tutor_agent)
    graph.add_node("quiz_grader_agent", quiz_grader_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("action_agent", action_agent)
    graph.add_node("fallback_agent", fallback_agent)

    graph.add_edge(START, "load_context")
    graph.add_conditional_edges(
        "load_context", decide_onboarding, ["onboard", "classify_intent"]
    )
    graph.add_edge("onboard", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent", decide_agent, sorted(set(INTENT_ROUTES.values()))
    )
    return graph


graph = build_graph()
