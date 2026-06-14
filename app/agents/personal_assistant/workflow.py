"""LangGraph nodes and graph wiring for the personal-assistant agent."""

import logging
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import START, StateGraph, END
from langgraph.types import interrupt

from app.core.llm import llm
from app.core.config import supabase
from .state import (
    PAState,
    IntentOutput,
    TaskInput,
    TaskUpdateInput,
    TaskSelector,
    NoteInput,
    BreakdownOutput,
    ResearchOutput,
    SynthesisOutput,
)
from .repository import (
    insert_todo,
    fetch_todos,
    find_pending_todos,
    complete_todo,
    delete_todos_by_ids,
    update_todo,
    append_memory_list,
    add_note,
    fetch_notes,
    categorize_agenda,
    insert_subtasks,
)
from .tools import web_search

logger = logging.getLogger(__name__)

MEMORY_RESEARCH_KEY = "pa_research_history"
MEMORY_COMPLETED_KEY = "pa_completed_history"

# How many prior turns to feed the classifier/extractors for follow-up context.
HISTORY_WINDOW = 6


def _recent_history(state: PAState) -> str:
    """Format the last few turns as plain text so the LLM can resolve
    references like 'that task' or 'the second one' in follow-up messages."""
    messages = state.get("messages") or []
    lines = []
    for m in messages[-HISTORY_WINDOW:]:
        role = getattr(m, "type", "") or m.__class__.__name__
        speaker = "User" if role in ("human", "HumanMessage") else "Assistant"
        content = getattr(m, "content", "")
        if content:
            lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


async def load_memory(state: PAState):
    user_id = state["user_id"]
    memory: dict = {}
    try:
        rows = (
            supabase.table("memory")
            .select("key, value")
            .eq("user_id", user_id)
            .execute()
        )
        if rows and rows.data:
            memory = {r["key"]: r["value"] for r in rows.data}
    except Exception as e:
        logger.error("pa load memory error: %s", e)
    # Record the user's turn in the conversation history (add_messages appends).
    return {"memory": memory, "messages": [HumanMessage(content=state.get("query", ""))]}


async def classify_intent(state: PAState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the user's message into exactly one intent:\n"
                "- add: create a new to-do / task\n"
                "- list: view or check existing tasks\n"
                "- complete: mark an existing task as done\n"
                "- delete: remove/cancel an existing task (destructive)\n"
                "- update: change an existing task's title, priority, due date, or details\n"
                "- research: look up information about a topic or task\n"
                "- note: remember a personal fact about the user (e.g. 'remember my "
                "wife's birthday is June 2')\n"
                "- recall: retrieve remembered facts (e.g. 'what do you know about me')\n"
                "- agenda: show what's due — overdue, today, or upcoming reminders\n"
                "- breakdown: split a larger goal or project into smaller subtasks\n"
                "Use the conversation history to resolve follow-up references. "
                "Reply with one word only.",
            ),
            (
                "human",
                "Conversation so far:\n{history}\n\nNew message: {text}",
            ),
        ]
    )
    chain = prompt | llm.with_structured_output(IntentOutput)
    result: IntentOutput = await chain.ainvoke(
        {"text": state.get("query", ""), "history": _recent_history(state)}
    )
    logger.info("pa intent: %s", result)
    return {"intent": result.intent}


async def _extract_selector(text: str) -> TaskSelector:
    chain = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "From the user's message, identify which existing task they mean. "
                "Return the task title to match on, or set match_all=true if they "
                "refer to all/every task.",
            ),
            ("human", "{text}"),
        ]
    ) | llm.with_structured_output(TaskSelector)
    return await chain.ainvoke({"text": text})


async def todo_agent(state: PAState):
    """Manages the MongoDB to-do list. Deleting tasks is gated behind HITL."""
    intent = state.get("intent")
    user_id = state["user_id"]

    if intent == "add":
        chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract a single to-do task from the user's message: a short "
                    "title, optional details, priority (low/medium/high), an "
                    "ISO due date if one is mentioned, and a recurrence "
                    "(daily/weekly/monthly) if the task clearly repeats "
                    "(e.g. 'every Monday' -> weekly).",
                ),
                ("human", "{text}"),
            ]
        ) | llm.with_structured_output(TaskInput)
        task: TaskInput = await chain.ainvoke({"text": state["query"]})
        created = await insert_todo(user_id, task.model_dump(exclude_none=True))
        return {"intent": "add", "task_status": "added", "todos": [created]}

    if intent == "list":
        todos = await fetch_todos(user_id, status="pending")
        return {"intent": "list", "task_status": "listed", "todos": todos}

    if intent == "complete":
        selector = await _extract_selector(state["query"])
        done = await complete_todo(user_id, selector.title or state["query"])
        if not done:
            return {"intent": "complete", "task_status": "not_found"}
        await append_memory_list(user_id, MEMORY_COMPLETED_KEY, done["title"])
        return {
            "intent": "complete",
            "task_status": "completed",
            "todos": await fetch_todos(user_id, status="pending"),
        }

    if intent == "delete":
        return await _delete_with_approval(state)

    if intent == "update":
        chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "From the user's message, extract: which existing task to update "
                    "(title field to match) and what changes to apply "
                    "(new_title, new_priority, new_due_at, new_details). "
                    "Only populate fields that the user explicitly wants to change.",
                ),
                ("human", "{text}"),
            ]
        ) | llm.with_structured_output(TaskUpdateInput)
        update_input: TaskUpdateInput = await chain.ainvoke({"text": state["query"]})
        updates = {
            k: v
            for k, v in {
                "title": update_input.new_title,
                "priority": update_input.new_priority,
                "due_at": update_input.new_due_at,
                "details": update_input.new_details,
            }.items()
            if v is not None
        }
        updated = await update_todo(
            user_id, update_input.title or state["query"], updates
        )
        if not updated:
            return {"intent": "update", "task_status": "not_found"}
        return {"intent": "update", "task_status": "updated", "todos": [updated]}

    return {"intent": intent, "task_status": "unknown"}


async def _delete_with_approval(state: PAState):
    """Destructive: build a deletion proposal, pause for human approval, then act.

    Re-run safe: on resume LangGraph replays this node from the top, so we look
    up an existing pending approval for this thread before creating a new one.
    """
    user_id = state["user_id"]
    thread_id = state["thread_id"]

    approval_id = None
    proposed = None
    try:
        existing = (
            supabase.table("approvals")
            .select("id, payload")
            .eq("thread_id", thread_id)
            .eq("status", "pending")
            .maybe_single()
            .execute()
        )
        if existing and existing.data:
            approval_id = existing.data["id"]
            proposed = existing.data["payload"]["tasks"]
    except Exception as e:
        logger.error("pa approval lookup error: %s", e)

    if not approval_id:
        selector = await _extract_selector(state["query"])
        matches = await find_pending_todos(user_id, selector.title, selector.match_all)
        if not matches:
            return {"intent": "delete", "task_status": "not_found"}
        proposed = [{"id": m["id"], "title": m["title"]} for m in matches]
        try:
            res = (
                supabase.table("approvals")
                .insert(
                    {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "action_type": "pa_delete_task",
                        "payload": {"tasks": proposed},
                        "status": "pending",
                    }
                )
                .execute()
            )
            approval_id = res.data[0]["id"] if res.data else None
        except Exception as e:
            logger.error("pa approval insert error: %s", e)

    decision = interrupt(
        {"type": "pa_delete_task", "approval_id": approval_id, "tasks": proposed}
    )

    if decision != "approved":
        if approval_id:
            supabase.table("approvals").update(
                {"status": "rejected", "resolved_at": datetime.now().isoformat()}
            ).eq("id", approval_id).execute()
        return {"intent": "delete", "task_status": "delete_rejected"}

    deleted = await delete_todos_by_ids(user_id, [t["id"] for t in (proposed or [])])
    if approval_id:
        supabase.table("approvals").update(
            {"status": "approved", "resolved_at": datetime.now().isoformat()}
        ).eq("id", approval_id).execute()
    return {
        "intent": "delete",
        "task_status": f"deleted:{deleted}",
        "todos": await fetch_todos(user_id, status="pending"),
    }


async def research_agent(state: PAState):
    """Researches the user's topic via the Search API and summarizes it."""
    topic = state["query"]
    search = await web_search(topic)

    context_lines = []
    if search.get("answer"):
        context_lines.append(f"Search answer: {search['answer']}")
    for r in search.get("results", []):
        context_lines.append(f"- {r['title']} ({r['url']})\n  {r['content']}")
    context = "\n".join(context_lines) or "No external results available."

    messages = [
        SystemMessage(
            content=(
                "You are a research assistant. Using the provided search results, "
                "write a concise summary, a few key points, and list the source "
                "URLs. If the results are empty, answer from general knowledge and "
                "leave sources empty."
            )
        ),
        HumanMessage(content=f"Topic: {topic}\n\nSearch results:\n{context}"),
    ]
    structured: ResearchOutput = await llm.with_structured_output(
        ResearchOutput
    ).ainvoke(messages)

    await append_memory_list(state["user_id"], MEMORY_RESEARCH_KEY, topic)

    return {
        "intent": "research",
        "research": structured.model_dump(),
        "suggestions": structured.key_points,
    }


async def notes_agent(state: PAState):
    """Stores free-form personal facts and recalls them on demand."""
    intent = state.get("intent")
    user_id = state["user_id"]

    if intent == "note":
        chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract the personal fact the user wants remembered as a "
                    "concise statement, plus an optional one-word category "
                    "(e.g. family, work, health, preferences).",
                ),
                ("human", "{text}"),
            ]
        ) | llm.with_structured_output(NoteInput)
        note: NoteInput = await chain.ainvoke({"text": state["query"]})
        await add_note(user_id, note.content, note.category)
        return {
            "intent": "note",
            "task_status": "noted",
            "notes": await fetch_notes(user_id),
        }

    # recall
    return {
        "intent": "recall",
        "task_status": "recalled",
        "notes": await fetch_notes(user_id),
    }


async def agenda_agent(state: PAState):
    """Surfaces due-date awareness: overdue, today, and upcoming tasks."""
    user_id = state["user_id"]
    todos = await fetch_todos(user_id, status="pending")
    agenda = categorize_agenda(todos)
    return {
        "intent": "agenda",
        "task_status": "agenda",
        "agenda": agenda,
        "todos": todos,
    }


async def breakdown_agent(state: PAState):
    """Splits a larger goal into ordered subtasks, persisting each as a child task."""
    user_id = state["user_id"]
    chain = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "The user wants to break a larger goal into actionable steps. "
                "Return a concise parent_title for the overall goal and an ordered "
                "list of 3-7 concrete subtasks.",
            ),
            ("human", "{text}"),
        ]
    ) | llm.with_structured_output(BreakdownOutput)
    bd: BreakdownOutput = await chain.ainvoke({"text": state["query"]})
    parent = await insert_todo(user_id, {"title": bd.parent_title})
    children = await insert_subtasks(user_id, parent["id"], bd.subtasks)
    return {
        "intent": "breakdown",
        "task_status": f"broke_down:{len(children)}",
        "todos": [parent],
        "subtasks": children,
    }


def decide_agent(state: PAState):
    intent = state.get("intent")
    if intent in ("add", "list", "complete", "delete", "update"):
        return "todo_agent"
    if intent == "research":
        return "research_agent"
    if intent in ("note", "recall"):
        return "notes_agent"
    if intent == "agenda":
        return "agenda_agent"
    if intent == "breakdown":
        return "breakdown_agent"
    return END


async def synthesize_response(state: PAState):
    """Turn the structured operation result into a short, friendly plain-English reply."""
    intent = state.get("intent", "")
    task_status = state.get("task_status", "")
    todos = state.get("todos") or []
    research = state.get("research")
    notes = state.get("notes") or []
    agenda = state.get("agenda")
    subtasks = state.get("subtasks") or []

    parts = [f"Intent: {intent}", f"Status: {task_status}"]
    if todos:
        task_lines = "\n".join(
            f"  - [{t.get('priority', '?')}] {t.get('title', '')} "
            f"(due: {t.get('due_at') or 'none'}, status: {t.get('status', '')})"
            for t in todos[:20]
        )
        parts.append(f"Tasks:\n{task_lines}")
    if subtasks:
        sub_lines = "\n".join(f"  {i + 1}. {s.get('title', '')}" for i, s in enumerate(subtasks))
        parts.append(f"Subtasks:\n{sub_lines}")
    if agenda:
        parts.append(
            "Agenda counts — "
            f"overdue: {len(agenda.get('overdue', []))}, "
            f"today: {len(agenda.get('today', []))}, "
            f"upcoming: {len(agenda.get('upcoming', []))}, "
            f"no date: {len(agenda.get('no_date', []))}"
        )
        for label in ("overdue", "today", "upcoming"):
            items = agenda.get(label) or []
            if items:
                names = "; ".join(t.get("title", "") for t in items[:10])
                parts.append(f"{label.capitalize()}: {names}")
    if notes:
        note_lines = "\n".join(
            f"  - {n.get('content', '')}"
            + (f" ({n.get('category')})" if n.get("category") else "")
            for n in notes[-20:]
        )
        parts.append(f"Known facts about the user:\n{note_lines}")
    if research:
        parts.append(f"Research summary: {research.get('summary', '')}")
        if research.get("key_points"):
            parts.append("Key points: " + "; ".join(research["key_points"]))

    context = "\n".join(parts)
    messages = [
        SystemMessage(
            content=(
                "You are a concise personal assistant. Based on the operation result "
                "below, write a short, friendly reply to the user. Describe what "
                "happened in plain language — do not dump raw data."
            )
        ),
        HumanMessage(
            content=f'User said: "{state.get("query", "")}"\n\nResult:\n{context}'
        ),
    ]
    result: SynthesisOutput = await llm.with_structured_output(SynthesisOutput).ainvoke(
        messages
    )
    # Record the assistant's turn so the next message has conversation context.
    return {
        "response": result.response,
        "messages": [AIMessage(content=result.response)],
    }


def build_graph() -> StateGraph:
    graph = StateGraph(PAState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("todo_agent", todo_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("notes_agent", notes_agent)
    graph.add_node("agenda_agent", agenda_agent)
    graph.add_node("breakdown_agent", breakdown_agent)
    graph.add_node("synthesize", synthesize_response)
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        decide_agent,
        [
            "todo_agent",
            "research_agent",
            "notes_agent",
            "agenda_agent",
            "breakdown_agent",
            END,
        ],
    )
    graph.add_edge("todo_agent", "synthesize")
    graph.add_edge("research_agent", "synthesize")
    graph.add_edge("notes_agent", "synthesize")
    graph.add_edge("agenda_agent", "synthesize")
    graph.add_edge("breakdown_agent", "synthesize")
    graph.add_edge("synthesize", END)
    return graph


graph = build_graph()
