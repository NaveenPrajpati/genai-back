"""LangGraph nodes and graph wiring for the personal-assistant agent."""

import logging
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import START, StateGraph, END
from langgraph.types import interrupt

from app.core.llm import llm, fast_llm
from app.services.cache import cached_value
from app.core.config import CACHE_CLASSIFY_THRESHOLD
from app.agents.approval_store import get_pending, create_pending, resolve
from app.agents.react import run_tool_loop
from app.agents.memory_store import get_profile
from .state import (
    PAState,
    IntentOutput,
    TaskInput,
    TaskUpdateInput,
    TaskSelector,
    NoteInput,
    BreakdownOutput,
    SubtaskInput,
    PlanOutput,
    ResearchOutput,
    ReviewOutput,
    SynthesisOutput,
)
from .repository import (
    insert_todo,
    fetch_todos,
    fetch_completed_since,
    find_pending_todos,
    describe_selection,
    complete_todo,
    delete_todos_by_ids,
    update_todo,
    append_memory_list,
    add_note,
    fetch_notes,
    categorize_agenda,
    insert_subtasks,
)
from .tools import build_research_loop

logger = logging.getLogger(__name__)

MEMORY_RESEARCH_KEY = "pa_research_history"
MEMORY_COMPLETED_KEY = "pa_completed_history"

# How many prior turns to feed the classifier/extractors for follow-up context.
HISTORY_WINDOW = 6

# How far back a "what did I get done" review looks. A week: long enough to
# have something to say, short enough that the answer is still about now.
REVIEW_WINDOW_DAYS = 7


def _today_for(state: PAState) -> str:
    """Today's date in the user's own timezone, named and dated.

    Anything extracting a due date needs this. Without it the model resolves
    "tomorrow" against whatever its training suggests today is, which is how a
    task added on 29 Aug 2026 acquires a due date in October — a wrong date that
    then drives the agenda, the digest and every overdue count downstream.

    `planner_agent` has passed a date anchor since it was written. The two nodes
    that actually write dates to the database did not, which is the wrong way
    round: the planner's mistake lasts one reply, theirs is persisted.
    """
    now = datetime.now(timezone.utc)
    name = (state.get("memory") or {}).get("timezone")
    if name:
        try:
            now = now.astimezone(ZoneInfo(name))
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning(
                "unknown timezone %r for user=%s, dating in UTC",
                name,
                state.get("user_id"),
            )
    return f"{now.strftime('%A')} {now.date().isoformat()}"


# Appended to every prompt that may return a due date. Relative dates are the
# whole reason a date anchor is passed at all.
_DATE_RULES = (
    "\nToday is {today}. Resolve every relative date — 'tomorrow', 'Friday', "
    "'next week' — against that date, and return it as an ISO 8601 string. "
    "Leave the date empty when the user did not give one; never invent it."
)


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
    # Long-term profile + app-managed memory (notes, history, prefs) all live in
    # the shared Mongo `memories` doc.
    memory: dict = await get_profile(user_id)
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
                "- plan: organise today into a schedule (e.g. 'plan my day', "
                "'how should I spend today', 'time-block this')\n"
                "- review: look back over what they've recently finished "
                "(e.g. 'what did I get done this week', 'weekly review')\n"
                "Prefer 'agenda' when they just want to see what's due, and "
                "'plan' when they want it arranged into a day. "
                "Use the conversation history to resolve follow-up references. "
                "Reply with one word only.",
            ),
            (
                "human",
                "Conversation so far:\n{history}\n\nNew message: {text}",
            ),
        ]
    )
    chain = prompt | fast_llm.with_structured_output(IntentOutput)
    query = state.get("query", "")
    history = _recent_history(state)

    async def produce():
        result: IntentOutput = await chain.ainvoke({"text": query, "history": history})
        logger.info("pa intent: %s", result)
        return result.model_dump()

    if history:
        # Follow-ups ("that one", "make it high priority") depend on context, so
        # classify fresh — never serve a context-free cached intent for them.
        data = await produce()
    else:
        # Fresh command (no history): intent depends only on the message and is
        # user-independent → global scope shared across users, loose threshold.
        data = await cached_value(
            query, "agent:pa:classify_intent", CACHE_CLASSIFY_THRESHOLD, produce
        )
    return {"intent": data["intent"]}


async def _extract_selector(text: str) -> TaskSelector:
    """Work out which existing task(s) a complete/delete/update request means.

    The old wording — "set match_all=true if they refer to all/every task" — read
    "delete all my Learn: tasks" as an every-task request, because the sentence
    does say "all". A qualified all is a filter, and it has to come back as one.
    """
    chain = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Identify which existing task(s) the user means.\n"
                "- title: the distinctive fragment of the task title to match "
                "on — the words that pick the task out, not the whole sentence.\n"
                "- match_all: true ONLY for an unqualified request covering "
                "every task they have ('delete everything', 'clear my whole "
                "list').\n"
                "A qualified 'all' is a filter, not match_all: 'all my Learn: "
                "tasks' sets title='Learn:', and 'every shopping task' sets "
                "title='shopping'. When in doubt, prefer a title — it narrows, "
                "and match_all does not.",
            ),
            ("human", "{text}"),
        ]
    ) | fast_llm.with_structured_output(TaskSelector)
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
                    "(e.g. 'every Monday' -> weekly)." + _DATE_RULES,
                ),
                ("human", "{text}"),
            ]
        ) | fast_llm.with_structured_output(TaskInput)
        task: TaskInput = await chain.ainvoke(
            {"text": state["query"], "today": _today_for(state)}
        )
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
                    "Only populate fields that the user explicitly wants to change."
                    + _DATE_RULES,
                ),
                ("human", "{text}"),
            ]
        ) | fast_llm.with_structured_output(TaskUpdateInput)
        update_input: TaskUpdateInput = await chain.ainvoke(
            {"text": state["query"], "today": _today_for(state)}
        )
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
    scope = None
    existing = await get_pending(thread_id)
    if existing:
        approval_id = str(existing["_id"])
        proposed = existing["payload"]["tasks"]
        # Absent on rows written before the card described its own filter.
        scope = existing["payload"].get("scope")

    if not approval_id:
        selector = await _extract_selector(state["query"])
        matches = await find_pending_todos(user_id, selector.title, selector.match_all)
        if not matches:
            return {"intent": "delete", "task_status": "not_found"}
        proposed = [{"id": m["id"], "title": m["title"]} for m in matches]
        scope = describe_selection(selector.title, selector.match_all, len(matches))
        logger.info(
            "pa delete proposal user=%s scope=%s matched=%d",
            user_id,
            scope,
            len(matches),
        )
        approval_id = await create_pending(
            user_id, thread_id, "pa_delete_task", {"tasks": proposed, "scope": scope}
        )

    decision = interrupt(
        {
            "type": "pa_delete_task",
            "approval_id": approval_id,
            "tasks": proposed,
            # What the filter resolved to, so the user can check the rule and not
            # just the list it produced. A card headed "12 tasks whose title
            # contains 'Learn:'" is falsifiable at a glance; a bare list of
            # titles is not.
            "count": len(proposed or []),
            "scope": scope,
        }
    )

    if decision != "approved":
        await resolve(approval_id, "rejected")
        return {"intent": "delete", "task_status": "delete_rejected"}

    deleted = await delete_todos_by_ids(user_id, [t["id"] for t in (proposed or [])])
    await resolve(approval_id, "approved")
    return {
        "intent": "delete",
        "task_status": f"deleted:{deleted}",
        "todos": await fetch_todos(user_id, status="pending"),
    }


async def research_agent(state: PAState):
    """Researches the user's topic via web search (ReAct: the LLM runs its own
    searches, refining as needed) and summarizes the findings."""
    topic = state["query"]

    messages = [
        SystemMessage(
            content=(
                "You are a research assistant. Use the web search tool to gather "
                "up-to-date information on the user's topic, searching more than "
                "once with refined queries if helpful. Then write a concise "
                "summary, a few key points, and list the source URLs. If search "
                "returns nothing, answer from general knowledge and leave sources "
                "empty.\n"
                "If the user's intent is to *learn or study* a subject (not just "
                "look up a fact), call start_learning_roadmap to build them a "
                "structured roadmap instead of only summarizing."
            )
        ),
        HumanMessage(content=f"Topic: {topic}"),
    ]
    research_model, tool_node = build_research_loop(state["user_id"])
    messages = await run_tool_loop(research_model, tool_node, messages)

    structured: ResearchOutput = await llm.with_structured_output(
        ResearchOutput
    ).ainvoke(
        messages
        + [
            HumanMessage(
                content="Return the summary, key points, and source URLs."
            )
        ]
    )

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
        ) | fast_llm.with_structured_output(NoteInput)
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
                "Return a concise parent_title for the overall goal and an "
                "ordered list of 3-7 concrete subtasks.\n"
                "Each subtask has its own `priority` and `due_at`. Scheduling "
                "goes in those fields and never in the title: a subtask titled "
                "'Monday: draft the outline' is wrong, because a day written "
                "inside a title cannot be sorted, reminded on, rolled over or "
                "shown on the agenda. Title it 'Draft the outline' and put "
                "Monday in due_at.\n"
                "Set due_at only when the user asked for the work to be "
                "scheduled, and priority only when they said something about "
                "what matters most — including when they single out one step "
                "('make the first one high priority')." + _DATE_RULES,
            ),
            ("human", "{text}"),
        ]
    ) | llm.with_structured_output(BreakdownOutput)
    bd: BreakdownOutput = await chain.ainvoke(
        {"text": state["query"], "today": _today_for(state)}
    )
    parent = await insert_todo(user_id, {"title": bd.parent_title})
    children = await insert_subtasks(
        user_id, parent["id"], [s.model_dump(exclude_none=True) for s in bd.subtasks]
    )
    return {
        "intent": "breakdown",
        "task_status": f"broke_down:{len(children)}",
        "todos": [parent],
        "subtasks": children,
    }


async def planner_agent(state: PAState):
    """Lay today's pending tasks out as time blocks.

    The pieces for this were already in place and unused: the agenda buckets
    tasks by due date, and the learned profile records `work_hours` and standing
    `priorities` that until now only shaped the wording of replies. "What should
    I do today" could be answered with a list, but not with a shape for the day.

    The plan is not persisted. It is advice about one day, it goes stale by
    evening, and storing it would create a second, competing source of truth
    next to the tasks themselves.
    """
    user_id = state["user_id"]
    memory = state.get("memory") or {}
    pending = await fetch_todos(user_id, status="pending")
    agenda = categorize_agenda(pending)

    # Overdue first, then today, then undated — upcoming tasks are not yet the
    # user's problem and scheduling them today would crowd out what is.
    candidates = agenda["overdue"] + agenda["today"] + agenda["no_date"]
    if not candidates:
        return {
            "intent": "plan",
            "task_status": "nothing_to_plan",
            "plan": {"blocks": [], "unscheduled": [], "notes": None},
            "agenda": agenda,
        }

    task_lines = "\n".join(
        f"  - id={t['id']} [{t.get('priority', 'medium')}] {t.get('title', '')}"
        f" (due: {t.get('due_at') or 'none'})"
        for t in candidates[:20]
    )
    profile = "; ".join(
        f"{label}: {memory[key]}"
        for key, label in (
            ("work_hours", "Working hours"),
            ("priorities", "Standing priorities"),
            ("recurring_commitments", "Recurring commitments"),
        )
        if memory.get(key)
    )

    chain = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Build a realistic time-blocked plan for the user's day from "
                "their pending tasks.\n"
                "Rules:\n"
                "- Stay inside their working hours when given; otherwise assume "
                "09:00-17:00.\n"
                "- Respect their recurring commitments — do not schedule over "
                "them.\n"
                "- Overdue and high-priority work goes earliest.\n"
                "- Use realistic durations and leave short breaks between "
                "blocks. Do not fill every minute.\n"
                "- Set task_id to the id given for that task; leave it null for "
                "breaks and buffers.\n"
                "- Anything that genuinely does not fit goes in `unscheduled` "
                "by title. Do not silently drop it.\n"
                "- Times are local wall-clock, 'HH:MM', ordered by start.",
            ),
            (
                "human",
                "Today is {today}.\n"
                "User profile: {profile}\n\nTasks:\n{tasks}\n\n"
                "Their request: {text}",
            ),
        ]
    ) | llm.with_structured_output(PlanOutput)

    plan: PlanOutput = await chain.ainvoke(
        {
            "today": date.today().isoformat(),
            "profile": profile or "none recorded",
            "tasks": task_lines,
            "text": state["query"],
        }
    )
    return {
        "intent": "plan",
        "task_status": f"planned:{len(plan.blocks)}",
        "plan": plan.model_dump(),
        "todos": candidates,
        "agenda": agenda,
    }


async def review_agent(state: PAState):
    """Look back over the week: what got finished, and what is still open.

    This is what `MEMORY_COMPLETED_KEY` was always for. The graph has appended
    every completed task title to it since the feature shipped, and nothing has
    ever read it back. Paired here with the tasks actually completed in the
    window — the memory list is capped and undated, so it is useful for
    recognising themes but cannot answer "this week" on its own.
    """
    user_id = state["user_id"]
    memory = state.get("memory") or {}
    since = (datetime.now() - timedelta(days=REVIEW_WINDOW_DAYS)).isoformat()

    completed = await fetch_completed_since(user_id, since)
    pending = await fetch_todos(user_id, status="pending")
    agenda = categorize_agenda(pending)

    completed_lines = (
        "\n".join(f"  - {t.get('title', '')}" for t in completed[:40]) or "  (none)"
    )
    open_lines = (
        "\n".join(
            f"  - [{t.get('priority', 'medium')}] {t.get('title', '')}"
            for t in (agenda["overdue"] + agenda["today"])[:20]
        )
        or "  (none)"
    )
    researched = (memory.get(MEMORY_RESEARCH_KEY) or [])[-10:]

    chain = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Write a short, encouraging weekly review of the user's work. "
                "Summarise what they finished, pull out a few genuine "
                "highlights, name what is still open and worth carrying into "
                "next week, and offer one or two concrete suggestions.\n"
                "Be honest: if very little was completed, say so kindly rather "
                "than inflating it. Do not invent work that is not listed.",
            ),
            (
                "human",
                "Window: the last {days} days.\n\n"
                "Completed ({completed_count}):\n{completed}\n\n"
                "Still open (overdue and due today):\n{open}\n\n"
                "Topics they researched: {researched}\n\n"
                "Their request: {text}",
            ),
        ]
    ) | llm.with_structured_output(ReviewOutput)

    review: ReviewOutput = await chain.ainvoke(
        {
            "days": REVIEW_WINDOW_DAYS,
            "completed_count": len(completed),
            "completed": completed_lines,
            "open": open_lines,
            "researched": ", ".join(researched) or "none",
            "text": state["query"],
        }
    )
    return {
        "intent": "review",
        "task_status": f"reviewed:{len(completed)}",
        "review": {**review.model_dump(), "completed_count": len(completed)},
        "suggestions": review.suggestions,
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
    if intent == "plan":
        return "planner_agent"
    if intent == "review":
        return "review_agent"
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
    memory = state.get("memory") or {}
    plan = state.get("plan")
    review = state.get("review")

    parts = [f"Intent: {intent}", f"Status: {task_status}"]
    # Surface the learned profile so the reply can reflect the user's standing
    # priorities and preferred communication style.
    profile_bits = [
        f"{label}: {memory[key]}"
        for key, label in (
            ("communication_style", "Preferred style"),
            ("priorities", "Standing priorities"),
            ("work_hours", "Work hours"),
        )
        if memory.get(key)
    ]
    if profile_bits:
        parts.append("User profile — " + "; ".join(profile_bits))
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
    if plan:
        blocks = plan.get("blocks") or []
        if blocks:
            block_lines = "\n".join(
                f"  {b.get('start', '')}-{b.get('end', '')} {b.get('title', '')}"
                for b in blocks
            )
            parts.append(f"Day plan:\n{block_lines}")
        else:
            parts.append("Day plan: nothing needed scheduling.")
        if plan.get("unscheduled"):
            parts.append("Did not fit today: " + "; ".join(plan["unscheduled"]))
    if review:
        parts.append(
            f"Weekly review — completed {review.get('completed_count', 0)}: "
            f"{review.get('summary', '')}"
        )
        if review.get("highlights"):
            parts.append("Highlights: " + "; ".join(review["highlights"]))
        if review.get("carry_over"):
            parts.append("Still open: " + "; ".join(review["carry_over"]))

    context = "\n".join(parts)
    messages = [
        SystemMessage(
            content=(
                "You are a concise personal assistant. Based on the operation result "
                "below, write a short, friendly reply to the user. Describe what "
                "happened in plain language — do not dump raw data. If a user "
                "profile is given, honor their preferred communication style."
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
    graph.add_node("planner_agent", planner_agent)
    graph.add_node("review_agent", review_agent)
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
            "planner_agent",
            "review_agent",
            END,
        ],
    )
    graph.add_edge("todo_agent", "synthesize")
    graph.add_edge("research_agent", "synthesize")
    graph.add_edge("notes_agent", "synthesize")
    graph.add_edge("agenda_agent", "synthesize")
    graph.add_edge("breakdown_agent", "synthesize")
    graph.add_edge("planner_agent", "synthesize")
    graph.add_edge("review_agent", "synthesize")
    graph.add_edge("synthesize", END)
    return graph


graph = build_graph()
