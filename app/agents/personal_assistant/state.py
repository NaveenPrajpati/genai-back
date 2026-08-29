"""State and structured-output schemas for the personal-assistant agent."""

from typing import TypedDict, Optional, List, Literal, Annotated
from pydantic import BaseModel
from langgraph.graph.message import add_messages


class PAState(TypedDict, total=False):
    query: str
    intent: str
    current_user: dict
    user_id: str
    thread_id: str
    memory: dict
    # Conversation history, persisted per thread by the checkpointer. The
    # add_messages reducer appends rather than overwrites, so follow-ups like
    # "make that one high priority" can see what came before.
    messages: Annotated[list, add_messages]
    todos: Optional[list]
    notes: Optional[list]
    agenda: Optional[dict]
    subtasks: Optional[list]
    research: Optional[dict]
    suggestions: Optional[list]
    # A time-blocked schedule for today (intent: plan).
    plan: Optional[dict]
    # A look back over the last week's finished work (intent: review).
    review: Optional[dict]
    task_status: Optional[str]
    response: Optional[str]


class IntentOutput(BaseModel):
    intent: Literal[
        "add",
        "list",
        "complete",
        "delete",
        "update",
        "research",
        "note",
        "recall",
        "agenda",
        "breakdown",
        "plan",
        "review",
    ]


class TaskInput(BaseModel):
    title: str
    details: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high"]] = "medium"
    due_at: Optional[str] = None
    # If the task repeats, how often. complete_todo spawns the next occurrence.
    recurrence: Optional[Literal["daily", "weekly", "monthly"]] = None


class TaskUpdateInput(BaseModel):
    """Fields the LLM extracts when the user wants to modify an existing task."""

    title: Optional[str] = None
    new_title: Optional[str] = None
    new_priority: Optional[Literal["low", "medium", "high"]] = None
    new_due_at: Optional[str] = None
    new_details: Optional[str] = None


class TaskSelector(BaseModel):
    """Which existing task(s) a complete/delete request refers to."""

    title: Optional[str] = None
    match_all: bool = False


class NoteInput(BaseModel):
    """A free-form personal fact the user wants the assistant to remember."""

    content: str
    category: Optional[str] = None


class SubtaskInput(BaseModel):
    """One step of a broken-down goal.

    Carries the fields that make a step schedulable. Subtasks used to be bare
    strings, so a breakdown asked to "schedule these across next week" could
    only put the days *inside the title* — "Monday: draft the outline" — which
    reads fine and sorts, reminds, rolls over and shows on the agenda not at
    all. A date the user is meant to act on has to be a date field.
    """

    title: str
    priority: Optional[Literal["low", "medium", "high"]] = "medium"
    due_at: Optional[str] = None


class BreakdownOutput(BaseModel):
    """An LLM-proposed split of a larger goal into ordered subtasks."""

    parent_title: str
    subtasks: List[SubtaskInput] = []


class ResearchOutput(BaseModel):
    summary: str
    key_points: List[str] = []
    sources: List[str] = []


class PlanBlock(BaseModel):
    """One slot in a day plan."""

    # Local wall-clock, "HH:MM". Not a full timestamp: the plan is for today by
    # construction, and a date would only invite the model to invent one.
    start: str
    end: str
    title: str
    # Which existing to-do this block is for, when it is for one. Breaks and
    # buffers are legitimately task-less.
    task_id: Optional[str] = None
    reason: Optional[str] = None


class PlanOutput(BaseModel):
    """A time-blocked plan for the day, ordered by start time."""

    blocks: List[PlanBlock] = []
    # Tasks that did not fit, so the reply can say so rather than quietly
    # dropping them.
    unscheduled: List[str] = []
    notes: Optional[str] = None


class ReviewOutput(BaseModel):
    """A look back over a period of finished work."""

    summary: str
    highlights: List[str] = []
    # Things still open that the user may want to carry into next week.
    carry_over: List[str] = []
    suggestions: List[str] = []


class SynthesisOutput(BaseModel):
    response: str


class PAMemoryExtract(BaseModel):
    """Durable personal facts that help the assistant tailor its help. Every
    field is optional — only fill one when the message gives clear evidence."""

    work_hours: Optional[str] = None  # e.g. "9-5 weekdays"
    timezone: Optional[str] = None
    priorities: Optional[List[str]] = None  # standing focuses, e.g. "fitness", "job hunt"
    communication_style: Optional[str] = None  # e.g. "brief", "detailed"
    recurring_commitments: Optional[List[str]] = None  # e.g. "gym Mon/Wed/Fri"


PA_MEMORY_INSTRUCTIONS = (
    "Extract durable personal facts about the user that would help an assistant "
    "tailor its help — typical work hours, timezone, standing priorities/focuses, "
    "preferred communication style (brief/detailed), and recurring commitments. "
    "Do not extract one-off tasks or transient requests."
)
