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


class BreakdownOutput(BaseModel):
    """An LLM-proposed split of a larger goal into ordered subtasks."""

    parent_title: str
    subtasks: List[str] = []


class ResearchOutput(BaseModel):
    summary: str
    key_points: List[str] = []
    sources: List[str] = []


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
