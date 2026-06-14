"""State and structured-output schemas for the learning-tracker agent."""

from typing import TypedDict, Optional, List, Literal
from pydantic import BaseModel


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
    title: str
    summary: str
    status: Literal["active", "archived", "completed"] = "archived"
    total_estimated_hours: Optional[int] = None
    stages: List[str]
    topics: List[TopicNode]


class Question(BaseModel):
    question: str
    options: list[str]
    answer: int


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
    progress: Optional[dict]
    topic_explaination: str
    quiz: list[dict]
    quizId: str
    active_topic: TopicNode


class IntentOutput(BaseModel):
    intent: Literal[
        "create_roadmap",
        "explain",
        "quiz",
        "submit_quiz",
        "find_resources",
        "update_progress",
        "query_roadmap",
        "modify_roadmap",
    ]


class TutorOutput(BaseModel):
    topic: str
    topic_explaination: str


class UpdateProgressOutput(BaseModel):
    topicId: Optional[str] = None


class ResearchOutput(BaseModel):
    resources: list[str]


class MemoryExtract(BaseModel):
    """Durable learner facts worth remembering across sessions. Every field is
    optional — only fill one when the message gives clear evidence for it."""

    skill_level: Optional[str] = None  # beginner | intermediate | advanced
    preferred_resource_types: Optional[List[str]] = None  # video | text | interactive
    goals: Optional[List[str]] = None
    availability: Optional[str] = None  # e.g. "~1h/day"
    known_topics: Optional[List[str]] = None


class TopicTipsOutput(BaseModel):
    bullets: list[str]
