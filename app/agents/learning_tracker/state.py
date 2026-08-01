"""State and structured-output schemas for the learning-tracker agent.

Two families of model live here, and the split is load-bearing:

* **Draft models** (`RoadmapDraft`, `TopicDraft`, …) are what the LLM is asked to
  produce. They carry curriculum *content* only — no ids, no lifecycle status, no
  progress, no timestamps. A model that can emit `progress_status` can also
  silently reset it on a roadmap edit, so those fields are deliberately out of
  its reach.
* **Persisted models** (`RoadmapOutput`, `TopicNode`, …) are what the repository
  writes to Mongo. Identity, lifecycle, and progress are assigned server-side by
  `materialize_roadmap` / `merge_roadmap`.

The single exception is `TopicDraft.existing_id`: on a *modify* the prompt shows
the model each stored topic's id and asks it to echo that id back for topics it
kept. That is what lets `merge_roadmap` carry a topic's progress across an edit.
It is validated against the ids on that one roadmap — an unknown, duplicated, or
foreign id is treated as a brand-new topic, never as a way to address other data.

Timestamps are ISO-8601 strings, matching approval_store / trigger_store and
keeping every payload JSON-safe for the SSE stream.
"""

from typing import TypedDict, Optional, Literal

from pydantic import BaseModel, Field

ResourceType = Literal[
    "article",
    "video",
    "course",
    "documentation",
    "book",
    "exercise",
    "project",
    "other",
]
Difficulty = Literal["beginner", "intermediate", "advanced"]
ProgressStatus = Literal[
    "not_started", "in_progress", "needs_review", "completed", "skipped"
]
RoadmapStatus = Literal["draft", "active", "paused", "archived", "completed"]

# A topic in one of these states is behind the learner: `active_topic` skips it.
DONE_STATUSES: frozenset[str] = frozenset({"completed", "skipped"})


# ── what the LLM generates ───────────────────────────────────────────────────
class ResourceDraft(BaseModel):
    title: str
    url: Optional[str] = None
    resource_type: ResourceType = "other"
    is_required: bool = False


class StageDraft(BaseModel):
    order: int
    title: str
    description: Optional[str] = None


class TopicDraft(BaseModel):
    order: int
    # Links to StageDraft.order. The model orders stages; the server ids them.
    stage_order: Optional[int] = None
    title: str
    description: str
    learning_outcomes: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    estimated_minutes: Optional[int] = None
    difficulty: Optional[Difficulty] = None
    resources: list[ResourceDraft] = Field(default_factory=list)
    # Modify path only: the id of the stored topic this one continues, so its
    # progress survives the edit. Null on creation and for genuinely new topics.
    existing_id: Optional[str] = None


class RoadmapDraft(BaseModel):
    title: str
    summary: str
    learner_goal: Optional[str] = None
    target_date: Optional[str] = None
    total_estimated_hours: Optional[int] = None
    stages: list[StageDraft] = Field(default_factory=list)
    topics: list[TopicDraft] = Field(default_factory=list)


# ── what the server persists ─────────────────────────────────────────────────
class Resource(BaseModel):
    title: str
    url: Optional[str] = None
    resource_type: ResourceType = "other"
    is_required: bool = False


class Stage(BaseModel):
    id: str
    order: int
    title: str
    description: Optional[str] = None


class TopicNode(BaseModel):
    id: str
    stage_id: Optional[str] = None
    order: int

    title: str
    description: str
    learning_outcomes: list[str] = Field(default_factory=list)

    prerequisites: list[str] = Field(default_factory=list)
    estimated_minutes: Optional[int] = None
    difficulty: Optional[Difficulty] = None

    resources: list[Resource] = Field(default_factory=list)

    # Server-owned. Never populated from a model response — see module docstring.
    progress_status: ProgressStatus = "not_started"
    mastery_score: Optional[int] = Field(default=None, ge=0, le=100)
    completed_at: Optional[str] = None
    # Spaced repetition: consecutive checkpoint passes, and when this topic
    # should resurface. A failed review resets the count, pulling the next
    # review back to the start of the ladder.
    review_count: int = 0
    next_review_at: Optional[str] = None


class RoadmapOutput(BaseModel):
    """One roadmap as stored. Mongo's `_id` is the identifier — there is no
    second `id` field to drift out of sync with it."""

    title: str
    summary: str
    status: RoadmapStatus = "active"
    learner_goal: Optional[str] = None
    target_date: Optional[str] = None
    total_estimated_hours: Optional[int] = None
    stages: list[Stage] = Field(default_factory=list)
    topics: list[TopicNode] = Field(default_factory=list)

    # Snapshot of the learner-profile fields that shaped this roadmap, taken at
    # generation time. Kept so the UI can show *what* was personalized instead of
    # asserting that something was, and so we can tell when the profile has moved
    # on. None means "generated before we recorded this" — unknown, not empty.
    personalization: Optional[dict] = None

    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class Question(BaseModel):
    question: str
    options: list[str]
    answer: int


class QuizOutput(BaseModel):
    quiz: list[Question]


class CheckpointOutcome(BaseModel):
    """What a graded checkpoint did to the topic. Returned to the client so the
    UI can explain the result rather than silently ticking (or not ticking) a box."""

    passed: bool
    score: int
    pass_score: int
    total: int
    correct: int
    review: list[dict] = Field(default_factory=list)
    progress_status: ProgressStatus
    next_review_at: Optional[str] = None
    review_count: int = 0
    # True when this checkpoint was a scheduled review of an already-completed
    # topic, rather than the first attempt that completes it.
    was_review: bool = False


class SubmittedAnswer(BaseModel):
    """One answer parsed out of a free-text quiz submission."""

    question_index: int = Field(ge=0, description="0-based position of the question")
    selected_index: int = Field(ge=0, description="0-based position of the chosen option")


class QuizSubmission(BaseModel):
    answers: list[SubmittedAnswer] = Field(default_factory=list)


class LearningState(TypedDict, total=False):
    query: str
    intent: str
    current_user: dict
    user_id: str
    thread_id: str
    memory: dict
    plan_status: Optional[str]
    log_status: Optional[str]
    conflict: Optional[dict]
    roadmapId: Optional[str]
    suggestions: Optional[list]
    roadmap: Optional[RoadmapOutput]
    roadmap_status: Optional[str]
    # How many PA to-dos were created from this roadmap's topics (cross-agent).
    pa_tasks_created: Optional[int]
    next_topic: str
    progress: Optional[dict]
    topic_explaination: str
    quiz: list[dict]
    quizId: str
    quiz_result: Optional[dict]


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
        "chitchat",
        "fallback",
    ]


class TutorOutput(BaseModel):
    topic: str
    topic_explaination: str


class UpdateProgressOutput(BaseModel):
    topicId: Optional[str] = None


class ResearchOutput(BaseModel):
    resources: list[Resource] = Field(default_factory=list)


class LearningAvailability(BaseModel):
    minutes_per_day: Optional[int] = Field(default=None, ge=0)
    days_per_week: Optional[int] = Field(default=None, ge=1, le=7)
    preferred_days: list[str] = Field(default_factory=list)
    deadline: Optional[str] = None


class LearnerMemory(BaseModel):
    # Stable learner preferences
    skill_level: Optional[Difficulty] = None

    preferred_resource_types: list[
        Literal["video", "article", "documentation", "interactive", "project", "book"]
    ] = Field(default_factory=list)

    preferred_explanation_style: Optional[
        Literal["concise", "step_by_step", "examples_first", "visual", "socratic"]
    ] = None

    preferred_language: Optional[str] = None

    # Goals and learning context
    goals: list[str] = Field(default_factory=list)
    availability: Optional[LearningAvailability] = None

    # Knowledge the learner explicitly demonstrates or confirms
    known_topics: list[str] = Field(default_factory=list)
    weak_topics: list[str] = Field(default_factory=list)

    # Behavioural learning preferences
    wants_hints_before_answers: Optional[bool] = None
    preferred_quiz_difficulty: Optional[Literal["easy", "adaptive", "challenging"]] = (
        None
    )


# The onboarding interrupt collects exactly these keys. Anything else the client
# sends back is dropped rather than trusted into the profile.
ONBOARDING_KEYS: frozenset[str] = frozenset(
    {"skill_level", "preferred_explanation_style"}
)

# Set once onboarding has run — including when the learner skips it — so the
# interrupt never fires a second time. Lives beside the LearnerMemory fields in
# the profile doc but is intentionally NOT part of that schema: it is a server
# flag, not something an extraction model may set.
ONBOARDED_FLAG = "onboarded"


class TopicTipsOutput(BaseModel):
    bullets: list[str]
