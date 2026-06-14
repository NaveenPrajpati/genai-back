"""State and structured-output schemas for the meal-planner agent."""

from typing import TypedDict, Optional, List, Literal
from pydantic import BaseModel


class PlannerState(TypedDict, total=False):
    query: str
    intent: str
    current_user: dict
    user_id: str
    thread_id: str
    memory: dict
    plan_status: Optional[str]
    log_status: Optional[str]
    conflict: Optional[dict]
    plan_id: Optional[str]
    suggestions: Optional[list]
    meal_slots: Optional[list]


class IntentOutput(BaseModel):
    intent: str


class LogOutput(BaseModel):
    recipe: str
    day_of_week: int
    meal_type: str
    conflict: bool
    suggestion: Optional[str]


class GroceryItem(BaseModel):
    plan_id: Optional[str] = None
    name: str
    qty: Optional[float] = None
    unit: Optional[str] = None
    checked: bool = False


class RecipeOutput(BaseModel):
    name: str
    ingredients: list[GroceryItem] = []
    protein_g: Optional[int] = None
    prep_minutes: Optional[int] = None
    source_url: Optional[str] = None
    summary: Optional[str] = None


class QueryOutput(BaseModel):
    meal_type: List[str]
    time: str


class NutritionData(BaseModel):
    calories: float = 0
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0


class ResearchMeal(BaseModel):
    meal_type: str
    recipe_name: str
    ingredients: list[str]
    prep_minutes: int
    nutrition: Optional[NutritionData] = None


class ResearchOutput(BaseModel):
    suggestions: List[ResearchMeal]


class MealSlots(BaseModel):
    plan_id: Optional[str] = None
    day_of_week: int = 0
    meal_type: Literal["dinner", "lunch", "breakfast"]
    recipe_id: Optional[str] = None
    recipe_name: Optional[str] = None
    protein_g: Optional[int] = None


class PlanOutput(BaseModel):
    plan: list[MealSlots] = []
