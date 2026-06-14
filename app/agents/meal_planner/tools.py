"""External tool calls (nutrition lookup) for the meal-planner research agent."""

import logging
import os

import httpx
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from app.core.llm import llm

logger = logging.getLogger(__name__)


@tool
async def get_nutrition(ingredients: list[str]) -> dict:
    """Fetch accurate nutrition data for a recipe from the Edamam API.
    Call this for every meal you suggest.
    Pass ingredients with quantities e.g. ['200g chicken breast', '1 cup rice'].
    Returns calories, protein_g, carbs_g, fat_g for the full recipe."""
    app_id = os.getenv("EDAMAM_APP_ID", "")
    app_key = os.getenv("EDAMAM_APP_KEY", "")
    if not app_id or not app_key:
        return {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.edamam.com/api/nutrition-details",
                params={"app_id": app_id, "app_key": app_key},
                json={"ingr": ingredients},
            )
            if resp.status_code != 200:
                logger.error(f"Edamam error {resp.status_code}: {resp.text}")
                return {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}
            data = resp.json()
            n = data.get("totalNutrients", {})
            return {
                "calories": round(data.get("calories", 0), 1),
                "protein_g": round(n.get("PROCNT", {}).get("quantity", 0), 1),
                "carbs_g": round(n.get("CHOCDF", {}).get("quantity", 0), 1),
                "fat_g": round(n.get("FAT", {}).get("quantity", 0), 1),
            }
    except Exception as e:
        logger.error(f"nutrition API error: {e}")
        return {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}


research_tools = [get_nutrition]
research_tool_node = ToolNode(research_tools)
research_llm = llm.bind_tools(research_tools)
