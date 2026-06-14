import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

router = APIRouter(
    prefix="/recipe-generator",
    tags=["recipe-generator"],
    responses={404: {"description": "Not found"}},
)


@tool
def generate_recipe(
    ingredients: str,
    cuisine: str = "Indian",
    diet: str = "veg",
) -> str:
    """Generate a recipe based on available ingredients, cuisine type, and dietary preference.
    - ingredients: comma-separated list of available ingredients
    - cuisine: cuisine type (Indian, Chinese, Italian, Mexican, Thai, Japanese, Continental, etc.)
    - diet: dietary preference - 'veg' for vegetarian or 'nonveg' for non-vegetarian
    """
    return (
        f"RECIPE_REQUEST:\n"
        f"Ingredients: {ingredients}\n"
        f"Cuisine: {cuisine}\n"
        f"Diet: {diet}"
    )


@tool
def suggest_substitution(ingredient: str, cuisine: str = "Indian") -> str:
    """Suggest substitutions for a missing ingredient within a specific cuisine context.
    Use this when the user is missing an ingredient and needs an alternative."""
    return f"SUBSTITUTION_REQUEST: {ingredient} in {cuisine} cooking"


@tool
def get_nutritional_info(recipe_name: str) -> str:
    """Provide approximate nutritional information for a recipe.
    Use this when the user asks about calories, protein, or nutritional value."""
    return f"NUTRITION_REQUEST: {recipe_name}"


model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

agent = create_agent(
    model=model,
    tools=[generate_recipe, suggest_substitution, get_nutritional_info],
    system_prompt=(
        "You are a world-class chef and recipe creator. You generate detailed, "
        "delicious recipes based on the ingredients users have available.\n\n"
        "Rules:\n"
        "1. ONLY use the ingredients the user provides. You may assume basic pantry "
        "staples are available (salt, pepper, oil, water, common spices).\n"
        "2. Respect dietary preferences strictly — if 'veg' is selected, NEVER "
        "include meat, fish, or eggs unless specifically listed in ingredients.\n"
        "3. Match the cuisine style authentically.\n"
        "4. Always use the generate_recipe tool first to structure your response.\n\n"
        "Recipe format:\n"
        "- Recipe name (creative and appetizing)\n"
        "- Prep time & cook time\n"
        "- Servings\n"
        "- Ingredients list with quantities\n"
        "- Step-by-step instructions (clear and numbered)\n"
        "- Pro tips for best results\n\n"
        "Be enthusiastic about cooking and make the recipe sound delicious!"
    ),
)


class RecipeRequest(BaseModel):
    ingredients: str
    cuisine: str = "Indian"
    diet: str = "veg"


INGREDIENT_SUGGESTIONS = {
    "vegetables": [
        "Potato",
        "Tomato",
        "Onion",
        "Garlic",
        "Ginger",
        "Green Chili",
        "Capsicum",
        "Carrot",
        "Cauliflower",
        "Broccoli",
        "Spinach",
        "Peas",
        "Corn",
        "Mushroom",
        "Cabbage",
        "Eggplant",
        "Okra",
        "Bottle Gourd",
        "Bitter Gourd",
        "Ridge Gourd",
        "Pumpkin",
        "Sweet Potato",
        "Beetroot",
        "Radish",
        "Cucumber",
        "Zucchini",
        "Bell Pepper",
        "Spring Onion",
        "Lettuce",
        "Avocado",
        "Asparagus",
        "Artichoke",
        "Celery",
        "Leek",
    ],
    "fruits": [
        "Lemon",
        "Lime",
        "Mango",
        "Banana",
        "Apple",
        "Orange",
        "Coconut",
        "Pineapple",
        "Tamarind",
        "Pomegranate",
    ],
    "dairy": [
        "Milk",
        "Curd",
        "Paneer",
        "Butter",
        "Ghee",
        "Cream",
        "Cheese",
        "Mozzarella",
        "Parmesan",
        "Yogurt",
        "Cottage Cheese",
        "Whipped Cream",
        "Condensed Milk",
    ],
    "proteins_veg": [
        "Tofu",
        "Soy Chunks",
        "Chickpeas",
        "Lentils",
        "Kidney Beans",
        "Black Beans",
        "Green Gram",
        "Bengal Gram",
        "Peanuts",
        "Cashew",
        "Almond",
        "Walnut",
        "Sesame Seeds",
        "Flax Seeds",
    ],
    "proteins_nonveg": [
        "Chicken",
        "Chicken Breast",
        "Chicken Thigh",
        "Mutton",
        "Lamb",
        "Fish",
        "Prawns",
        "Shrimp",
        "Eggs",
        "Egg White",
        "Salmon",
        "Tuna",
        "Crab",
        "Squid",
        "Pork",
        "Beef",
        "Turkey",
        "Duck",
        "Bacon",
        "Sausage",
    ],
    "grains": [
        "Rice",
        "Basmati Rice",
        "Wheat Flour",
        "Maida",
        "Bread",
        "Pasta",
        "Noodles",
        "Oats",
        "Semolina",
        "Poha",
        "Quinoa",
        "Couscous",
        "Tortilla",
        "Roti",
        "Puff Pastry",
    ],
    "spices": [
        "Turmeric",
        "Cumin",
        "Coriander Powder",
        "Red Chili Powder",
        "Garam Masala",
        "Mustard Seeds",
        "Curry Leaves",
        "Bay Leaf",
        "Cinnamon",
        "Cardamom",
        "Cloves",
        "Black Pepper",
        "Fennel Seeds",
        "Fenugreek",
        "Asafoetida",
        "Saffron",
        "Oregano",
        "Basil",
        "Thyme",
        "Rosemary",
        "Paprika",
        "Star Anise",
        "Nutmeg",
    ],
    "sauces_condiments": [
        "Soy Sauce",
        "Tomato Ketchup",
        "Vinegar",
        "Mustard",
        "Sriracha",
        "Hot Sauce",
        "Worcestershire Sauce",
        "Fish Sauce",
        "Oyster Sauce",
        "Hoisin Sauce",
        "Tahini",
        "Pesto",
        "Coconut Milk",
        "Cream Cheese",
    ],
}


@router.get("/suggestions")
async def get_suggestions(diet: str = "veg"):
    """Return ingredient suggestions filtered by dietary preference."""
    suggestions = {}
    for category, items in INGREDIENT_SUGGESTIONS.items():
        if category == "proteins_nonveg" and diet == "veg":
            continue
        suggestions[category] = items
    return {"suggestions": suggestions}


@router.post("/generate")
async def generate(request: RecipeRequest):
    user_message = (
        f"Generate a {request.diet} {request.cuisine} recipe using these ingredients: "
        f"{request.ingredients}"
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    ai_message = result["messages"][-1].content
    return {"recipe": ai_message}


@router.post("/generate/stream")
async def generate_stream(request: RecipeRequest):
    user_message = (
        f"Generate a {request.diet} {request.cuisine} recipe using these ingredients: "
        f"{request.ingredients}"
    )

    async def event_generator():
        try:
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": user_message}]},
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        data = json.dumps({"token": chunk.content})
                        yield f"data: {data}\n\n"

                elif kind == "on_tool_start":
                    tool_name = event.get("name", "tool")
                    status_map = {
                        "generate_recipe": "Cooking up a recipe...",
                        "suggest_substitution": "Finding substitutions...",
                        "get_nutritional_info": "Calculating nutrition...",
                    }
                    status = status_map.get(tool_name, f"Using {tool_name}...")
                    data = json.dumps({"status": status})
                    yield f"data: {data}\n\n"

                elif kind == "on_tool_end":
                    data = json.dumps({"status": "Preparing your recipe..."})
                    yield f"data: {data}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
