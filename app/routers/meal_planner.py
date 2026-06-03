from typing import TypedDict
from fastapi import APIRouter
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from app.core.config import supabase

load_dotenv()

mealRouter = APIRouter(
    prefix="/meal-planner",
    tags=["summarizer"],
    responses={404: {"description": "Not found"}},
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class QueryRequest(BaseModel):
    text: str


class ProfileState(TypedDict):
    display_name: str
    diet: str
    protein_target: int


class PlannerState(TypedDict):
    query: str
    profile: ProfileState


graph = StateGraph(PlannerState)


class ProfileOutput(BaseModel):
    display_name: str
    diet: str
    protein_target: int


def insertDataindb(data: ProfileOutput):
    try:

        supabase.table("profiles").insert(
            {
                "display_name": data.display_name,
                "diet": data.diet,
                "protein_target": data.protein_target,
            }
        ).execute()
    except FileNotFoundError:
        print(FileNotFoundError)


def classify_intent(state: PlannerState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at extracting user profile info. Extract name, diet preference, and daily protein target (in grams) from the text.",
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | llm.with_structured_output(ProfileOutput)
    result: ProfileOutput = chain.invoke({"text": state["query"]})
    print(result)
    insertDataindb(result)
    return {
        "profile": {
            "display_name": result.display_name,
            "diet": result.diet,
            "protein_target": result.protein_target,
        }
    }


graph.add_node("classify_intent", classify_intent)
graph.add_edge(START, "classify_intent")
graph.add_edge("classify_intent", END)

agent = graph.compile()


@mealRouter.post("/query")
async def summarize(body: QueryRequest):
    result = agent.invoke({"query": body.text})
    return {"profile": result["profile"]}
