from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

from app.routers import (
    emailassistant,
    recipegenerator,
    users,
    webscraping,
    meal_planner,
)
from app.database import connect_db, close_db
from app.routers import rag
from app.routers import chat
import redis.asyncio as redis

redis_client = None
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prefix="/api/user", router=users.router)
app.include_router(prefix="/api", router=rag.router)
app.include_router(prefix="/api", router=chat.router)
app.include_router(prefix="/api", router=meal_planner.mealRouter)
app.include_router(prefix="/api", router=webscraping.router)
app.include_router(prefix="/api", router=emailassistant.router)
app.include_router(prefix="/api", router=recipegenerator.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
