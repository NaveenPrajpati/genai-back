from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.routers import (
    emailassistant,
    learning_tracker,
    recipegenerator,
    users,
    webscraping,
    personal_assistant,
)
from app.database import connect_db, close_db
from app.routers import rag
from app.routers import chat
from app.routers.learning_tracker import graph, run_triggers
from app.routers.personal_assistant import graph as pa_graph, run_pa_triggers
from app.services.user_service import cleanup_expired_guests
from app.database import get_db
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver
import redis.asyncio as redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

redis_client = None
load_dotenv()

DB_URI = os.environ["DATABASE_URL"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP PHASE ---
    print("[Lifespan] Starting application initialization...")

    scheduler = AsyncIOScheduler()

    # 1. Initialize MongoDB
    try:
        await connect_db()
        # TTL index so MongoDB auto-deletes guest accounts after their expires_at
        await get_db()["users"].create_index(
            "expires_at", expireAfterSeconds=0, sparse=True
        )
        print("[Lifespan] MongoDB TTL index on users.expires_at ready")
    except Exception as e:
        print(f"MongoDB connection failed (non-fatal): {e}")

    # 2. Initialize Postgres Checkpointer
    checkpointer_context = None
    try:
        # We manually enter the context manager to keep it alive across the yield
        checkpointer_context = AsyncPostgresSaver.from_conn_string(DB_URI)
        checkpointer = await checkpointer_context.__aenter__()

        # CRUCIAL: Must await setup() to create necessary tables
        await checkpointer.setup()

        app.state.agent = graph.compile(checkpointer=checkpointer)
        app.state.pa_agent = pa_graph.compile(checkpointer=checkpointer)
        print("Using PostgresSaver checkpointer")

    except Exception as e:
        print(f"PostgresSaver failed, falling back to MemorySaver: {e}")
        if checkpointer_context:
            await checkpointer_context.__aexit__(None, None, None)
            checkpointer_context = None

        app.state.agent = graph.compile(checkpointer=MemorySaver())
        app.state.pa_agent = pa_graph.compile(checkpointer=MemorySaver())

    # Daily learning digest: bullet-point tips on each user's active topic at 09:00.
    scheduler.add_job(
        run_triggers,
        CronTrigger(hour=9, minute=0),
        args=[app.state.agent],
    )
    # Personal-assistant daily task digest at 08:00.
    scheduler.add_job(
        run_pa_triggers,
        CronTrigger(hour=8, minute=0),
        args=[app.state.pa_agent],
    )
    # Failsafe: sweep any guests the TTL index missed (e.g. during downtime)
    scheduler.add_job(cleanup_expired_guests, "interval", hours=1)
    scheduler.start()

    # --- ACTIVE PHASE ---
    yield

    # --- SHUTDOWN PHASE ---
    # Clean up Postgres checkpointer if it was successfully running
    if checkpointer_context and not isinstance(
        app.state.agent.checkpointer, MemorySaver
    ):
        print("Closing Postgres checkpointer connection...")
        await checkpointer_context.__aexit__(None, None, None)

    # Clean up MongoDB
    print("Closing MongoDB connection...")
    await close_db()
    scheduler.shutdown()


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
app.include_router(prefix="/api", router=learning_tracker.mealRouter)
app.include_router(prefix="/api", router=personal_assistant.paRouter)
app.include_router(prefix="/api", router=webscraping.router)
app.include_router(prefix="/api", router=emailassistant.router)
app.include_router(prefix="/api", router=recipegenerator.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
