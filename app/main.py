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
    learning_tracker,
    users,
    personal_assistant,
    meal_planner,
)
from app.database import connect_db, close_db
from app.core.observability import init_tracing
from app.routers import rag
from app.routers import chat
from app.agents.learning_tracker import graph, run_triggers
from app.agents.personal_assistant import graph as pa_graph, run_pa_triggers
from app.agents.meal_planner import (
    graph as meal_graph,
    run_triggers as run_meal_triggers,
)
from app.services.user_service import deactivate_expired_guests
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

    # Opt-in LangSmith tracing (no-op unless LANGSMITH_TRACING is set).
    init_tracing()

    scheduler = AsyncIOScheduler()

    # 1. Initialize MongoDB
    try:
        await connect_db()
        # NOTE: guests are now *deactivated* at expiry (not deleted), so we no
        # longer create a TTL index on users.expires_at. If a previous deploy
        # created one, drop it by hand or it will keep hard-deleting guests:
        #   db.users.dropIndex("expires_at_1")
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

        app.state.learning_agent = graph.compile(checkpointer=checkpointer)
        app.state.pa_agent = pa_graph.compile(checkpointer=checkpointer)
        app.state.meal_agent = meal_graph.compile(checkpointer=checkpointer)
        print("Using PostgresSaver checkpointer")

    except Exception as e:
        print(f"PostgresSaver failed, falling back to MemorySaver: {e}")
        if checkpointer_context:
            await checkpointer_context.__aexit__(None, None, None)
            checkpointer_context = None

        app.state.learning_agent = graph.compile(checkpointer=MemorySaver())
        app.state.pa_agent = pa_graph.compile(checkpointer=MemorySaver())
        app.state.meal_agent = meal_graph.compile(checkpointer=MemorySaver())

    # Learning digest: hourly sweep. run_triggers fires per user only when the
    # current hour matches their chosen schedule_hour in their own timezone.
    scheduler.add_job(
        run_triggers,
        CronTrigger(minute=0),
        args=[app.state.learning_agent],
    )
    # Personal-assistant task digest: hourly sweep, fires per user at their
    # chosen schedule_hour/timezone (default 08:00 daily).
    scheduler.add_job(
        run_pa_triggers,
        CronTrigger(minute=0),
        args=[app.state.pa_agent],
    )
    # Meal-planner plan generation: hourly sweep, fires per user at their chosen
    # schedule_dow/schedule_hour/timezone (default Sunday 18:00).
    scheduler.add_job(
        run_meal_triggers,
        CronTrigger(minute=0),
        args=[app.state.meal_agent],
    )
    # Hourly sweep: deactivate (not delete) guests past their 24h deadline.
    scheduler.add_job(deactivate_expired_guests, "interval", hours=1)
    scheduler.start()

    # --- ACTIVE PHASE ---
    yield

    # --- SHUTDOWN PHASE ---
    # Clean up Postgres checkpointer if it was successfully running
    if checkpointer_context and not isinstance(
        app.state.learning_agent.checkpointer, MemorySaver
    ):
        print("Closing Postgres checkpointer connection...")
        await checkpointer_context.__aexit__(None, None, None)

    # Clean up MongoDB
    print("Closing MongoDB connection...")
    await close_db()
    scheduler.shutdown()


app = FastAPI(
    lifespan=lifespan,
    title="My Awesome API",
    version="1.0.0",
    swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}},
)

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
app.include_router(prefix="/api", router=learning_tracker.router)
app.include_router(prefix="/api", router=personal_assistant.router)
app.include_router(prefix="/api", router=meal_planner.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
