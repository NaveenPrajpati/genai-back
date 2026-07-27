import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.routers import (
    learning_tracker,
    users,
    personal_assistant,
    meal_planner,
    supervisor,
    rag,
    chat,
)
from app.database import connect_db, close_db
from app.core.observability import init_tracing
from app.agents.learning_tracker import graph, run_triggers
from app.agents.personal_assistant import graph as pa_graph, run_pa_triggers
from app.agents.meal_planner import (
    graph as meal_graph,
    run_triggers as run_meal_triggers,
)
from app.agents.supervisor import graph as supervisor_graph
from app.mcp.meal_planner import mcp as meal_mcp
from app.mcp.mount import guarded_app, session_lifespan
from app.services.user_service import deactivate_expired_guests
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver
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
        # The supervisor embeds the learning and PA graphs as subgraphs. They are
        # compiled inside it WITHOUT a checkpointer and inherit this one, which is
        # what lets an approval raised deep in a subgraph pause the supervisor
        # thread and resume from a single Command on the parent.
        app.state.supervisor_agent = supervisor_graph.compile(checkpointer=checkpointer)
        print("Using PostgresSaver checkpointer")

    except Exception as e:
        print(f"PostgresSaver failed, falling back to MemorySaver: {e}")
        if checkpointer_context:
            await checkpointer_context.__aexit__(None, None, None)
            checkpointer_context = None

        app.state.learning_agent = graph.compile(checkpointer=MemorySaver())
        app.state.pa_agent = pa_graph.compile(checkpointer=MemorySaver())
        app.state.meal_agent = meal_graph.compile(checkpointer=MemorySaver())
        app.state.supervisor_agent = supervisor_graph.compile(checkpointer=MemorySaver())

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

    # Warm the RAG retrieval stack (BM25 params + nltk data download, Pinecone
    # handle) so the first user query doesn't pay cold-start latency and 50 cold
    # queries don't each trigger the download. Best-effort + time-bounded: on a
    # slow/failing dependency we log and continue, and the first query lazy-loads
    # as it does today — a flaky Pinecone/download must not fail the boot.
    try:
        from app.core.config import RAG_WARMUP_TIMEOUT
        from app.services.rag.step4_retrieval import warmup as _rag_warmup

        await asyncio.wait_for(
            asyncio.to_thread(_rag_warmup), timeout=RAG_WARMUP_TIMEOUT
        )
        print("[Lifespan] RAG retrieval stack warmed")
    except Exception as e:
        print(f"[Lifespan] RAG warmup skipped (non-fatal): {e}")

    # The meal planner is mounted at /mcp as a Model Context Protocol server.
    # FastAPI does not run a mounted sub-app's lifespan, so its streamable-HTTP
    # session manager has to be started here or every MCP request fails.
    mcp_stack = AsyncExitStack()
    try:
        await mcp_stack.enter_async_context(session_lifespan(meal_mcp))
        print("[Lifespan] Meal MCP server ready at /mcp")
    except Exception as e:
        print(f"[Lifespan] MCP server failed to start (meal skill degraded): {e}")

    # --- ACTIVE PHASE ---
    yield

    # --- SHUTDOWN PHASE ---
    await mcp_stack.aclose()

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
app.include_router(prefix="/api", router=supervisor.router)

# The meal planner as a Model Context Protocol server. The supervisor consumes
# it over this endpoint like any other MCP client would; point an external
# client (Claude Desktop, the MCP Inspector) here too. Additional servers mount
# alongside this one and join the session_lifespan(...) call. See app/mcp/mount.py.
app.mount("/mcp", guarded_app(meal_mcp))


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request):
    """Prometheus scrape endpoint (ops metrics — see core/metrics.py).

    Exposed on the app itself (single worker, so one in-process registry). Keep it
    off the public internet: restrict to the monitoring host via the EC2 security
    group, and/or set METRICS_TOKEN to require `Authorization: Bearer <token>`
    (Prometheus sends it via `authorization` in its scrape config)."""
    token = os.getenv("METRICS_TOKEN")
    if token and request.headers.get("authorization") != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="unauthorized")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
