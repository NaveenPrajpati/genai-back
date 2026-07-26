# Agent System — Supervisor, Subgraphs, and MCP

How the three agents became one assistant: a supervisor graph that reaches two skills
as **LangGraph subgraphs** and the third over the **Model Context Protocol**.

- **Companion doc:** [RAG_SYSTEM.md](RAG_SYSTEM.md) (the retrieval subsystem).
- **Code root:** [`app/agents/supervisor/`](app/agents/supervisor/) (orchestration),
  [`app/mcp/`](app/mcp/) (MCP server + client),
  [`app/routers/supervisor.py`](app/routers/supervisor.py) (HTTP).

---

## 1. What it is

One conversational endpoint over three previously separate agents — a learning tracker,
a personal assistant, and a meal planner. A message is routed to the skills that can
serve it, those skills run in order, and a single reply comes back in one voice.

The per-agent APIs (`/learning`, `/personal-assistant`, `/meal-planner`) are unchanged
and still work. The supervisor is an additional surface, not a migration.

---

## 2. Architecture at a glance

```
POST /api/assistant/query[/stream]
      │
      ▼
  load_context ──▶ route ──┬─▶ learning_agent  ─┐   LangGraph subgraph
                           ├─▶ assistant_agent ─┤   LangGraph subgraph
                           ├─▶ meal_agent ──────┤   MCP client ─▶ /mcp
                           │                    │
                           │   ◀── dispatch ◀───┘   (next queued skill)
                           ▼
                        finalize ──▶ END          (token-streamed reply)
```

`route` returns an **ordered queue** of skills. `dispatch` is a conditional edge that
walks it — running whatever is in `route` but not yet in `completed` — and falls through
to `finalize` when the queue is empty.

---

## 3. Three transports, on purpose

| Skill | Reached via | Why |
|---|---|---|
| learning | compiled subgraph | Rich internal state and its own HITL approval |
| assistant | compiled subgraph | Same, plus it already synthesises its own prose |
| meal | MCP over streamable HTTP | Proves the domain is a separable service, and exposes it to any MCP client |

### Subgraphs (learning, assistant)

Each is compiled **without a checkpointer** and invoked inside a node. A subgraph
inherits the parent's checkpointer, and that inheritance is what makes the interesting
behaviour work: an `interrupt()` raised deep inside the learning tracker's roadmap
approval pauses **the supervisor thread**, and a single `Command(resume=...)` on the
parent resumes it. The state schemas differ, so each node maps supervisor state in and
a compact summary out — the full roadmap never reaches the finalizer's context.

### MCP (meal)

`app/agents/supervisor/workflow.py` does not import the meal agent at all. It loads
tools from the MCP server over HTTP, so pointing `MEAL_MCP_URL` at another host moves
the meal planner out of process with no code change.

---

## 4. One skill per node, not a loop inside one node

Each skill is its own graph node. LangGraph checkpoints per node, so when a later skill
pauses for approval, resuming replays **only that node** — skills that already ran are
neither re-executed nor re-billed.

Running the queue inside a single node would redo all of them on every resume. This is
the main reason the dispatcher exists as an edge rather than a `for` loop.

---

## 5. Human-in-the-loop across a protocol boundary

MCP is stateless request/response: a tool call has nowhere to pause for a human. So the
meal planner splits plan creation in two:

- `propose_meal_plan` — generates, persists **nothing**. Safe for the model to call.
- `save_meal_plan` — persists an already-approved plan. **Withheld** from the tool loop
  (`AUTONOMOUS_TOOLS` in [`app/mcp/client.py`](app/mcp/client.py)).

The node runs the tool loop, extracts any proposal from the transcript, raises
`interrupt()`, and calls `save_meal_plan` itself once the approval resolves.

**Replay safety.** Resuming replays the node from the top, so before running anything it
checks the approval store for a pending proposal on this thread and reuses it — the user
gets saved the plan they approved, not a freshly generated different one. The same
pattern the meal graph's own `plan_agent` uses.

---

## 6. Identity over MCP

Every meal tool acts on one user's data, but `user_id` is deliberately **not a tool
argument** — a model that can name the tenant it reads is a model that can hallucinate
or be talked into a different one. Identity travels out-of-band like a bearer token:

- **streamable HTTP** — `X-User-Id` header, set per user by the client
- **stdio** — `MCP_USER_ID` env var (single-user: Claude Desktop, MCP Inspector)

Tools that take a `plan_id` still verify ownership, so a guessed id from any client is
rejected rather than trusted. Both properties are pinned by tests in
[`tests/test_mcp_meal_server.py`](tests/test_mcp_meal_server.py).

---

## 7. API

| Endpoint | Purpose |
|---|---|
| `POST /api/assistant/query` | Run a turn; returns `done` or `needs_approval` |
| `POST /api/assistant/query/stream` | SSE: `step` per node, `token` per word of the reply |
| `POST /api/assistant/approve` | Resolve whichever skill paused this thread |
| `GET /api/assistant/approvals` | Everything awaiting a decision, any skill |
| `GET /api/assistant/skills` | Capability list + live MCP reachability |
| `/mcp/` | The meal planner as an MCP server |

The stream uses **two LangGraph stream modes at once**: `updates` drives per-node
progress, `messages` streams the final reply. Only `finalize`'s tokens are forwarded —
every other node emits structured output or tool-call JSON.

Approvals from all three skills surface identically, because the pause is a property of
the supervisor thread rather than of the skill that raised it.

---

## 8. Cost control

- **Routing** runs on the fast model behind the same semantic cache the per-agent
  classifiers use (routing depends only on the message, so the cache is global).
- **Memory extraction** runs only for the skills that actually ran — a meal turn pays
  for one extraction, not three; a pure-chat turn pays for none.
- **Skipped skills** cost nothing: `dispatch` never enters a node it does not need.
- **The finalizer** receives compact per-skill summaries, not raw agent state.

---

## 9. Using the MCP server from an external client

Mounted at `/mcp` of the main app. Standalone:

```bash
MCP_USER_ID=<uid> python -m app.mcp.meal_server     # stdio (Claude Desktop, Inspector)
python -m app.mcp.meal_server --http                # streamable HTTP on :8100
```

Nine tools: `list_meal_plans`, `get_meal_plan`, `get_grocery_list`,
`get_food_preferences`, `propose_meal_plan`, `suggest_meals`, `save_meal_plan`,
`log_meal`, `add_food_dislike`.

---

## 10. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MEAL_MCP_URL` | `http://127.0.0.1:8000/mcp/` | Where the client finds the server (trailing slash matters) |
| `MEAL_MCP_TIMEOUT` | `60` | Per-request timeout, seconds |
| `MCP_AUTH_TOKEN` | unset | Bearer token for `/mcp`; unset means no check |
| `MCP_USER_ID` | unset | stdio-only identity fallback |

`MCP_AUTH_TOKEN` unset is only acceptable when the port is unreachable from outside the
host; the app logs a warning at startup when it is.

---

## 11. Failure behaviour

| Failure | Result |
|---|---|
| MCP server unreachable | Meal skill returns an error; the other skills and the reply still work |
| Router returns no skill | Supervisor answers directly from the conversation |
| Server restarts mid-approval | `/approve` returns 404 asking the user to re-submit |
| Approval belongs to another user | 403 before the graph is touched |
