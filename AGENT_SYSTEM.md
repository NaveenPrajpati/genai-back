# Agent System — Supervisor, Subgraphs, and MCP

How four agents became one assistant: a supervisor graph that reaches three skills
as **LangGraph subgraphs** and one over the **Model Context Protocol** — with a fourth
that is both, a subgraph whose own tool loop runs on MCP.

- **Companion doc:** [RAG_SYSTEM.md](RAG_SYSTEM.md) (the retrieval subsystem).
- **Code root:** [`app/agents/supervisor/`](app/agents/supervisor/) (orchestration),
  [`app/mcp/`](app/mcp/) (MCP servers + client),
  [`app/routers/supervisor.py`](app/routers/supervisor.py) (HTTP).

```
app/mcp/
├── mount.py              shared: bearer guard + session lifespan, any server
├── client.py             shared: per-user client, tool loading, result decoding
├── meal_planner/
│   ├── server.py         the MCP server itself
│   └── client.py         its URL + which tools a model may drive alone
└── job_helper/
    ├── server.py         14 tools over applications and resumes
    └── client.py         same shape; withholds save_tailored_resume
```

One package per server. Adding another means a new package plus two lines in
`app.main` — mount `guarded_app(<pkg>.mcp)` and add it to `session_lifespan(...)`.
Mount order matters: a Starlette mount claims everything under its prefix, so
`/mcp/jobs` is registered **before** `/mcp` or the meal planner would swallow it.

---

## 1. What it is

One conversational endpoint over four previously separate agents — a learning tracker,
a personal assistant, a job helper, and a meal planner. A message is routed to the
skills that can serve it, those skills run in order, and a single reply comes back in
one voice.

The per-agent APIs (`/learning`, `/personal-assistant`, `/job-helper`, `/meal-planner`)
are unchanged and still work. The supervisor is an additional surface, not a migration.

---

## 2. Architecture at a glance

```
POST /api/assistant/query[/stream]
      │
      ▼
  load_context ──▶ route ──┬─▶ learning_agent  ─┐   LangGraph subgraph
                           ├─▶ assistant_agent ─┤   LangGraph subgraph
                           ├─▶ job_agent ───────┤   subgraph ─▶ MCP ─▶ /mcp/jobs
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
| job | subgraph whose middle node is an MCP tool loop | Needs both: an approval that pauses the parent thread, and a tool surface external clients can drive |
| meal | MCP over streamable HTTP | Proves the domain is a separable service, and exposes it to any MCP client |

### Subgraphs (learning, assistant, job)

Each is compiled **without a checkpointer** and invoked inside a node. A subgraph
inherits the parent's checkpointer, and that inheritance is what makes the interesting
behaviour work: an `interrupt()` raised deep inside the learning tracker's roadmap
approval — or the job helper's tailored-resume approval — pauses **the supervisor
thread**, and a single `Command(resume=...)` on the parent resumes it. The state schemas
differ, so each node maps supervisor state in and a compact summary out — the full
roadmap never reaches the finalizer's context.

### MCP (meal)

`app/agents/supervisor/workflow.py` does not import the meal agent at all. It loads
tools from the MCP server over HTTP, so pointing `MEAL_MCP_URL` at another host moves
the meal planner out of process with no code change.

### Both (job)

The job helper is the one skill that is a subgraph *and* an MCP client, because the two
choices answer different questions. It is a subgraph so its approval can pause the
supervisor; the protocol hop lives one level down, inside its own `agent` node. The
supervisor needs no special handling for it — from here it looks like any other
subgraph, and `JOB_MCP_URL` still moves the skill out of process.

Its middle is a **ReAct tool loop rather than an intent classifier**, unlike the other
three. A job-search turn does not decompose into one verb: "should I apply to this?" is
read the posting, score the resume against it, and answer — three tools whose order
depends on what the first returns. The graph structure around the loop exists for the
one thing a loop cannot express: stopping for a human.

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
  (`AUTONOMOUS_TOOLS` in [`app/mcp/meal_planner/client.py`](app/mcp/meal_planner/client.py)).

The node runs the tool loop, extracts any proposal from the transcript, raises
`interrupt()`, and calls `save_meal_plan` itself once the approval resolves.

**Replay safety.** Resuming replays the node from the top, so before running anything it
checks the approval store for a pending proposal on this thread and reuses it — the user
gets saved the plan they approved, not a freshly generated different one. The same
pattern the meal graph's own `plan_agent` uses.

The job helper splits the same way — `propose_tailored_resume` writes nothing,
`save_tailored_resume` is withheld from the loop — with higher stakes behind the gate: a
tailored resume goes to an employer under the user's name, so the interrupt carries the
`changes` changelog and nothing is written until the user has read it. Its replay check
also stores the **query** alongside the proposal, and matches on it. A pending approval
outlives the turn that raised it, so matching on the thread alone would answer every
later question on that thread with the abandoned proposal.

---

## 6. Identity over MCP

Every tool on both servers acts on one user's data, but `user_id` is deliberately **not
a tool argument** — a model that can name the tenant it reads is a model that can
hallucinate or be talked into a different one. Identity travels out-of-band like a
bearer token:

- **streamable HTTP** — `X-User-Id` header, set per user by the client
- **stdio** — `MCP_USER_ID` env var (single-user: Claude Desktop, MCP Inspector)

Tools that take a `plan_id` or an `application_id` still verify ownership, so a guessed
id from any client is rejected rather than trusted. In the job helper every repository
read is scoped by the caller's id, which means a foreign id resolves to nothing rather
than to someone else's row. Both properties are pinned by tests in
[`tests/test_mcp_meal_planner.py`](tests/test_mcp_meal_planner.py) and
[`tests/test_job_helper.py`](tests/test_job_helper.py) — including one that walks every
published tool asserting none of them accepts a `user_id`.

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
| `/mcp/jobs/` | The job helper as an MCP server |

The stream uses **two LangGraph stream modes at once**: `updates` drives per-node
progress, `messages` streams the final reply. Only `finalize`'s tokens are forwarded —
every other node emits structured output or tool-call JSON.

Approvals from all four skills surface identically, because the pause is a property of
the supervisor thread rather than of the skill that raised it.

The job helper adds routes that are **not** agent turns, because files are not chat:
`POST /api/job-helper/resumes/upload` (multipart in) and
`GET /api/job-helper/resumes/{id}/download?fmt=pdf|docx` (bytes out, with a
`Content-Disposition` filename). See §12.

---

## 8. Cost control

- **Routing** runs on the fast model behind the same semantic cache the per-agent
  classifiers use (routing depends only on the message, so the cache is global).
- **Memory extraction** runs only for the skills that actually ran — a meal turn pays
  for one extraction, not three; a pure-chat turn pays for none.
- **Skipped skills** cost nothing: `dispatch` never enters a node it does not need.
- **The finalizer** receives compact per-skill summaries, not raw agent state.

---

## 9. Using the MCP servers from an external client

Both are mounted on the main app. Standalone:

```bash
MCP_USER_ID=<uid> python -m app.mcp.meal_planner     # stdio (Claude Desktop, Inspector)
python -m app.mcp.meal_planner --http                # streamable HTTP on :8100

MCP_USER_ID=<uid> python -m app.mcp.job_helper       # stdio
python -m app.mcp.job_helper --http                  # streamable HTTP on :8101
```

**meal** — nine tools: `list_meal_plans`, `get_meal_plan`, `get_grocery_list`,
`get_food_preferences`, `propose_meal_plan`, `suggest_meals`, `save_meal_plan`,
`log_meal`, `add_food_dislike`.

**jobs** — eighteen. Read: `list_applications`, `get_application`, `list_resumes`,
`get_resume_details`, `get_job_search_profile`. Find: `search_jobs`,
`track_job_posting`. Analyse: `assess_resume_fit`, `propose_tailored_resume`,
`draft_cover_letter`, `prepare_for_interview`. Write: `save_resume_details`,
`update_resume_details`, `update_application_status`, `add_application_note`,
`save_tailored_resume`. Hand off: `add_followup_tasks`, `start_roadmap_for_gap`.

All but `save_tailored_resume` are drivable by a tool loop — that one is withheld because
a tailored resume goes to an employer under the user's name. A test walks the published
list and asserts the withheld set is exactly `{save_tailored_resume}`, so a tool added
later cannot quietly become autonomous.

---

## 10. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MEAL_MCP_URL` | `http://127.0.0.1:8000/mcp/` | Where the client finds the server (trailing slash matters) |
| `MEAL_MCP_TIMEOUT` | `60` | Per-request timeout, seconds |
| `JOB_MCP_URL` | `http://127.0.0.1:8000/mcp/jobs/` | Same, for the job helper |
| `JOB_MCP_TIMEOUT` | `120` | Longer: a resume rewrite over a full posting is its slowest call |
| `MCP_AUTH_TOKEN` | unset | Bearer token for both mounts; unset means no check |
| `MCP_USER_ID` | unset | stdio-only identity fallback |
| `JOB_QUERY_RATE_LIMIT` / `_WINDOW` | `30` / `60` | Per-user cap on job chat turns and fit assessments |
| `JOB_WRITE_RATE_LIMIT` / `_WINDOW` | `20` / `3600` | Per-user cap on resume uploads, posting analysis, and job searches |
| `TAVILY_API_KEY` | unset | Also used to read a posting from a URL; without it, paste-only |
| `JOB_SEARCH_PROVIDER` | `auto` | `auto`, `jsearch`, `adzuna`, `remotive`, `arbeitnow` — see §13 |
| `RAPIDAPI_KEY` | unset | Enables `jsearch`, the best-coverage provider |
| `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` | unset | Enables `adzuna` |
| `ADZUNA_COUNTRY` | `in` | Adzuna is per-country |
| `RESUME_BUCKET` | `resumes` | Supabase Storage bucket for original resume files. **Create it private.** |
| `RESUME_URL_TTL` | `3600` | Lifetime of a signed resume link, seconds |

`MCP_AUTH_TOKEN` unset is only acceptable when the port is unreachable from outside the
host; the app logs a warning at startup when it is.

---

## 11. Failure behaviour

| Failure | Result |
|---|---|
| MCP server unreachable | That skill reports it; the other skills and the reply still work |
| Job MCP server dies between propose and approve | The save is reported as failed, never claimed as done |
| Job posting URL blocked (LinkedIn, Indeed) | The user is asked to paste the text — not an error |
| Job board down or rate-limited | Empty listings plus a note saying so; the turn still answers |
| No job-search API keys configured | A keyless board is used; coverage is narrower and the note says so |
| Resume storage bucket missing | The resume still parses and is usable; the response says the original was not kept |
| Scanned image-only resume uploaded | 422 naming the problem, rather than an invented resume |
| Router returns no skill | Supervisor answers directly from the conversation |
| Server restarts mid-approval | `/approve` returns 404 asking the user to re-submit |
| Approval belongs to another user | 403 before the graph is touched |

---

## 12. The job helper's resumes

A resume exists here in **two representations, and neither replaces the other**:

| | What it is | What it's for |
|---|---|---|
| the original file | The `.docx`/`.pdf` exactly as uploaded, stored untouched | What the user sends to employers — their layout, their typography |
| the structure | `ResumeContent` ([`state.py`](app/agents/job_helper/state.py)) | What the agent reasons over: fit scoring, tailoring, rendering |

```
upload .docx/.pdf ─┬─▶ files.store ─────▶ Supabase Storage (private bucket)
                   └─▶ extract_text ─▶ parse (LLM) ─┐
                                                    ├─▶ Mongo `resumes` + users.resume
typed in by hand ──────────────────────────────────┘
                                                          │
                          tailor (LLM, against one posting)│
                                                          ▼
                             approval gate ─▶ new version + in-place `edits`
```

The structure can be filled **either way** — parsed from an upload, or typed in via
`save_resume_details` / `POST /resumes`. Manual entry is not a lesser path: it is how
someone with no resume file yet gets a profile, and how a mis-parsed phone number gets
corrected. `update_resume_details` merges section by section, so fixing one line never
means re-sending five years of bullets.

**The original is never modified.** Rewriting a `.docx` in place means matching new text
onto existing runs, which resumes split at arbitrary points (bold fragments, spell-check
boundaries) — any edit either loses formatting or corrupts the paragraph. So tailoring
returns `edits`: before/after pairs, where `before` is the exact existing line, which the
user applies inside their own document and keeps its design. Rendering the structure to a
fresh `.docx`/`.pdf` stays available, and is the right answer for the two cases with no
original to preserve — a typed-in resume and a tailored version.

`GET /resumes/{id}/download` therefore takes `fmt=original|docx|pdf`, defaulting to
`original`. The renderers are pure-Python (`python-docx`, `reportlab`) rather than
headless LibreOffice, which would add ~500 MB of system dependency to a runtime image
built to carry none.

**Storage is private and links are short-lived.** A resume is a home address, a phone
number and an employment history in one file, so the bucket is private, reads go through
signed URLs minted per request, and the download route serves bytes through our own auth
rather than handing out a storage link. Storage failure is non-fatal: the resume still
parses and is still usable, and the response says the original was not kept.

`users.resume` mirrors the active base resume — id, label, filename, storage path — so
anything holding a user can say "they have a resume, here it is" without a second query.
The versions themselves stay in their own collection, because an array that grows on a
document read by every authenticated request is the wrong shape.

The tailoring prompt may reorder, re-word and drop. It may never add a skill, employer,
date or metric the original does not contain — missing skills come back as a learning
roadmap instead, and every edit is visible before the user approves it.

---

## 13. Finding jobs

`search_jobs` browses; `track_job_posting` commits. Search saves nothing, so the model
may drive it freely; tracking analyses one posting and creates the record everything else
hangs off.

Four providers sit behind one normalised `JobListing`
([`jobsearch.py`](app/agents/job_helper/jobsearch.py)), because boards differ in exactly
the two ways that matter — whether they need a key, and whether they return the *full*
description:

| Provider | Key | Coverage | Description |
|---|---|---|---|
| `jsearch` | `RAPIDAPI_KEY` | Global, aggregates Google for Jobs | Full |
| `adzuna` | `ADZUNA_APP_ID` + `_KEY` | Global incl. India | Truncated |
| `remotive` | none | Remote tech only | Full (HTML) |
| `arbeitnow` | none | Mostly EU | Full (HTML) |

`auto` (the default) takes the best provider whose keys are actually present and falls
back to a keyless one — so a fresh clone searches with no configuration and improves when
a key is added, the same degradation the Tavily web search and the OCR path already use.
Each result carries a `note` about the board's coverage ("Remotive lists remote tech roles
only"), and the agent is instructed to pass that on rather than present a partial view as
the whole market. A board that is down returns an empty list and a note, never an
exception: the honest answer to "find me jobs" when the board is unreachable is "I could
not reach it — paste one and I'll analyse it".

HTML descriptions are converted with block tags becoming newlines and inline tags closing
up, so a requirements list keeps its structure and `we use <b>Python</b> and Postgres`
stays one sentence.
