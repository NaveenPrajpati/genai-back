# Learning Assistant — Features and Flows

How the learning tracker turns "I want to learn Rust" into a roadmap, then keeps the
learner moving through it: daily digests that teach a topic a few points at a time, an
active-recall checkpoint that gates completion, and spaced repetition that brings
finished topics back before they fade.

- **Companion docs:** [AGENT_SYSTEM.md](AGENT_SYSTEM.md) (how this runs as a supervisor
  subgraph), [RAG_SYSTEM.md](RAG_SYSTEM.md) (the retrieval subsystem).
- **Code root:** [`app/agents/learning_tracker/`](app/agents/learning_tracker/),
  [`app/routers/learning_tracker.py`](app/routers/learning_tracker.py) (HTTP).
- **Client:** `aiapps/src/app/learning/` (screens), `aiapps/src/features/learning/`
  (store, API client, types).

```
app/agents/learning_tracker/
├── state.py        the vocabulary: statuses, drafts vs stored shapes, graph state
├── workflow.py     the LangGraph: intent classification → one specialist agent
├── service.py      LLM calls that aren't graph nodes (checkpoints, coverage, quizzes)
├── repository.py   Mongo persistence + the pure domain logic (progress, forecasts)
├── triggers.py     digest generation and the hourly sweep that schedules it
└── tools.py        the web-search tool the research + digest paths use
```

---

## 1. What it does

| Feature | What the learner sees | Where it lives |
|---|---|---|
| **Roadmap generation** | Describe a goal in chat, get a staged roadmap of topics to approve | `roadmap_agent` + HITL interrupt |
| **Edits keep progress** | Revising a roadmap doesn't reset what's already done | `merge_roadmap` |
| **One topic at a time** | Exactly one topic is "in progress"; starting another hands over the slot | `start_topic` |
| **Daily digests** | 3–5 teaching bullets on the current topic, on a schedule they pick | `build_digest` + `run_triggers` |
| **Reading is checked** | Every other digest carries a recall check over the ones since the last | `digest_carries_quiz` |
| **Can't run ahead** | The next check-bearing digest waits until the last check was passed | `digest_quiz_gate` |
| **Knows when to stop** | Once the tips have covered the topic, digests stop and the checkpoint takes over | `check_coverage` |
| **Completion is earned** | A topic completes by passing a checkpoint, not by ticking a box | `apply_checkpoint` |
| **Spaced repetition** | Finished topics resurface on an expanding ladder | `REVIEW_LADDER_DAYS` |
| **At most two roadmaps running** | Two active at once; the rest are parked until a slot frees | `MAX_ACTIVE_ROADMAPS` |
| **Notes per topic** | Jottings, snippets, links, and questions to revisit | `learning_notes` |
| **Personalized** | Skill level, goals, pace and format preferences steer generation | `learning` memory namespace |
| **Pace forecast** | "At your stated pace you finish around <date>" | `completion_forecast` |

---

## 2. The chat graph

`POST /learning/query` (and `/query/stream`) run a LangGraph over `LearningState`.
It is a **classify-then-dispatch** graph — one specialist runs per turn, and the
graph ends there. There is no loop back to the router.

```
START ──▶ load_memory ──┬─(first run)─▶ onboard ──┐
                        │                          │
                        └──────────────────────────┴─▶ classify_intent
                                                            │
                    ┌───────────────────────────────────────┤
                    ▼             ▼            ▼            ▼           ▼
             roadmap_agent   tutor_agent  quiz_grader  research   progress_agent
                    │                                                   │
                 (interrupt: approve the roadmap)              (reads/writes topics)
```

**Intent → agent** ([`workflow.py`](app/agents/learning_tracker/workflow.py) `INTENT_ROUTES`):

| Intent | Agent |
|---|---|
| `create_roadmap`, `modify_roadmap` | `roadmap_agent` |
| `explain`, `quiz` | `tutor_agent` |
| `submit_quiz` | `quiz_grader_agent` |
| `find_resources` | `research_agent` |
| `update_progress`, `query_roadmap` | `progress_agent` |
| `chitchat`, `fallback` | `fallback_agent` |

Two things interrupt a turn rather than finishing it:

- **Onboarding** — a first-time learner is asked two profile questions. Resumes via
  `POST /learning/onboarding` (answers, or `null` to skip; either way it's recorded so
  the prompt doesn't reappear).
- **Roadmap approval** — a generated or edited roadmap is a proposal, not a write.
  Resumes via `POST /learning/approvals`.

Both surface to the client as a paused turn (`needs_input` / `needs_approval`) through
one shared response shape. Answering onboarding often runs straight into a roadmap
approval, so a resume can pause again immediately — the client has to handle that.

**Response projection.** `_CLIENT_FIELDS` in the router is a whitelist. The graph state
also carries `current_user` and the learner's whole memory profile, none of which
belongs in a chat response.

---

## 3. Roadmap lifecycle

```
draft ──▶ active ⇄ paused ──▶ archived
             │
             └──▶ completed  (every topic done; reopening a topic rolls it back)
```

A topic moves `not_started → in_progress → needs_review → completed`, plus `skipped`.
`DONE_STATUSES = {completed, skipped}` is what "behind the learner" means.

**Exactly one topic is in progress.** `start_topic` demotes whichever topic held the
slot and promotes the new one in a single update with two array filters, so the roadmap
is never momentarily two-in-progress or none.

**The server owns identity and progress.** `materialize_roadmap` mints topic ids and
progress fields; the model never supplies them. On an edit, `merge_roadmap` keeps a
topic's id — and therefore its progress — when the model echoes `existing_id` or the
title matches. Only ids on that roadmap are honoured and each stored topic is claimed
once, so a model repeating or inventing an id can't copy one topic's progress onto
several.

**At most two roadmaps are active** (`MAX_ACTIVE_ROADMAPS`, default 2). `active` is what
the digest sweep runs on and what a bare "what should I study next?" resolves to, so the
cap is really a limit on how many drip-feeds run in parallel. Enforced in the repository,
not the route, because three separate paths can mint an active roadmap:

| Path | Behavior at the cap |
|---|---|
| `PATCH /roadmaps/:id` → `active` | Raises `ActiveRoadmapLimit` → **409**, naming which roadmaps hold the slots |
| A newly approved roadmap | Saved as `paused` — refusing would discard what the learner just built |
| Reopening a topic on a finished roadmap | The rollup parks it at `paused` instead of silently granting a third slot |

Parking (`paused` / `archived`) is never capped, or a learner already over the limit
would have no way back under.

---

## 4. Digests — teaching a topic a few points at a time

A digest is 3–5 bullets that **teach**, plus reference links. Generated by
`build_digest` ([`triggers.py`](app/agents/learning_tracker/triggers.py)), shared by the
daily sweep and the on-demand pull, so a digest fetched early is the same artefact as
one that arrived on schedule.

```
run_triggers (hourly)
  └─ for each user whose local schedule_hour matches now
       └─ for each ACTIVE roadmap
            └─ build_digest(current in_progress topic)
                 ├─ nothing in progress?           → skip
                 ├─ DIGEST_MAX_UNREAD already?     → skip   ┐ both before spending
                 ├─ last recall check unpassed?    → skip   ┘ a search + an LLM call
                 ├─ web search + LLM               → bullets, steered away from
                 │                                    what earlier digests covered
                 ├─ even-numbered digest           → recall check over the digests
                 │                                    since the last check
                 ├─ check_coverage                 → topic fully taught?
                 │      └─ yes: topic → needs_review, drip-feed stops
                 └─ store + push notification
```

Three guards keep the inbox honest:

- **`DIGEST_MAX_UNREAD`** (default 3) — a stack of unread nudges is just noise, and each
  one costs a web search and an LLM call. The backlog read fails *closed*: if it can't
  tell how many are waiting, it doesn't add another.
- **The recall gate** — see below.
- **Only `in_progress` topics** — drip-feeding tips about something nobody has opened is
  how an inbox fills with things nobody asked for.

### The recall cadence

A check rides **every other digest** — #2, #4, #6 — not every digest past the first.
Back-to-back checks turn a nudge into homework; the point is to catch a digest that was
swiped away, not to examine.

| Digest | Carries a check? | Covers | Blocked until |
|---|---|---|---|
| #1 | no | — | — |
| #2 | yes | #1 | — |
| #3 | no | — | — |
| #4 | yes | #2, #3 | #2's check passed |
| #5 | no | — | — |
| #6 | yes | #4, #5 | #4's check passed |

Each check covers only what's happened since the last one, and never the digest it's
attached to — the learner hasn't read that yet, so quizzing on it would make marking
impossible.

**Only check-bearing digests are gated.** #3 arrives while #2's check is outstanding;
#4 does not. That's what keeps the three-deep buffer usable — the learner can read one
ahead, just not indefinitely.

**Acknowledging.** `POST /digests/{id}/mark` is the only signal a digest landed. Marking
a check-bearing digest requires passing its check, so `status: "marked"` *is* the record
that the check was passed — there's no separate flag. A wrong set comes back **422**
with the grading attached, and the digest stays unread.

**Coverage complete** flips the topic to `needs_review`: no further digests are generated
for it (`in_progress_topic` no longer matches), and the checkpoint is what comes next.

**Scheduling** is one trigger per user, not per roadmap — `schedule_hour` in their
timezone, optionally narrowed to one weekday. `next_run_at` reads the same rules forward,
which is what the home screen counts down to.

---

## 5. Checkpoints and spaced repetition

Completion is gated on active recall. `build_checkpoint` grounds the questions in the
topic's own description and learning outcomes, so they can't wander into material the
learner hasn't reached — the failure mode that would make the gate feel arbitrary.
Answers never reach the client; grading is server-side.

`apply_checkpoint` folds the score in, and is **deliberately asymmetric**:

| Attempt | Score ≥ `CHECKPOINT_PASS_SCORE` | Below |
|---|---|---|
| First | → `completed`, review ladder starts | Stays `in_progress`; nothing is taken away |
| Review of a completed topic | Ladder advances one rung | Stays `completed`; ladder resets to the front |

A failed review never un-completes a topic. Clawing back progress for an honest attempt
would punish the exact behaviour the feature exists to encourage.

**The ladder** — `REVIEW_LADDER_DAYS = (1, 3, 7, 16, 35)`. The better you know something,
the less often it comes back. `GET /reviews` lists what's due, soonest first.

---

## 6. The home screen contract

`GET /learning/focus` answers "nothing's waiting — so what now?".

```json
{ "roadmaps": [ { "roadmapId": "…", "roadmapTitle": "Rust",
                  "topic": { "id": "t1", "title": "Ownership", … },
                  "progress": { "completed_count": 2, "total": 9, "percent": 22 },
                  "unread": 1, "can_generate": true, "blocked_reason": null } ],
  "unread": 1, "cap": 3, "next_at": "2026-08-05T09:00:00+00:00",
  "blocked_reason": null }
```

**One entry per active roadmap**, because the sweep digests each of them — reporting a
single "current" roadmap hid the queues the learner was really accumulating. `next_at`
and `cap` sit at the top: the digest schedule is one per-account setting.

Every entry carries a `blocked_reason` rather than going quiet, because "no digest is
coming" always has a cause worth showing:

| Reason | Level | Means |
|---|---|---|
| `no_roadmap` | account | Nothing active — no roadmaps, or all parked |
| `digests_off` | account (mirrored per roadmap) | Never opted in, or switched off. Manual pull still works |
| `cap_reached` | roadmap | `DIGEST_MAX_UNREAD` already waiting on this topic |
| `awaiting_quiz` | roadmap | An earlier digest's recall check hasn't been passed |
| `needs_review` | roadmap | Fully taught; the checkpoint is what's next |
| `roadmap_complete` | roadmap | Every topic done |

Anything that blocks one roadmap stays on that roadmap's entry — the others may still be
running.

---

## 7. HTTP surface

All routes are under `/learning` and require a bearer token. Every read and write is
scoped by `user_id` in the query, so an id belonging to someone else matches nothing
rather than reading across.

**Chat**
| Route | Purpose |
|---|---|
| `POST /query`, `POST /query/stream` | One turn. Stream is SSE; only `tutor_agent` text tokens are forwarded |
| `POST /approvals` | Approve or reject a roadmap proposal |
| `POST /onboarding` | Answer or skip the first-run profile questions |

**Roadmaps**
| Route | Purpose |
|---|---|
| `GET /roadmaps`, `GET /roadmaps/{id}` | List (paginated, status-filterable) and detail |
| `PATCH /roadmaps/{id}` | Park, resume, archive. **409** past the active cap |
| `GET /stats` | Aggregate across all roadmaps, incl. `active` / `paused` / `max_active` |
| `GET /focus` | See §6 |
| `GET /current-state` | The active roadmap and its progress, no id needed |
| `POST /progress` | Set a topic's progress directly |

**Digests**
| Route | Purpose |
|---|---|
| `GET /digests` | The archive. Filters: `status`, `active_only`, `roadmapId`, `topicId` |
| `POST /digests/{id}/mark` | Acknowledge; `generate_next` pulls the following one |
| `POST /digests/generate` | Pull now instead of waiting for the sweep |

**Checkpoints, notes, profile, schedule**
| Route | Purpose |
|---|---|
| `POST /topics/{topicId}/checkpoint`, `POST /checkpoint/submit` | Issue and grade |
| `GET /reviews` | Completed topics whose review is due |
| `POST /submit-quiz` | Grade a chat-issued quiz |
| `GET/POST/PATCH/DELETE /notes` | Notes, snippets, links, questions |
| `GET/PUT/DELETE /memory` | The learning profile (`/state` is a deprecated alias) |
| `GET /triggers`, `POST /toggle-trigger`, `PATCH /trigger-settings` | Digest schedule |

`GET /digests` filters in the Mongo query rather than client-side, so `limit` means "the
newest N of what you asked for" — narrowing an already-truncated page would show an empty
list for any topic not written about recently, indistinguishable from "no digests".

---

## 8. Storage

| Collection | Holds |
|---|---|
| `roadmaps` | The roadmap document, topics embedded, all progress |
| `learning_digests` | One doc per digest: bullets, resources, quiz id, coverage flag, status |
| `learning_notes` | Notes scoped to a roadmap + topic |
| `quizzes` | Checkpoint and digest recall questions, **with** the answer key |
| `quiz_attempts` | Grading history, feeding the quiz average in `/stats` |
| `triggers` | Per-user digest schedule (`action_type: "learning_digest"`) |
| `memories` | The learning profile, under the `learning` namespace |

Progress lives in `progress_status` only. Documents written before that field existed
(a bare `covered: bool`) are not supported and are not read.

---

## 9. Configuration

All in [`app/core/config.py`](app/core/config.py), all env-overridable.

| Setting | Default | Controls |
|---|---|---|
| `MAX_ACTIVE_ROADMAPS` | 2 | How many roadmaps run at once |
| `DIGEST_MAX_UNREAD` | 3 | Unread digests one topic may accumulate |
| `DIGEST_QUIZ_EVERY` | 2 | Cadence of the recall check (#2, #4, #6 …) |
| `DIGEST_QUIZ_QUESTIONS` | 2 | Length of the "did you read it" recall check |
| `DIGEST_QUIZ_PASS_SCORE` | 100 | All-or-nothing: it's a read check, not an exam |
| `CHECKPOINT_QUESTIONS` | 4 | Length of the completion checkpoint |
| `CHECKPOINT_PASS_SCORE` | 80 | The bar to complete a topic, and to advance a review |

---

## 10. Client screens

| Screen | Reads |
|---|---|
| `learning/index.tsx` | `/focus`, `/stats`, `/reviews`, `/digests?status=unread&active_only=true` — one focus card per active roadmap, then the catch-up queue |
| `learning/roadmaps.tsx` | `/roadmaps`, `/stats` — the management surface; pause / resume / archive, with resume disabled at the cap |
| `learning/[id].tsx` | `/roadmaps/{id}` — topics, checkpoints, notes |
| `learning/digests.tsx` | `/digests` — the archive, filtered by roadmap then topic |
| `learning/notes.tsx` | `/notes` — the consolidated view |
| `learning/settings.tsx` | `/memory`, `/triggers` — profile and digest schedule |

State lives in one Zustand store (`features/learning/store.ts`); all network access goes
through `learningApi.ts`, and tokens are handled by the shared `http` interceptor.
