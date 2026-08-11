"""Live end-to-end smoke test for the learning tracker.

The other two harnesses here grade *model output* — `harness.py` scores structured
output against a teacher, `rag_eval.py` scores RAG answers. This one grades
nothing. It walks one topic through the entire learning loop against real
infrastructure and asserts that each step does what the design says it does:

    chat turn → onboarding → roadmap approval → digests (with recall checks and,
    from #4, a written answer) → coverage → checkpoint → fail → revision gate →
    retry → pass → spaced review + Feynman → what the screens read

Everything the app does goes through the real HTTP routes, so the gates, the
background tasks and the response shapes are the ones a client meets. The unit
tests fake Mongo and the models; this one fakes nothing, which is the point —
both bugs it caught on its first two runs were invisible to a fake:

  * the Feynman judge scored a correct explanation `2` against a pass mark of 70,
    because "the proportion of outcomes conveyed" was read as a count;
  * coverage completed on the FIRST digest of a narrow topic, so the recall check
    (which rides the second) never fired, and neither did the written question or
    the re-teach path.

**It costs money and it writes to a database.** Roughly 35-45 model calls plus a
web search per digest. It runs against a *scratch* database on the same cluster —
derived from MONGO_URI, refused if it collides with the real one, dropped at the
end — so your own collections are never opened.

    # what it would do, no connection, no spend
    .venv/bin/python -m app.evals.learning_loop --dry-run

    # the real thing
    .venv/bin/python -m app.evals.learning_loop

    # keep the scratch data to poke at afterwards
    .venv/bin/python -m app.evals.learning_loop --keep --db lt_scratch

Exits non-zero if any check fails, so it can gate a release by hand. It is
deliberately NOT in CI: it needs live keys, it spends real money, and an LLM in
the loop makes it too flaky to block a merge on.
"""

import argparse
import os
import re
import sys
from contextlib import asynccontextmanager

DEFAULT_SCRATCH_DB = "lt_loop_scratch"

# The learner's own words, used for the Feynman checkpoint. Deliberately correct
# and plainly phrased: the judge is meant to pass this, and the run that scored it
# 2/100 is what exposed the scale bug in the prompt.
EXPLANATION = (
    "Ownership means every value has exactly one owner, and when the owner goes "
    "out of scope the value is dropped. Assigning it to another variable moves it "
    "rather than copying, so the old name can't be used afterwards. Borrowing "
    "lends a reference instead, either many readers or one writer."
)
WRITTEN_ANSWER = (
    "Because the value is moved rather than copied, so the original binding can "
    "no longer be used."
)
GOAL = "Build me a roadmap for Rust ownership and borrowing, I already know Python"


class Report:
    """Every assertion the walk makes, and whether it held."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def check(self, label: str, condition: bool, detail: str = "") -> bool:
        (self.passed if condition else self.failed).append(label)
        mark = "\033[32m✓\033[0m" if condition else "\033[31m✗\033[0m"
        print(f"  {mark} {label}{f' — {detail}' if detail else ''}")
        return bool(condition)

    def head(self, text: str) -> None:
        print(f"\n\033[1m{text}\033[0m")

    def summary(self) -> int:
        total = len(self.passed) + len(self.failed)
        print(f"\n\033[1m{len(self.passed)}/{total} checks passed\033[0m")
        for f in self.failed:
            print(f"  \033[31m✗ {f}\033[0m")
        return 1 if self.failed else 0


def scratch_uri(uri: str, db_name: str) -> str:
    """The same cluster, a different database.

    `get_db()` takes the database from the URI's path segment, so repointing the
    whole app is a matter of rewriting that one part.
    """
    m = re.match(r"^(mongodb(?:\+srv)?://[^/?]+)(/[^?]*)?(\?.*)?$", uri)
    if not m:
        raise SystemExit("MONGO_URI is not in a shape this can rewrite safely.")
    current = (m.group(2) or "/").lstrip("/")
    if current and current == db_name:
        # The one mistake that would matter, so it is refused rather than warned
        # about: this harness ends by dropping the database it ran against.
        raise SystemExit(
            f"Refusing to run: {db_name!r} is the database MONGO_URI already "
            "points at. Pass --db with a name that isn't your real one."
        )
    return f"{m.group(1)}/{db_name}{m.group(3) or ''}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default=DEFAULT_SCRATCH_DB, help="scratch database name")
    p.add_argument(
        "--digests",
        type=int,
        default=6,
        help="ceiling on digests generated; hitting it is not a failure, only a "
        "meatier topic than the walk needs",
    )
    p.add_argument("--keep", action="store_true", help="don't drop the scratch database")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and check the environment; no connection, no spend",
    )
    return p.parse_args(argv)


def plan() -> None:
    print(__doc__.split("**It costs money")[0].strip())
    print("\nEnvironment:")
    for var in ("MONGO_URI", "OPENAI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY"):
        print(f"  {'✓' if os.getenv(var) else '✗'} {var}")
    print("\nNothing was connected to and nothing was spent.")


def main(argv=None) -> int:
    args = parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    if args.dry_run:
        plan()
        return 0

    uri = os.getenv("MONGO_URI")
    if not uri:
        raise SystemExit("MONGO_URI is not set.")

    # Repointed before anything imports config, which reads the environment once.
    os.environ["MONGO_URI"] = scratch_uri(uri, args.db)
    return walk(args)


def walk(args) -> int:
    from bson import ObjectId
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pymongo import MongoClient

    from app.agents.learning_tracker.repository import retry_block as repo_retry_block
    from app.database import close_db, connect_db
    from app.dependencies import get_current_user
    from app.routers import learning_tracker as route

    uid = "learning-loop-smoke"
    r = Report()

    probe = MongoClient(os.environ["MONGO_URI"])
    scratch = probe[args.db]
    if scratch.list_collection_names():
        raise SystemExit(
            f"Refusing to run: {args.db!r} already has data in it. Drop it, or "
            "pass --db with an unused name."
        )

    def answers_for(quiz_id):
        """The answer key, read straight from Mongo.

        A harness privilege, and a demonstration in itself: the API never hands
        these out, so passing a check any other way is impossible.
        """
        quiz = scratch["quizzes"].find_one({"_id": ObjectId(quiz_id)})
        taps = [
            {"question": i, "answer": q.get("answer", 0)}
            for i, q in enumerate(quiz["questions"])
            if q.get("kind") != "open"
        ]
        written = {
            str(i): WRITTEN_ANSWER
            for i, q in enumerate(quiz["questions"])
            if q.get("kind") == "open"
        }
        return taps, written

    @asynccontextmanager
    async def lifespan(app):
        await connect_db()
        yield
        await close_db()

    def build_app():
        from langgraph.checkpoint.memory import MemorySaver

        from app.agents.learning_tracker.workflow import build_graph

        app = FastAPI(lifespan=lifespan)
        app.include_router(route.router)
        app.state.learning_agent = build_graph().compile(checkpointer=MemorySaver())
        app.dependency_overrides[get_current_user] = lambda: {"uid": uid}
        return app

    print(f"\033[2mscratch database: {args.db}\033[0m")

    try:
        with TestClient(build_app()) as client:
            # ── 1. the first run a visitor actually meets ──────────────────────
            r.head("1. First run: a chat turn that becomes a roadmap")
            turn = client.post("/learning/query", json={"text": GOAL}).json()
            r.check(
                "the first turn pauses for onboarding",
                turn.get("status") == "needs_input",
                str(turn.get("status")),
            )
            thread = turn.get("thread_id")

            resumed = client.post(
                "/learning/onboarding",
                json={
                    "thread_id": thread,
                    "answers": {
                        "skill_level": "intermediate",
                        "preferred_explanation_style": "examples_first",
                    },
                },
            ).json()
            r.check(
                "answering it runs straight on into the roadmap",
                resumed.get("status") == "needs_approval",
                str(resumed.get("status")),
            )
            proposal = (resumed.get("proposal") or {}).get("roadmap") or {}
            r.check(
                "a roadmap is proposed for approval",
                bool(proposal.get("topics")),
                f"{len(proposal.get('topics') or [])} topics",
            )
            r.check(
                "the proposal carries no ids or progress",
                all(
                    "id" not in t and "progress_status" not in t
                    for t in proposal.get("topics") or []
                ),
            )

            approved = client.post(
                "/learning/approvals",
                json={"thread_id": thread, "decision": "approved"},
            ).json()
            rid = (approved.get("result") or {}).get("roadmapId")
            if not r.check("approving saves it", bool(rid), str(approved.get("status"))):
                return r.summary()
            r.check(
                "its topics reach the assistant as to-dos",
                (approved.get("result") or {}).get("pa_tasks_created", 0) > 0,
            )

            doc = scratch["roadmaps"].find_one({"user_id": uid})
            started = [t for t in doc["topics"] if t["progress_status"] == "in_progress"]
            if not r.check(
                "exactly one topic underway",
                len(started) == 1,
                started[0]["title"] if started else "none",
            ):
                return r.summary()
            topic_id = started[0]["id"]
            r.check(
                "the profile answers were stored",
                (client.get("/learning/memory").json()["result"] or {}).get("skill_level")
                == "intermediate",
            )

            # ── 2. the drip-feed ──────────────────────────────────────────────
            r.head("2. The drip-feed")
            seen_check = seen_written = False
            for n in range(1, args.digests + 1):
                res = client.post("/learning/digests/generate", params={"roadmapId": rid})
                if res.status_code == 409:
                    print(f"  · declined at #{n}: {res.json()['detail']}")
                    break
                if not r.check(
                    f"digest #{n} generated",
                    res.status_code == 200,
                    res.text[:120] if res.status_code != 200 else "",
                ):
                    break

                digest = res.json()["result"]
                quiz = digest.get("quiz") or []
                print(
                    f"    seq={digest['sequence']} bullets={len(digest['bullets'])} "
                    f"check={len(quiz)}q coverage={digest['coverage_complete']}"
                )

                if quiz:
                    seen_check = True
                    r.check(
                        f"  #{n}'s check ships no answer key",
                        all(set(q) == {"question", "options", "kind"} for q in quiz),
                    )
                    if any(q["kind"] == "open" for q in quiz):
                        seen_written = True
                        r.check("  the written question is typed as open", True)

                taps, written = (
                    ([], {}) if not digest.get("quizId") else answers_for(digest["quizId"])
                )
                marked = client.post(
                    f"/learning/digests/{digest['_id']}/mark",
                    json={"answers": taps, "written": written, "generate_next": False},
                )
                if not r.check(
                    f"digest #{n} marked",
                    marked.status_code == 200,
                    str(marked.json())[:160] if marked.status_code != 200 else "",
                ):
                    break
                if digest["coverage_complete"]:
                    r.check("coverage complete — the checkpoint comes next", True)
                    break

            # The floor under the drip-feed exists to make both of these certain.
            r.check("a recall check appeared", seen_check)
            r.check("a written question appeared", seen_written)

            # ── 3. failing the checkpoint ─────────────────────────────────────
            r.head("3. Failing the checkpoint")
            issued = client.post(
                f"/learning/topics/{topic_id}/checkpoint", json={"roadmapId": rid}
            )
            if not r.check("checkpoint issued", issued.status_code == 200, issued.text[:140]):
                return r.summary()

            cp = issued.json()["result"]
            r.check(
                "questions carry no answers",
                all(set(q) == {"question", "options"} for q in cp["questions"]),
            )
            key, _ = answers_for(cp["quizId"])
            wrong = [{"question": a["question"], "answer": 1 - a["answer"]} for a in key]
            graded = client.post(
                "/learning/checkpoint/submit", json={"quizId": cp["quizId"], "answers": wrong}
            )
            out = graded.json().get("result", {})
            r.check("a failure is graded and returned", graded.status_code == 200)
            r.check("answers withheld on a failure", out.get("answers_revealed") is False)
            r.check(
                "no correctOption leaked",
                all("correctOption" not in x for x in out.get("review", [])),
            )
            r.check("hints returned instead", any(x.get("hint") for x in out.get("review", [])))
            r.check("revision debt opened", out.get("needs_revision") is True)
            r.check("weak points named", bool(out.get("weak_points")))

            # ── 4. the revision gate ──────────────────────────────────────────
            r.head("4. The revision gate")
            again = client.post(
                "/learning/checkpoint/submit", json={"quizId": cp["quizId"], "answers": key}
            )
            r.check(
                "the same set cannot be graded twice",
                again.status_code == 409,
                str(again.json().get("detail", {}).get("blocked_reason")),
            )
            retry = client.post(
                f"/learning/topics/{topic_id}/checkpoint", json={"roadmapId": rid}
            )
            r.check(
                "a retry is refused while revision is owed",
                retry.status_code == 409
                and retry.json()["detail"]["blocked_reason"] == "needs_revision",
                f"got {retry.status_code}",
            )

            rev = client.post("/learning/digests/generate", params={"roadmapId": rid})
            if r.check("revision digest generated", rev.status_code == 200, rev.text[:140]):
                rd = rev.json()["result"]
                r.check(
                    "it is written against the misses",
                    rd.get("kind") == "revision" and bool(rd.get("weak_points")),
                )
                cleared = client.post(f"/learning/digests/{rd['_id']}/mark", json={})
                r.check(
                    "marking it clears the debt",
                    cleared.json().get("result", {}).get("revision_cleared") is True,
                )

            # ── 5. passing ────────────────────────────────────────────────────
            r.head("5. Passing")
            # The cooldown outlives the revision: clearing the debt gets the
            # learner past the first gate and straight into the second, which is
            # the whole point of having two. Asserted rather than dodged — and
            # then stood down, because it is wall-clock and waiting it out would
            # add a minute to every run for a rule `retry_block` unit-tests.
            cooling = client.post(
                f"/learning/topics/{topic_id}/checkpoint", json={"roadmapId": rid}
            )
            r.check(
                "the cooldown still holds after the debt is cleared",
                cooling.status_code == 429
                and cooling.json()["detail"]["blocked_reason"] == "cooldown",
                f"got {cooling.status_code}",
            )
            print("\033[2m    (standing the cooldown down for the rest of the walk)\033[0m")
            route.retry_block = lambda *a, **kw: None

            second = client.post(
                f"/learning/topics/{topic_id}/checkpoint", json={"roadmapId": rid}
            )
            if r.check(
                "a new checkpoint is issued after revising",
                second.status_code == 200,
                second.text[:140],
            ):
                cp2 = second.json()["result"]
                r.check("a fresh set, not the one just failed", cp2["quizId"] != cp["quizId"])
                key2, _ = answers_for(cp2["quizId"])
                passed = client.post(
                    "/learning/checkpoint/submit",
                    json={"quizId": cp2["quizId"], "answers": key2},
                )
                res2 = passed.json().get("result", {})
                r.check(
                    "passing completes the topic",
                    res2.get("progress_status") == "completed",
                    f"score {res2.get('score')}",
                )
                r.check("answers revealed on a pass", res2.get("answers_revealed") is not False)
                r.check(
                    "next review scheduled",
                    bool(res2.get("next_review_at")),
                    str(res2.get("next_review_at"))[:10],
                )
                r.check(
                    "the slot moved to the next topic",
                    bool(res2.get("advanced_to")),
                    (res2.get("advanced_to") or {}).get("title", ""),
                )
            route.retry_block = repo_retry_block

            # ── 6. explaining it in your own words ────────────────────────────
            r.head("6. Explaining it in your own words")
            ex = client.post(
                f"/learning/topics/{topic_id}/explain",
                json={"roadmapId": rid, "text": EXPLANATION},
            )
            if r.check("explanation judged", ex.status_code == 200, ex.text[:140]):
                verdict = ex.json()["result"]
                print(
                    f"    score={verdict['score']} outcomes={len(verdict['outcomes'])} "
                    f"misconceptions={len(verdict['misconceptions'])}"
                )
                # The scale check. A correct explanation scoring single digits is
                # the judge counting outcomes instead of scoring a percentage,
                # which fails every learner who is actually right.
                r.check(
                    "a correct explanation passes",
                    verdict["passed"] is True,
                    f"score {verdict['score']} vs pass {verdict['pass_score']}",
                )
                r.check("probe withheld from the learner",
                        all("probe" not in m for m in verdict["misconceptions"]))

            # ── 7. what the screens read ──────────────────────────────────────
            r.head("7. What the screens read")
            focus = client.get("/learning/focus").json()["result"]
            r.check(
                "focus reports the roadmap",
                len(focus["roadmaps"]) == 1,
                str(focus["roadmaps"][0]["blocked_reason"]) if focus["roadmaps"] else "none",
            )
            stats = client.get("/learning/stats").json()["result"]
            r.check(
                "stats count the completed topic",
                stats["topics"]["completed"] >= 1,
                f"mastery {stats['mastery']['score']}",
            )
            mis = client.get("/learning/misconceptions").json()["result"]
            r.check(
                "misconception report readable",
                isinstance(mis, list),
                f"{sum(len(m['patterns']) for m in mis)} pattern(s)",
            )
            digests = client.get("/learning/digests", params={"limit": 50}).json()["result"]
            r.check(
                "every listed check keeps its kind",
                all(
                    set(q) == {"question", "options", "kind"}
                    for d in digests
                    for q in d["quiz"]
                ),
                f"{len(digests)} digests",
            )
    finally:
        if args.keep:
            print(f"\n\033[2mkept scratch database {args.db!r}\033[0m")
        else:
            names = scratch.list_collection_names()
            probe.drop_database(args.db)
            print(f"\n\033[2mdropped scratch database ({len(names)} collections)\033[0m")
        probe.close()

    return r.summary()


if __name__ == "__main__":
    sys.exit(main())
