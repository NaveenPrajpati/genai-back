"""Tests for the job helper: documents, the graph's approval gate, the MCP
surface, and the HTTP routes.

Offline. The LLM, the MCP client and Mongo are all replaced with fakes — what is
worth pinning down is the machinery around the model, not its prose: that a
rendered resume is really a .docx/.pdf and survives a round trip, that nothing
is written before the user approves it, that identity cannot be supplied by the
calling model, and that one user cannot reach another's application.
"""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage
from mcp.server.fastmcp.exceptions import ToolError

import app.agents.job_helper.jobsearch as jobsearch
import app.agents.job_helper.workflow as wf
import app.mcp.job_helper.server as server
import app.routers.job_helper as routes
from app.agents.job_helper.documents import (
    UnsupportedResumeFile,
    extract_text,
    render,
    render_docx,
    suggest_filename,
)
from app.agents.job_helper.state import (
    ResumeContact,
    ResumeContent,
    ResumeExperience,
)
from app.dependencies import get_current_user
from app.mcp.job_helper.client import AUTONOMOUS_TOOLS, SAVE_RESUME_TOOL

RESUME = ResumeContent(
    contact=ResumeContact(name="Ada Lovelace", email="ada@example.com"),
    headline="Backend Engineer",
    summary="Shipped R&D tooling with <10ms p99 reads.",
    skills=["Python", "C++", "PostgreSQL"],
    experience=[
        ResumeExperience(
            company="Acme & Co",
            title="Engineer",
            start="2021",
            end="Present",
            bullets=["Cut latency 340ms -> 45ms."],
        )
    ],
)


def _client(agent=None, user=None):
    app = FastAPI()
    app.include_router(routes.router)
    app.state.job_agent = agent or MagicMock()
    app.dependency_overrides[get_current_user] = lambda: user or {"uid": "u1"}
    return TestClient(app)


def _ctx(headers=None):
    """A Context as the HTTP transport supplies it (stdio passes request=None)."""
    request = SimpleNamespace(headers=headers) if headers is not None else None
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


def _tool_message(name, payload_text):
    return ToolMessage(
        name=name,
        tool_call_id=f"call_{name}",
        content=[{"type": "text", "text": payload_text}],
    )


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fmt, magic, expected_type",
    [
        ("pdf", b"%PDF-", "application/pdf"),
        # A .docx is a zip container, and its media type is the long OOXML one —
        # a browser handed anything else offers to save "download" with no
        # extension.
        (
            "docx",
            b"PK",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
    ],
)
def test_a_rendered_resume_is_really_that_file_type(fmt, magic, expected_type):
    payload, media_type, filename = render(RESUME, fmt, "Backend Engineer")
    assert payload.startswith(magic)
    assert filename.endswith(f".{fmt}")
    assert media_type == expected_type


def test_rendering_survives_characters_that_look_like_markup():
    """reportlab parses Paragraph text as markup, so an unescaped `&` or `<` in
    a real bullet ("R&D", "<10ms", "C++") would raise mid-render."""
    payload, _, _ = render(RESUME, "pdf", "x")
    assert len(payload) > 1000


def test_an_empty_resume_still_renders():
    # A user can reach a download before anything has been parsed into a resume.
    for fmt in ("pdf", "docx"):
        payload, _, _ = render(ResumeContent(), fmt, "Empty")
        assert len(payload) > 500


def test_a_rendered_docx_can_be_read_back():
    """The round trip that matters: what we generate, we must be able to parse
    again — a user downloads a tailored resume, edits it, and re-uploads it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(render_docx(RESUME, "Resume"))
        path = tmp.name
    try:
        text = extract_text(path, "resume.docx")
    finally:
        os.unlink(path)

    for needle in ("Ada Lovelace", "Acme & Co", "340ms -> 45ms", "C++"):
        assert needle in text


def test_an_unsupported_file_type_is_named_not_guessed():
    with pytest.raises(UnsupportedResumeFile, match="docx"):
        extract_text("/tmp/whatever", "resume.pages")


def test_a_readable_file_with_no_text_is_reported():
    """A scanned-image PDF arrives as a perfectly valid file containing zero
    characters; handing that empty string to the parser would produce an
    invented resume."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"   \n  \n")
        path = tmp.name
    try:
        with pytest.raises(ValueError, match="No text could be read"):
            extract_text(path, "resume.txt")
    finally:
        os.unlink(path)


def test_the_download_filename_is_built_from_the_resume_not_the_id():
    assert suggest_filename(RESUME, "Backend Engineer", "pdf").startswith(
        "Ada_Lovelace"
    )
    # No name on the resume: fall back to the label rather than an empty stem.
    assert suggest_filename(ResumeContent(), "Tailored — Acme", "pdf") == (
        "Tailored_Acme.pdf"
    )


# --------------------------------------------------------------------------- #
# the graph: nothing is saved before the user says so
# --------------------------------------------------------------------------- #
def test_a_proposal_routes_to_the_approval_gate():
    assert wf.route({"tailored": {"content": {}}}) == "approve_resume"


def test_a_turn_with_no_proposal_skips_the_gate():
    # Reading applications or drafting a letter writes nothing to approve.
    assert wf.route({"summary": "You have 3 open applications."}) == "finalize"


def test_the_proposal_is_read_from_the_propose_tool_only():
    messages = [
        _tool_message("list_applications", "[]"),
        _tool_message("propose_tailored_resume", '{"content": {"skills": ["a"]}}'),
        _tool_message("draft_cover_letter", '{"body": "..."}'),
    ]
    assert wf._last_proposal(messages) == {"content": {"skills": ["a"]}}


def test_a_proposal_without_content_is_not_a_proposal():
    messages = [_tool_message("propose_tailored_resume", '{"error": "no resume"}')]
    assert wf._last_proposal(messages) is None


async def test_rejecting_a_tailored_resume_saves_nothing(monkeypatch):
    save_tool = AsyncMock()
    monkeypatch.setattr(wf, "get_pending", AsyncMock(return_value=None))
    monkeypatch.setattr(wf, "create_pending", AsyncMock(return_value="a1"))
    monkeypatch.setattr(wf, "resolve", AsyncMock())
    monkeypatch.setattr(wf, "load_job_tools", AsyncMock(return_value=[save_tool]))
    monkeypatch.setattr(wf, "interrupt", lambda payload: "rejected")

    out = await wf.approve_resume(
        {"user_id": "u1", "thread_id": "t1", "tailored": {"content": {}}}
    )

    assert out["resume_status"] == "rejected by user"
    save_tool.ainvoke.assert_not_called()
    wf.resolve.assert_awaited_with("a1", "rejected")


async def test_approving_saves_and_returns_the_new_version(monkeypatch):
    save_tool = MagicMock(name=SAVE_RESUME_TOOL)
    save_tool.ainvoke = AsyncMock(
        return_value=[{"type": "text", "text": '{"resume_id": "r9", "saved": true}'}]
    )
    monkeypatch.setattr(wf, "get_pending", AsyncMock(return_value=None))
    monkeypatch.setattr(wf, "create_pending", AsyncMock(return_value="a1"))
    monkeypatch.setattr(wf, "resolve", AsyncMock())
    monkeypatch.setattr(wf, "load_job_tools", AsyncMock(return_value=[save_tool]))
    monkeypatch.setattr(wf, "find_tool", lambda tools, name: save_tool)
    monkeypatch.setattr(wf, "interrupt", lambda payload: "approved")

    out = await wf.approve_resume(
        {
            "user_id": "u1",
            "thread_id": "t1",
            "tailored": {
                "content": {"skills": ["Python"]},
                "application_id": "app1",
                "base_resume_id": "r1",
                "changes": ["Summary: rewritten"],
            },
        }
    )

    assert out == {"resume_status": "approved and saved", "resume_version_id": "r9"}
    assert save_tool.ainvoke.await_args.args[0]["application_id"] == "app1"
    wf.resolve.assert_awaited_with("a1", "approved")


async def test_a_save_tool_that_vanished_is_reported_not_claimed(monkeypatch):
    """The MCP server can go away between proposing and approving. Saying the
    resume was saved when it was not is the one answer we must never give."""
    monkeypatch.setattr(wf, "get_pending", AsyncMock(return_value=None))
    monkeypatch.setattr(wf, "create_pending", AsyncMock(return_value="a1"))
    monkeypatch.setattr(wf, "resolve", AsyncMock())
    monkeypatch.setattr(wf, "load_job_tools", AsyncMock(return_value=[]))
    monkeypatch.setattr(wf, "interrupt", lambda payload: "approved")

    out = await wf.approve_resume(
        {"user_id": "u1", "thread_id": "t1", "tailored": {"content": {}}}
    )
    assert "unavailable" in out["resume_status"]


def _pending(query):
    return {
        "payload": {
            "proposal": {"content": {"skills": ["Python"]}},
            "transcript": "Here is the rewrite.",
            "query": query,
        }
    }


async def test_the_agent_node_reuses_a_stored_proposal_on_replay(monkeypatch):
    """LangGraph re-runs the node from the top on resume. Without this the user
    pays for a second tool loop and can be handed a different rewrite than the
    one they just approved."""
    monkeypatch.setattr(wf, "get_pending", AsyncMock(return_value=_pending("tailor it")))
    load = AsyncMock()
    monkeypatch.setattr(wf, "load_job_tools", load)

    out = await wf.agent({"user_id": "u1", "thread_id": "t1", "query": "tailor it"})

    assert out["tailored"] == {"content": {"skills": ["Python"]}}
    load.assert_not_called()


async def test_an_abandoned_approval_does_not_hijack_the_next_question(monkeypatch):
    """A pending approval outlives the turn that raised it. Matching on the
    thread alone would answer every later question on that thread with the old
    proposal instead of running the loop."""
    monkeypatch.setattr(wf, "get_pending", AsyncMock(return_value=_pending("tailor it")))
    monkeypatch.setattr(wf, "load_job_tools", AsyncMock(return_value=[]))

    out = await wf.agent(
        {"user_id": "u1", "thread_id": "t1", "query": "what have I applied to?"}
    )
    assert out.get("tailored") is None
    wf.load_job_tools.assert_awaited_once()


async def test_an_unavailable_mcp_server_degrades_this_skill_only(monkeypatch):
    monkeypatch.setattr(wf, "get_pending", AsyncMock(return_value=None))
    monkeypatch.setattr(wf, "load_job_tools", AsyncMock(return_value=[]))

    out = await wf.agent({"user_id": "u1", "thread_id": "t1", "query": "hi"})
    assert out["resume_status"] == "unavailable"


async def test_finalize_hands_over_download_links_only_after_a_save():
    saved = await wf.finalize(
        {"summary": "Tailored it.", "resume_status": "approved and saved",
         "resume_version_id": "r9"}
    )
    assert "/api/job-helper/resumes/r9/download?fmt=pdf" in saved["summary"]

    rejected = await wf.finalize(
        {"summary": "Tailored it.", "resume_status": "rejected by user"}
    )
    assert "download" not in rejected["summary"]
    assert "unchanged" in rejected["summary"]


# --------------------------------------------------------------------------- #
# MCP surface
# --------------------------------------------------------------------------- #
def test_identity_comes_from_the_request_header():
    assert server._caller_id(_ctx({"x-user-id": "u1"})) == "u1"


def test_the_header_wins_over_the_env_var(monkeypatch):
    monkeypatch.setenv("MCP_USER_ID", "stdio-user")
    assert server._caller_id(_ctx({"x-user-id": "u1"})) == "u1"


def test_an_unidentified_call_is_refused_not_guessed(monkeypatch):
    monkeypatch.delenv("MCP_USER_ID", raising=False)
    with pytest.raises(ToolError, match="No user identity"):
        server._caller_id(_ctx({}))


def test_no_tool_takes_a_user_id_argument():
    """Identity travels out-of-band on purpose: a user_id the model fills in is
    a user_id the model can get wrong or be talked into changing."""
    for tool in server.mcp._tool_manager.list_tools():
        assert "user_id" not in (tool.parameters.get("properties") or {}), tool.name


def test_saving_a_resume_is_not_a_tool_the_model_may_drive_itself():
    """The whole point of the propose/save split: a resume goes out under the
    user's name, so the loop may generate one but never persist one."""
    assert SAVE_RESUME_TOOL not in AUTONOMOUS_TOOLS
    published = {t.name for t in server.mcp._tool_manager.list_tools()}
    assert SAVE_RESUME_TOOL in published
    # Every other published tool is drivable, or the loop cannot do its job.
    assert published - AUTONOMOUS_TOOLS == {SAVE_RESUME_TOOL}


async def test_a_foreign_application_id_is_rejected(monkeypatch):
    monkeypatch.setattr(server.repo, "get_application", AsyncMock(return_value=None))
    with pytest.raises(ToolError, match="does not belong to you"):
        await server._require_application("u1", "someone-elses-app")


@pytest.mark.parametrize(
    "tool, kwargs",
    [
        (server.get_application, {}),
        (server.add_application_note, {"note": "called back"}),
        (server.update_application_status, {"status": "applied"}),
        (server.add_followup_tasks, {"titles": ["Email them"]}),
        (server.start_roadmap_for_gap, {}),
    ],
)
async def test_application_scoped_tools_check_ownership_first(
    monkeypatch, tool, kwargs
):
    monkeypatch.setattr(server.repo, "get_application", AsyncMock(return_value=None))
    with pytest.raises(ToolError, match="does not belong to you"):
        await tool(_ctx({"x-user-id": "u1"}), application_id="a1", **kwargs)


async def test_a_roadmap_handoff_needs_a_gap_to_have_been_measured(monkeypatch):
    monkeypatch.setattr(server.repo, "get_application", AsyncMock(return_value={"id": "a1"}))
    monkeypatch.setattr(server.service, "suggest_learning", AsyncMock(return_value=None))
    with pytest.raises(ToolError, match="assess_resume_fit first"):
        await server.start_roadmap_for_gap(
            _ctx({"x-user-id": "u1"}), application_id="a1"
        )


async def test_the_full_posting_is_not_echoed_back_through_the_tool(monkeypatch):
    """The structured analysis says the same thing; sending both would spend the
    tool budget on duplicated prose."""
    monkeypatch.setattr(
        server.repo,
        "get_application",
        AsyncMock(return_value={"id": "a1", "jd_text": "x" * 5000, "analysis": {}}),
    )
    out = await server.get_application(_ctx({"x-user-id": "u1"}), application_id="a1")
    assert "jd_text" not in out


# --------------------------------------------------------------------------- #
# HTTP routes
# --------------------------------------------------------------------------- #
async def test_a_foreign_application_id_is_refused_before_any_model_spend(monkeypatch):
    monkeypatch.setattr(
        routes.repo, "verify_application_ownership", AsyncMock(return_value=False)
    )
    agent = MagicMock()
    agent.ainvoke = AsyncMock()
    with _client(agent) as client:
        resp = client.post(
            "/job-helper/query", json={"text": "tailor it", "application_id": "a1"}
        )
    assert resp.status_code == 403
    agent.ainvoke.assert_not_called()


def test_a_pending_resume_comes_back_as_needs_approval(monkeypatch):
    monkeypatch.setattr(routes.rate_limit, "limit_user", AsyncMock())
    agent = MagicMock()
    agent.ainvoke = AsyncMock(
        return_value={
            "__interrupt__": [SimpleNamespace(value={"changes": ["Summary: rewritten"]})]
        }
    )
    with _client(agent) as client:
        resp = client.post("/job-helper/query", json={"text": "tailor my resume"})

    body = resp.json()
    assert body["status"] == "needs_approval"
    assert body["proposal"]["changes"] == ["Summary: rewritten"]
    assert body["thread_id"]  # the client needs it to call /approve


def test_approving_someone_elses_resume_is_refused(monkeypatch):
    monkeypatch.setattr(
        routes,
        "get_pending",
        AsyncMock(return_value={"user_id": "someone-else", "_id": "a1"}),
    )
    agent = MagicMock()
    agent.ainvoke = AsyncMock()
    with _client(agent) as client:
        resp = client.post(
            "/job-helper/approve", json={"thread_id": "t1", "decision": "approved"}
        )
    assert resp.status_code == 403
    agent.ainvoke.assert_not_called()


def test_downloading_a_resume_returns_a_file_with_a_filename(monkeypatch):
    monkeypatch.setattr(
        routes.repo,
        "get_resume",
        AsyncMock(
            return_value={
                "id": "r1",
                "label": "Backend Engineer",
                "content": RESUME.model_dump(),
            }
        ),
    )
    with _client() as client:
        resp = client.get("/job-helper/resumes/r1/download?fmt=pdf")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment" in resp.headers["content-disposition"]
    assert "Ada_Lovelace" in resp.headers["content-disposition"]
    assert resp.content.startswith(b"%PDF-")


def test_downloading_another_users_resume_is_a_404(monkeypatch):
    # The repository scopes every read by user_id, so a guessed id resolves to
    # nothing rather than to someone else's document.
    monkeypatch.setattr(routes.repo, "get_resume", AsyncMock(return_value=None))
    with _client() as client:
        resp = client.get("/job-helper/resumes/someone-elses/download?fmt=pdf")
    assert resp.status_code == 404


def test_an_unreadable_upload_is_a_4xx_not_a_500(monkeypatch):
    monkeypatch.setattr(routes.rate_limit, "limit_user", AsyncMock())
    store = AsyncMock()
    monkeypatch.setattr(routes.service, "store_uploaded_resume", store)
    with _client() as client:
        resp = client.post(
            "/job-helper/resumes/upload",
            files={"file": ("resume.pages", b"not a resume", "application/octet-stream")},
        )
    assert resp.status_code == 400
    store.assert_not_called()


def test_uploading_a_resume_keeps_the_original_and_parses_it(monkeypatch):
    """Both representations: the file the user sends to employers, and the
    structure the agent reasons over."""
    monkeypatch.setattr(routes.rate_limit, "limit_user", AsyncMock())
    stored = {"path": "u1/abc.txt", "filename": "resume.txt", "size": 28}
    monkeypatch.setattr(routes.files, "store", MagicMock(return_value=stored))
    monkeypatch.setattr(routes.files, "with_link", lambda r: r)
    monkeypatch.setattr(
        routes.service,
        "store_uploaded_resume",
        AsyncMock(return_value={"id": "r1", "kind": "base", "file": stored}),
    )
    with _client() as client:
        resp = client.post(
            "/job-helper/resumes/upload",
            files={"file": ("resume.txt", b"Ada Lovelace\nEngineer at Acme", "text/plain")},
        )

    assert resp.status_code == 201
    body = resp.json()
    assert body["result"]["id"] == "r1"
    assert body["original_stored"] is True
    # The text went to the parser, and the bytes went to storage.
    assert "Ada Lovelace" in routes.service.store_uploaded_resume.await_args.args[1]
    assert routes.files.store.call_args.args[1] == b"Ada Lovelace\nEngineer at Acme"
    assert (
        routes.service.store_uploaded_resume.await_args.kwargs["file_meta"] == stored
    )


def test_a_resume_survives_storage_being_down(monkeypatch):
    """Losing the convenience of keeping the original must not lose the resume."""
    monkeypatch.setattr(routes.rate_limit, "limit_user", AsyncMock())
    monkeypatch.setattr(routes.files, "store", MagicMock(return_value=None))
    monkeypatch.setattr(routes.files, "with_link", lambda r: r)
    monkeypatch.setattr(
        routes.service,
        "store_uploaded_resume",
        AsyncMock(return_value={"id": "r1", "kind": "base", "file": None}),
    )
    with _client() as client:
        resp = client.post(
            "/job-helper/resumes/upload",
            files={"file": ("resume.txt", b"Ada Lovelace", "text/plain")},
        )

    body = resp.json()
    assert resp.status_code == 201
    assert body["original_stored"] is False
    assert "could not be stored" in body["warning"]


def test_a_resume_can_be_typed_in_instead_of_uploaded(monkeypatch):
    """No file required: someone with no resume yet still gets a profile the
    agent can score and tailor."""
    monkeypatch.setattr(
        routes.service,
        "save_manual_resume",
        AsyncMock(return_value={"id": "r2", "source": "manual"}),
    )
    with _client() as client:
        resp = client.post(
            "/job-helper/resumes",
            json={
                "content": {
                    "contact": {"name": "Ada Lovelace"},
                    "skills": ["Python"],
                },
                "label": "First draft",
            },
        )

    assert resp.status_code == 201
    assert resp.json()["result"]["source"] == "manual"
    saved = routes.service.save_manual_resume.await_args.args[1]
    assert saved["contact"]["name"] == "Ada Lovelace"


def test_downloading_the_original_returns_the_users_own_file(monkeypatch):
    """The whole point of keeping it: byte-for-byte what they uploaded, not a
    regenerated approximation of it."""
    monkeypatch.setattr(
        routes.repo,
        "get_resume",
        AsyncMock(
            return_value={
                "id": "r1",
                "label": "Base",
                "content": RESUME.model_dump(),
                "file": {
                    "path": "u1/abc.docx",
                    "filename": "Ada_resume.docx",
                    "content_type": "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document",
                },
            }
        ),
    )
    monkeypatch.setattr(routes.files, "fetch", MagicMock(return_value=b"PK\x03\x04ORIG"))
    with _client() as client:
        resp = client.get("/job-helper/resumes/r1/download?fmt=original")

    assert resp.status_code == 200
    assert resp.content == b"PK\x03\x04ORIG"
    assert "Ada_resume.docx" in resp.headers["content-disposition"]


def test_asking_for_an_original_that_does_not_exist_says_so(monkeypatch):
    # A typed-in resume, or a tailored version: no file was ever uploaded.
    monkeypatch.setattr(
        routes.repo,
        "get_resume",
        AsyncMock(return_value={"id": "r2", "content": {}, "file": None}),
    )
    with _client() as client:
        resp = client.get("/job-helper/resumes/r2/download?fmt=original")
    assert resp.status_code == 404
    assert "pdf or docx" in resp.json()["detail"]


def test_deleting_a_resume_also_deletes_its_stored_file(monkeypatch):
    monkeypatch.setattr(
        routes.repo,
        "delete_resume",
        AsyncMock(return_value={"id": "r1", "file": {"path": "u1/abc.docx"}}),
    )
    remove = MagicMock()
    monkeypatch.setattr(routes.files, "remove", remove)
    with _client() as client:
        resp = client.delete("/job-helper/resumes/r1")

    assert resp.status_code == 200
    remove.assert_called_once_with("u1/abc.docx")


def test_patching_a_resume_sends_only_the_sections_that_changed(monkeypatch):
    """Sections left at their default must not reach the merge, or an omitted
    list reads as 'delete everything here'."""
    monkeypatch.setattr(routes.repo, "update_resume", AsyncMock())
    monkeypatch.setattr(
        routes.service, "update_resume_content", AsyncMock(return_value={"id": "r1"})
    )
    monkeypatch.setattr(routes.files, "with_link", lambda r: r)
    with _client() as client:
        resp = client.patch(
            "/job-helper/resumes/r1",
            json={"content": {"skills": ["Python", "Go"]}},
        )

    assert resp.status_code == 200
    changes = routes.service.update_resume_content.await_args.args[2]
    assert changes == {"skills": ["Python", "Go"]}


def test_tracking_a_posting_without_text_or_a_link_is_rejected(monkeypatch):
    monkeypatch.setattr(routes.rate_limit, "limit_user", AsyncMock())
    monkeypatch.setattr(
        routes.service,
        "track_application",
        AsyncMock(side_effect=ValueError("Provide the job description text or a link.")),
    )
    with _client() as client:
        resp = client.post("/job-helper/applications", json={})
    assert resp.status_code == 400
    assert "job description" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# job search
#
# Every provider is mocked at the HTTP boundary: these pin the normalisation and
# the degradation, which are ours, not the boards' uptime.
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def _fake_httpx(payload):
    """Stand in for `httpx.AsyncClient` used as an async context manager."""

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, *a, **kw):
            return _FakeResponse(payload)

    return _FakeClient


async def test_remotive_results_are_normalised(monkeypatch):
    monkeypatch.setattr(
        jobsearch.httpx,
        "AsyncClient",
        _fake_httpx(
            {
                "jobs": [
                    {
                        "title": "Backend Engineer",
                        "company_name": "Acme",
                        "candidate_required_location": "Worldwide",
                        "url": "https://remotive.com/j/1",
                        "description": "<p>We use <b>Python</b> &amp; Postgres.</p>",
                        "salary": "$100k",
                        "publication_date": "2026-08-01",
                    }
                ]
            }
        ),
    )
    out = await jobsearch.search_jobs("python", provider="remotive")

    listing = out["listings"][0]
    assert out["provider"] == "remotive"
    assert listing["title"] == "Backend Engineer"
    assert listing["company"] == "Acme"
    assert listing["remote"] is True
    # HTML in, text out — the model should never be handed markup to read.
    assert listing["description"] == "We use Python & Postgres."
    assert "remote tech roles only" in out["note"]


async def test_arbeitnow_is_filtered_here_because_it_has_no_search(monkeypatch):
    monkeypatch.setattr(
        jobsearch.httpx,
        "AsyncClient",
        _fake_httpx(
            {
                "data": [
                    {
                        "title": "Python Engineer",
                        "company_name": "Acme",
                        "tags": ["python"],
                        "url": "https://arbeitnow.com/1",
                        "description": "text",
                        "remote": True,
                        "location": "Berlin",
                    },
                    {
                        "title": "Sales Lead",
                        "company_name": "Other",
                        "tags": ["sales"],
                        "url": "https://arbeitnow.com/2",
                        "description": "text",
                        "remote": False,
                        "location": "Berlin",
                    },
                ]
            }
        ),
    )
    out = await jobsearch.search_jobs("python", provider="arbeitnow")
    assert [listing["title"] for listing in out["listings"]] == ["Python Engineer"]


async def test_a_board_that_is_down_returns_a_note_not_an_exception(monkeypatch):
    """"Find me jobs" when the board is unreachable should answer honestly, not
    fail the user's turn."""

    class _Boom:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            raise RuntimeError("connection refused")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(jobsearch.httpx, "AsyncClient", _Boom)
    out = await jobsearch.search_jobs("python", provider="remotive")

    assert out["listings"] == []
    assert "could not be reached" in out["note"]


def test_a_keyless_provider_is_always_available(monkeypatch):
    """A fresh clone with no API keys can still search — the same graceful
    default the Tavily-backed web search uses."""
    for key in ("RAPIDAPI_KEY", "ADZUNA_APP_ID", "ADZUNA_APP_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("JOB_SEARCH_PROVIDER", raising=False)

    assert jobsearch.configured_providers()[0] == "remotive"
    assert jobsearch._resolve(None) == "remotive"


def test_a_configured_key_upgrades_the_default_provider(monkeypatch):
    monkeypatch.setenv("RAPIDAPI_KEY", "x")
    monkeypatch.delenv("JOB_SEARCH_PROVIDER", raising=False)
    assert jobsearch._resolve(None) == "jsearch"


def test_an_explicit_provider_wins_over_the_configured_default(monkeypatch):
    monkeypatch.setenv("RAPIDAPI_KEY", "x")
    assert jobsearch._resolve("arbeitnow") == "arbeitnow"


def test_the_search_route_reports_which_board_answered(monkeypatch):
    monkeypatch.setattr(routes.rate_limit, "limit_user", AsyncMock())
    monkeypatch.setattr(
        routes.service,
        "find_jobs",
        AsyncMock(return_value={"provider": "remotive", "listings": [], "note": "n"}),
    )
    with _client() as client:
        resp = client.get("/job-helper/search?q=python&remote=true")

    assert resp.status_code == 200
    assert resp.json()["provider"] == "remotive"
    assert routes.service.find_jobs.await_args.args[0] == "python"
