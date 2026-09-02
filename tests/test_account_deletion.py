"""Tests for closing an account and erasing what it owns.

Offline: Mongo, Supabase, Pinecone and Redis are all replaced with fakes. The
thing actually under test is the *inventory* — that every store holding user
data is visited, in an order that does not strand anything — plus the rule that
a failing store must not be able to keep an account open.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.routers.users as users_router
import app.services.account_deletion as deletion
from app.dependencies import get_current_user

USER_ID = "651f1f77bcf86cd799439011"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cursor(docs=()):
    """An async cursor double for `find()` used with `async for`."""

    class _Cursor:
        def __aiter__(self):
            async def gen():
                for doc in docs:
                    yield dict(doc)

            return gen()

    return _Cursor()


def _mongo(resume_docs=(), deleted=1):
    """A db double where every collection deletes `deleted` rows."""
    col = MagicMock()
    col.delete_many = AsyncMock(return_value=MagicMock(deleted_count=deleted))
    col.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    col.find = MagicMock(return_value=_cursor(resume_docs))
    return {name: col for name in (*deletion.MONGO_COLLECTIONS, "users")}, col


def _make_client(user=None):
    app = FastAPI()
    app.include_router(users_router.router, prefix="/user")
    app.dependency_overrides[get_current_user] = lambda: user or {
        "_id": USER_ID,
        "uid": USER_ID,
        "is_guest": False,
    }
    return TestClient(app)


@pytest.fixture(autouse=True)
def _quiet_dependencies(monkeypatch):
    """Everything the purge reaches out to, stubbed by default."""
    monkeypatch.setattr(deletion, "_purge_supabase", AsyncMock())
    monkeypatch.setattr(deletion, "_purge_cache", AsyncMock())
    monkeypatch.setattr(users_router.rate_limit, "limit_key", AsyncMock())


# --------------------------------------------------------------------------- #
# the inventory
# --------------------------------------------------------------------------- #
async def test_purge_visits_every_known_collection(monkeypatch):
    db, col = _mongo()
    monkeypatch.setattr(deletion, "get_db", lambda: db)

    report = await deletion.purge_user_data(USER_ID)

    # One sweep per collection, each scoped to this user and nobody else.
    assert col.delete_many.await_count == len(deletion.MONGO_COLLECTIONS)
    for call in col.delete_many.await_args_list:
        assert call.args[0] == {"user_id": USER_ID}
    assert set(report.deleted) >= set(deletion.MONGO_COLLECTIONS)


def test_inventory_covers_the_collections_the_app_writes():
    """The list is the thing that goes stale. A collection added elsewhere and
    not added here is how a deletion quietly starts leaving data behind."""
    from app.agents.job_helper import repository as job_repo
    from app.agents.personal_assistant import repository as pa_repo

    assert pa_repo.TODOS in deletion.MONGO_COLLECTIONS
    assert pa_repo.MEMORIES in deletion.MONGO_COLLECTIONS
    assert job_repo.APPLICATIONS in deletion.MONGO_COLLECTIONS
    assert job_repo.RESUMES in deletion.MONGO_COLLECTIONS
    # The shared stores every agent pauses or schedules through.
    assert "approvals" in deletion.MONGO_COLLECTIONS
    assert "triggers" in deletion.MONGO_COLLECTIONS


async def test_resume_files_go_before_the_rows_that_name_them(monkeypatch):
    # The storage path only exists on the Mongo document. Sweeping the rows
    # first would leave the objects unreachable in the bucket forever.
    order: list[str] = []
    db, col = _mongo(resume_docs=[{"file": {"path": "u1/abc.pdf"}}])
    monkeypatch.setattr(deletion, "get_db", lambda: db)

    async def _files(user_id, report):
        order.append("files")

    async def _mongo_purge(user_id, report):
        order.append("mongo")

    monkeypatch.setattr(deletion, "_purge_resume_files", _files)
    monkeypatch.setattr(deletion, "_purge_mongo", _mongo_purge)

    await deletion.purge_user_data(USER_ID)

    assert order == ["files", "mongo"]


async def test_resume_files_are_removed_from_storage(monkeypatch):
    db, col = _mongo(resume_docs=[{"file": {"path": "a.pdf"}}, {"file": {"path": "b.pdf"}}])
    monkeypatch.setattr(deletion, "get_db", lambda: db)
    removed: list[str] = []
    monkeypatch.setattr(
        deletion, "_purge_resume_files_sync", lambda paths: (removed.extend(paths), len(paths))[1]
    )

    report = await deletion.purge_user_data(USER_ID)

    assert removed == ["a.pdf", "b.pdf"]
    assert report.deleted["resume_files"] == 2


async def test_documents_without_a_stored_file_are_skipped(monkeypatch):
    # A resume pasted as text has no object behind it.
    db, _ = _mongo(resume_docs=[{"file": None}, {}])
    monkeypatch.setattr(deletion, "get_db", lambda: db)
    monkeypatch.setattr(
        deletion,
        "_purge_resume_files_sync",
        MagicMock(side_effect=AssertionError("should not reach storage")),
    )

    report = await deletion.purge_user_data(USER_ID)

    assert "resume_files" not in report.deleted


# --------------------------------------------------------------------------- #
# partial failure
# --------------------------------------------------------------------------- #
async def test_a_failing_store_does_not_keep_the_account_open(monkeypatch):
    # A Pinecone or Supabase outage must not be the reason someone cannot close
    # their account.
    db, col = _mongo()
    monkeypatch.setattr(deletion, "get_db", lambda: db)
    monkeypatch.setattr(
        deletion, "_purge_supabase", AsyncMock(side_effect=None)
    )

    async def _boom(user_id, report):
        report.fail("supabase", RuntimeError("down"))

    monkeypatch.setattr(deletion, "_purge_supabase", _boom)

    report = await deletion.delete_account(USER_ID)

    assert "supabase" in report.failures
    # The account itself still went.
    col.delete_one.assert_awaited_once()
    assert report.deleted["users"] == 1


async def test_the_account_record_is_deleted_last(monkeypatch):
    order: list[str] = []
    col = MagicMock()
    col.delete_many = AsyncMock(
        side_effect=lambda *_: (order.append("data"), MagicMock(deleted_count=0))[1]
    )
    col.delete_one = AsyncMock(
        side_effect=lambda *_: (order.append("user"), MagicMock(deleted_count=1))[1]
    )
    col.find = MagicMock(return_value=_cursor())
    monkeypatch.setattr(
        deletion, "get_db", lambda: {n: col for n in (*deletion.MONGO_COLLECTIONS, "users")}
    )

    await deletion.delete_account(USER_ID)

    # Everything else, then the account — so a failure part-way leaves an
    # account that can retry rather than orphaned data.
    assert order[-1] == "user"
    assert "data" in order


async def test_a_failure_deleting_the_user_row_is_fatal(monkeypatch):
    col = MagicMock()
    col.delete_many = AsyncMock(return_value=MagicMock(deleted_count=0))
    col.delete_one = AsyncMock(side_effect=RuntimeError("mongo down"))
    col.find = MagicMock(return_value=_cursor())
    monkeypatch.setattr(
        deletion, "get_db", lambda: {n: col for n in (*deletion.MONGO_COLLECTIONS, "users")}
    )

    # Unlike the data stores, this one is the point of the request.
    with pytest.raises(RuntimeError):
        await deletion.delete_account(USER_ID)


# --------------------------------------------------------------------------- #
# the endpoint
# --------------------------------------------------------------------------- #
def test_delete_me_requires_the_password(monkeypatch):
    monkeypatch.setattr(users_router.account_deletion, "delete_account", AsyncMock())
    client = _make_client()

    resp = client.request("DELETE", "/user/me", json={})

    assert resp.status_code == 400
    users_router.account_deletion.delete_account.assert_not_awaited()


def test_delete_me_rejects_a_wrong_password(monkeypatch):
    # A stolen access token must not be enough on its own.
    monkeypatch.setattr(
        users_router.user_service, "verify_user_password", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(users_router.account_deletion, "delete_account", AsyncMock())
    client = _make_client()

    resp = client.request("DELETE", "/user/me", json={"password": "wrong"})

    assert resp.status_code == 401
    users_router.account_deletion.delete_account.assert_not_awaited()


def test_delete_me_with_the_right_password(monkeypatch):
    monkeypatch.setattr(
        users_router.user_service, "verify_user_password", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        users_router.account_deletion,
        "delete_account",
        AsyncMock(return_value=deletion.PurgeReport(deleted={"todos": 3, "users": 1})),
    )
    client = _make_client()

    resp = client.request("DELETE", "/user/me", json={"password": "correct horse"})

    assert resp.status_code == 200
    assert resp.json()["deleted"]["todos"] == 3
    users_router.account_deletion.delete_account.assert_awaited_once_with(USER_ID)


def test_a_guest_needs_no_password(monkeypatch):
    # Guests have none to give; refusing them would leave anonymous accounts
    # permanently undeletable.
    monkeypatch.setattr(
        users_router.user_service,
        "verify_user_password",
        AsyncMock(side_effect=AssertionError("should not check a guest's password")),
    )
    monkeypatch.setattr(
        users_router.account_deletion,
        "delete_account",
        AsyncMock(return_value=deletion.PurgeReport()),
    )
    client = _make_client({"_id": USER_ID, "uid": USER_ID, "is_guest": True})

    resp = client.request("DELETE", "/user/me", json={})

    assert resp.status_code == 200
    users_router.account_deletion.delete_account.assert_awaited_once()


def test_delete_me_reports_an_incomplete_purge(monkeypatch):
    monkeypatch.setattr(
        users_router.user_service, "verify_user_password", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        users_router.account_deletion,
        "delete_account",
        AsyncMock(
            return_value=deletion.PurgeReport(deleted={"users": 1}, failures=["supabase"])
        ),
    )
    client = _make_client()

    resp = client.request("DELETE", "/user/me", json={"password": "x"})

    # Surfaced rather than swallowed: the account is gone, but something still
    # holds rows and that needs to be visible.
    assert resp.json()["incomplete"] == ["supabase"]
