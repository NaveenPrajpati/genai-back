"""URL ingestion (step1.load_url + the /rag/ingest/url route + empty-source handling
in the worker). Uses a throwaway local HTTP server — no internet, no keys.

The SSRF guard blocks loopback, which is where the test server lives, so content
tests loosen `_ip_allowed`; the guard tests run it unpatched.
"""

import http.server
import json
import socketserver
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.dependencies import get_current_user
from app.routers import rag
from app.services import rate_limit
from app.services.rag import step1_ingestion as s1
from app.services.rag import step3_indexing_worker as s3

_REAL_READ_VIA_JINA = s1._read_via_jina  # captured before the autouse stub


def _page(ctype, body):
    return ctype, body if isinstance(body, bytes) else body.encode("utf-8")


PAGES = {
    "/nocharset": _page(
        "text/html", "<html><body><p>Café crème — 日本語</p></body></html>"
    ),
    "/metacharset": _page(
        "text/html",
        '<html><head><meta charset="utf-8"></head><body><p>naïve résumé</p></body></html>',
    ),
    "/latin1": (
        "text/html; charset=ISO-8859-1",
        "<html><body><p>Café</p></body></html>".encode("latin-1"),
    ),
    "/plain": _page("text/plain", "plain text — ok"),
    "/spa": _page(
        "text/html; charset=utf-8",
        '<html><body><div id="root"></div><script>render()</script></body></html>',
    ),
    "/doc.pdf": ("application/pdf", b"%PDF-1.4 binary"),
    "/layout": _page(
        "text/html; charset=utf-8",
        "<html><body><header>SITE NAV</header><nav>MENU</nav>"
        "<article><header><h1>Article Title</h1></header><p>Body text.</p></article>"
        "<footer>SITE FOOTER</footer></body></html>",
    ),
    "/big": _page("text/html", "<p>" + "x" * 5000 + "</p>"),
    "/long": _page("text/html", "<article><p>" + "word " * 200 + "</p></article>"),
    "/short": _page("text/html", "<html><body><div id='root'></div><p>Famliya</p></body></html>"),
}


class _Handler(http.server.BaseHTTPRequestHandler):
    redirect_to = None  # set per-server for the redirect test

    def do_GET(self):
        if self.path == "/redirect" and self.redirect_to:
            self.send_response(302)
            self.send_header("Location", self.redirect_to)
            self.end_headers()
            return
        ctype, body = PAGES.get(self.path, ("text/plain", b"not found"))
        self.send_response(200 if self.path in PAGES else 404)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _start_server(handler_cls=_Handler):
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler_cls)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture(scope="module")
def server():
    srv = _start_server()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def allow_loopback(monkeypatch):
    monkeypatch.setattr(s1, "_ip_allowed", lambda ip: True)


@pytest.fixture(autouse=True)
def renderer(monkeypatch):
    """Never call the real Jina Reader. Records calls; returns `renderer.result`
    (default "" = rendering found nothing) or raises it if it's an exception.
    Also drops the minimum-text floor so tiny fixture pages pass — the floor has
    its own test."""

    class Fake:
        calls: list = []
        result = ""

        def __call__(self, url):
            self.calls.append(url)
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    fake = Fake()
    fake.calls = []
    monkeypatch.setattr(s1, "_read_via_jina", fake)
    monkeypatch.setattr(s1, "URL_JS_FALLBACK", "jina")
    monkeypatch.setattr(s1, "URL_MIN_TEXT_CHARS", 1)
    return fake


# ── validate_url / _ip_allowed ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "ftp://example.com/f.txt",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://",
        "https://user:pass@example.com/",
        None,
        123,
    ],
)
def test_validate_url_rejects(url):
    with pytest.raises(ValueError):
        s1.validate_url(url)


def test_validate_url_accepts_and_strips():
    assert s1.validate_url("  https://example.com/a  ") == "https://example.com/a"


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # cloud instance metadata
        "100.64.0.1",  # CGNAT
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fe80::1%3",
        "fd00::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
    ],
)
def test_ip_blocked(ip):
    assert s1._ip_allowed(ip) is False


@pytest.mark.parametrize("ip", ["8.8.8.8", "93.184.216.34", "2606:4700:4700::1111"])
def test_ip_allowed(ip):
    assert s1._ip_allowed(ip) is True


# ── SSRF guard (unpatched) ──────────────────────────────────────────────────


def test_loopback_url_is_blocked(server):
    with pytest.raises(s1.UnsafeURLError):
        s1.load_url(server + "/plain")


def test_redirect_to_internal_is_blocked(monkeypatch, server):
    """A permitted first hop that 302s to a forbidden address must be refused:
    the guard runs on every connection, not just the initial URL."""
    internal = _start_server()
    internal_port = internal.server_address[1]

    class Redirecting(_Handler):
        redirect_to = f"http://127.0.0.1:{internal_port}/plain"

    front = _start_server(Redirecting)
    try:
        # "Public" = anything except the internal server's port.
        def check(sock):
            if sock.getpeername()[1] == internal_port:
                sock.close()
                raise s1.UnsafeURLError("internal")

        monkeypatch.setattr(s1, "_check_peer", check)
        with pytest.raises(s1.UnsafeURLError):
            s1.load_url(f"http://127.0.0.1:{front.server_address[1]}/redirect")
    finally:
        front.shutdown()
        internal.shutdown()


# ── content handling ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/nocharset", "Café crème — 日本語"),  # UTF-8, no charset anywhere
        ("/metacharset", "naïve résumé"),  # charset only in <meta>
        ("/latin1", "Café"),  # header charset honoured
        ("/plain", "plain text — ok"),
    ],
)
def test_decoding(allow_loopback, server, path, expected):
    docs = s1.load_url(server + path)
    assert docs[0].page_content == expected
    assert docs[0].metadata == {"source": server + path}


def test_layout_noise_stripped_but_article_header_kept(allow_loopback, server):
    text = s1.load_url(server + "/layout")[0].page_content
    assert "Article Title" in text and "Body text." in text
    for noise in ("SITE NAV", "MENU", "SITE FOOTER"):
        assert noise not in text


def test_non_html_content_type_rejected(allow_loopback, server):
    with pytest.raises(ValueError, match="Unsupported content type"):
        s1.load_url(server + "/doc.pdf")


def test_empty_page_rejected(allow_loopback, server):
    with pytest.raises(ValueError, match="Too little readable text"):
        s1.load_url(server + "/spa")


# ── JS-rendered pages: Jina Reader fallback ─────────────────────────────────


def test_js_page_uses_rendered_text(allow_loopback, server, renderer):
    renderer.result = "# Community Pods\n\n" + "Real article text. " * 20
    text = s1.load_url(server + "/short")[0].page_content
    assert text == renderer.result
    assert renderer.calls == [server + "/short"]


def test_rendered_text_only_used_if_longer(allow_loopback, server, renderer):
    renderer.result = "x"
    assert s1.load_url(server + "/short")[0].page_content == "Famliya"


def test_no_fallback_when_static_text_is_enough(allow_loopback, server, renderer):
    s1.load_url(server + "/long")
    assert renderer.calls == []


def test_no_fallback_for_plain_text(allow_loopback, server, renderer):
    s1.load_url(server + "/plain")
    assert renderer.calls == []


def test_fallback_can_be_disabled(allow_loopback, monkeypatch, server, renderer):
    monkeypatch.setattr(s1, "URL_JS_FALLBACK", "off")
    s1.load_url(server + "/short")
    assert renderer.calls == []


def test_thin_page_fails_when_rendering_fails(
    allow_loopback, monkeypatch, server, renderer
):
    """The original bug: a JS page yielded 7 chars, 'completed', and every
    question was refused. It must now fail loudly instead."""
    monkeypatch.setattr(s1, "URL_MIN_TEXT_CHARS", 50)
    renderer.result = RuntimeError("jina down")
    with pytest.raises(ValueError, match=r"Too little readable text .*7 characters"):
        s1.load_url(server + "/short")


def test_blocked_host_never_reaches_renderer(server, renderer):
    """The SSRF guard runs before the fallback, so an internal URL is never
    forwarded to the third-party renderer either."""
    with pytest.raises(s1.UnsafeURLError):
        s1.load_url(server + "/short")
    assert renderer.calls == []


def test_read_via_jina_cleans_markdown(monkeypatch):
    payload = {
        "data": {
            "title": "A Guide",  # not in the body → prepended as a heading
            "content": "Join [Community Pods](https://x.com/pods \"t\") today."
            "![logo](https://x.com/a.png)\n\n## Why",
        }
    }
    sent = {}

    class Resp:
        raw = type("R", (), {"read": lambda self, n, decode_content: json.dumps(payload).encode()})()

        def raise_for_status(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_post(url, json, headers, timeout, stream):
        sent.update(url=url, json=json, headers=headers)
        return Resp()

    monkeypatch.setenv("JINA_API_KEY", "k-test")
    monkeypatch.setattr(s1.requests, "post", fake_post)
    text = _REAL_READ_VIA_JINA("https://example.com/a")  # autouse stubbed the attr
    assert text == "# A Guide\n\nJoin Community Pods today.\n\n## Why"
    assert sent["json"] == {"url": "https://example.com/a"}
    assert sent["headers"]["Authorization"] == "Bearer k-test"


def test_size_cap(allow_loopback, monkeypatch, server):
    monkeypatch.setattr(s1, "URL_MAX_BYTES", 1000)
    with pytest.raises(ValueError, match="too large"):
        s1.load_url(server + "/big")


def test_http_error_raises(allow_loopback, server):
    with pytest.raises(Exception, match="404"):
        s1.load_url(server + "/missing")


# ── route: input validation ─────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch):
    async def _no_limit(*a, **k):
        return None

    monkeypatch.setattr(rate_limit, "limit_user", _no_limit)
    app = FastAPI()
    app.include_router(rag.router)
    app.dependency_overrides[get_current_user] = lambda: {"uid": "u-url"}
    return TestClient(app)


@pytest.mark.parametrize(
    "data",
    [
        "not json",
        json.dumps(["https://example.com"]),
        json.dumps({"link": "https://example.com"}),
        json.dumps({"url": "ftp://example.com/x"}),
        json.dumps({"url": ""}),
        json.dumps({"url": 42}),
    ],
)
def test_ingest_url_bad_input_is_400(client, data):
    r = client.post("/rag/ingest/url", data={"data": data})
    assert r.status_code == 400, r.text


def test_ingest_url_missing_data_is_400(client):
    assert client.post("/rag/ingest/url").status_code == 400


# ── worker: empty source fails instead of "completing" ──────────────────────


async def test_worker_marks_empty_source_failed(monkeypatch):
    updates = []
    monkeypatch.setattr(
        s3.storage, "update_ingestion_log", lambda job, u: updates.append(u)
    )

    def _no_index(*a, **k):
        raise AssertionError("must not index an empty source")

    monkeypatch.setattr(s3, "hybrid_add_texts", _no_index)

    s3.INGESTION_JOBS["job-empty"] = {"status": "queued"}
    await s3.run_ingestion(
        "job-empty",
        lambda: [Document(page_content="", metadata={})],
        "src",
        "url",
        None,
        "u1",
    )
    assert s3.INGESTION_JOBS["job-empty"]["status"] == "failed"
    assert updates[-1]["status"] == "failed"
    s3.INGESTION_JOBS.pop("job-empty", None)
