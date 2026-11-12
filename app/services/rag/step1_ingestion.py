"""
services/rag/step1_ingestion.py
=====================
STEP 1 OF THE RAG PIPELINE: INGESTION (a.k.a. "loading")

Ingestion = turning a raw source (PDF, .txt, web page, ...) into a list of
LangChain `Document` objects with clean text + metadata. Nothing is embedded
or chunked yet — this stage only answers "what is the raw text and where did
it come from?".

──────────────────────────────────────────────────────────────────────────────
WHAT YOU HAVE DONE
──────────────────────────────────────────────────────────────────────────────
  • PDF      → PyPDFLoader      (one Document per page, sets metadata["page"])
  • .txt     → TextLoader       (one Document for the whole file)
  • URL      → custom BeautifulSoup loader (strips nav/footer/script noise)

This covers the three most common inputs and is a solid baseline.

──────────────────────────────────────────────────────────────────────────────
WHAT YOU COULD DO TO IMPROVE INGESTION
──────────────────────────────────────────────────────────────────────────────
1. BETTER PDF EXTRACTION
   PyPDFLoader is fast but loses layout (tables become word soup, multi-column
   PDFs interleave columns). Upgrade paths, in rough order of power/cost:
     - PyMuPDFLoader (`fitz`)  → faster, keeps more structure, free.
     - UnstructuredPDFLoader   → detects titles/tables/lists; "hi_res" mode uses
                                 a layout model. Great for reports & forms.
     - LlamaParse / Azure Document Intelligence / AWS Textract → cloud OCR that
                                 returns Markdown tables. Best for scanned docs,
                                 invoices, financial statements. (paid)
   USE CASE → if your corpus has tables/figures you must answer questions about,
   layout-aware extraction is the single biggest quality lever.

2. OCR FOR SCANNED / IMAGE PDFs
   PyPDFLoader returns empty text for scanned pages. Detect "no text extracted"
   and fall back to OCR (Tesseract via `unstructured`, or a cloud OCR service).

3. MORE FILE TYPES (route by MIME / extension)
     - .docx                → Docx2txtLoader / UnstructuredWordDocumentLoader
     - .pptx                → UnstructuredPowerPointLoader
     - .xlsx / .csv         → UnstructuredExcelLoader / CSVLoader (one row = one doc)
     - .md / .html          → UnstructuredMarkdownLoader / BSHTMLLoader
     - source code          → language-aware loaders + RecursiveCharacterTextSplitter
                              .from_language(...)
     - audio/video          → Whisper transcription → text
   USE CASE → a "chat over my Google Drive" product needs a dispatcher that maps
   each MIME type to the right loader; expose this as a registry (see LOADER_REGISTRY).

4. RICHER WEB LOADING
   `requests + BeautifulSoup` fails on JS-rendered pages and ignores sitemaps.
     - Playwright / Selenium loader → renders JS-heavy SPAs.
     - SitemapLoader / RecursiveUrlLoader → crawl a whole docs site, not one page.
     - Firecrawl / Jina Reader → hosted "URL → clean Markdown" services.

5. METADATA YOU'LL WANT LATER
   Capture author, created_at, section/heading, and a stable source URL at
   ingestion time. You can't filter or cite on metadata you never captured.
"""

import io
import os
import re
import json
import time
import logging
import ipaddress
from typing import List, Optional
from urllib.parse import urlsplit

import requests
import pymupdf
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup, UnicodeDammit
from pypdf import PdfReader
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    AsyncHtmlLoader,
)
from langchain_community.document_transformers import Html2TextTransformer


# ── URL fetching ────────────────────────────────────────────────────────────
# A user-supplied URL is fetched BY THE SERVER, so without guards it is an SSRF
# primitive: ingest http://169.254.169.254/... (EC2 instance metadata) or
# http://127.0.0.1:8000/metrics, then ask the RAG to read the result back.

URL_MAX_BYTES = 5 * 1024 * 1024  # stop reading a response past this size
URL_CONNECT_READ_TIMEOUT = (5, 15)  # per-socket-op (connect, read) seconds
URL_TOTAL_TIMEOUT = 30  # wall-clock cap, defeats slow-drip responses
URL_MAX_REDIRECTS = 5

# JavaScript-rendered pages (React/Vite SPAs) serve an empty <div id="root"> and
# build the article in the browser, so a plain fetch extracts next to nothing.
# When that happens we ask Jina Reader — a hosted headless browser, same vendor
# and key as the reranker — to render it. Headless Chromium on the 1 GB EC2 box
# isn't an option. Set URL_JS_FALLBACK=off to never send URLs to Jina.
URL_JS_FALLBACK = os.getenv("URL_JS_FALLBACK", "jina").strip().lower()
URL_FALLBACK_BELOW_CHARS = 500  # static text shorter than this → try rendering
URL_MIN_TEXT_CHARS = 50  # still shorter than this → fail, nothing to answer from
JINA_READER_URL = "https://r.jina.ai/"
JINA_READER_TIMEOUT = (5, 45)  # rendering a page is slow; (connect, read)

logger = logging.getLogger(__name__)

_HTML_TYPES = {"text/html", "application/xhtml+xml"}
_TEXT_TYPES = {"text/plain"}


class UnsafeURLError(ValueError):
    """The URL is malformed, uses a disallowed scheme, or points at a non-public
    address. A ValueError, so callers can surface the message as a 400."""


def validate_url(url: str) -> str:
    """Cheap syntactic check, done in the request so bad input fails as a 400
    instead of a background job. The address check happens at connect time."""
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("url must be a non-empty string")
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise UnsafeURLError("Only http:// and https:// URLs are supported")
    if not parts.hostname:
        raise UnsafeURLError("URL has no host")
    if parts.username or parts.password:
        raise UnsafeURLError("URLs with embedded credentials are not allowed")
    return url


def _ip_allowed(ip: str) -> bool:
    """True only for publicly routable addresses. Rejects loopback, private
    (RFC1918), link-local (incl. 169.254.169.254 metadata), CGNAT, multicast,
    reserved and unspecified ranges, and IPv4-mapped IPv6 forms of those."""
    addr = ipaddress.ip_address(ip.split("%", 1)[0])  # drop IPv6 zone id
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_global and not addr.is_multicast


def _check_peer(sock) -> None:
    """Verify the address the socket ACTUALLY connected to. Checking here rather
    than resolving the hostname up front closes the DNS-rebinding gap (the name
    can't resolve to a public IP for the check and a private one for the fetch),
    and covers every redirect hop for free."""
    peer = sock.getpeername()[0]
    if not _ip_allowed(peer):
        sock.close()
        raise UnsafeURLError("URL resolves to a private or internal address")


class _GuardedHTTPConnection(HTTPConnection):
    def _new_conn(self):
        sock = super()._new_conn()
        _check_peer(sock)
        return sock


class _GuardedHTTPSConnection(HTTPSConnection):
    def _new_conn(self):
        sock = super()._new_conn()
        _check_peer(sock)
        return sock


class _GuardedHTTPPool(HTTPConnectionPool):
    ConnectionCls = _GuardedHTTPConnection


class _GuardedHTTPSPool(HTTPSConnectionPool):
    ConnectionCls = _GuardedHTTPSConnection


class _GuardedAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _GuardedHTTPPool,
            "https": _GuardedHTTPSPool,
        }


def _guarded_session() -> requests.Session:
    s = requests.Session()
    # Ignore HTTP(S)_PROXY env vars: through a proxy the peer we check would be
    # the proxy, not the target, and the guard would be meaningless.
    s.trust_env = False
    s.max_redirects = URL_MAX_REDIRECTS
    adapter = _GuardedAdapter(max_retries=0)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _parse_content_type(header: Optional[str]) -> tuple[str, Optional[str]]:
    """'text/html; charset=UTF-8' -> ('text/html', 'UTF-8')."""
    if not header:
        return "", None
    mime, *params = [p.strip() for p in header.split(";")]
    charset = None
    for p in params:
        key, _, value = p.partition("=")
        if key.strip().lower() == "charset" and value:
            charset = value.strip().strip("\"'")
    return mime.lower(), charset


def _fetch(url: str) -> tuple[bytes, str, Optional[str]]:
    """GET `url` through the SSRF guard with size and time caps.
    Returns (body, mime, declared_charset)."""
    deadline = time.monotonic() + URL_TOTAL_TIMEOUT
    with _guarded_session() as session:
        with session.get(
            url,
            stream=True,
            timeout=URL_CONNECT_READ_TIMEOUT,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; RAG-ingest/1.0)",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            },
        ) as resp:
            resp.raise_for_status()
            mime, charset = _parse_content_type(resp.headers.get("Content-Type"))
            # An absent Content-Type is treated as HTML and sniffed below.
            if mime and mime not in _HTML_TYPES | _TEXT_TYPES:
                raise ValueError(
                    f"Unsupported content type at URL: {mime}. Only web pages "
                    "(HTML) and plain text can be ingested by URL — download the "
                    "file and upload it instead."
                )
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > URL_MAX_BYTES:
                raise ValueError(
                    f"Page is too large (> {URL_MAX_BYTES // (1024 * 1024)} MB)"
                )

            body = bytearray()
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                body.extend(chunk)
                if len(body) > URL_MAX_BYTES:
                    raise ValueError(
                        f"Page is too large (> {URL_MAX_BYTES // (1024 * 1024)} MB)"
                    )
                if time.monotonic() > deadline:
                    raise ValueError("Timed out downloading the page")
    return bytes(body), mime, charset


def _html_to_text(raw: bytes, charset: Optional[str]) -> str:
    # Feed BYTES, not resp.text: requests falls back to ISO-8859-1 for text/*
    # without a header charset, which garbles UTF-8 pages. BeautifulSoup honours
    # the header charset when given, else sniffs <meta charset> / BOM / content.
    soup = BeautifulSoup(raw, "html.parser", from_encoding=charset)
    for tag in soup(["script", "style", "noscript", "template", "nav", "aside"]):
        tag.decompose()
    # Site-wide header/footer are chrome, but a <header> inside the article is
    # the article's own title block — keep that.
    for tag in soup(["header", "footer"]):
        if not tag.decomposed and tag.find_parent(["article", "main"]) is None:
            tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def load_url(url: str) -> List[Document]:
    """
    Fetch a web page and return its visible text as a single Document.

    We strip script/style/nav/aside and the site-level header/footer so the
    embedding model sees article content, not menus and cookie banners. This
    noise removal at ingestion time directly improves retrieval quality.

    Raises ValueError (incl. UnsafeURLError) for unsafe/unsupported URLs and for
    pages with no extractable text.
    """
    url = validate_url(url)
    # The guarded fetch runs first even when we'll end up using the renderer: it
    # is what proves the host is public (SSRF guard) before we hand the URL on.
    raw, mime, charset = _fetch(url)

    if mime in _TEXT_TYPES:
        text = UnicodeDammit(raw, [charset] if charset else []).unicode_markup or ""
        text = text.strip()
    else:
        text = _html_to_text(raw, charset)
        if len(text) < URL_FALLBACK_BELOW_CHARS and URL_JS_FALLBACK == "jina":
            try:
                rendered = _read_via_jina(url)
            except Exception as exc:
                logger.warning("JS-render fallback failed for %s: %s", url, exc)
            else:
                if len(rendered) > len(text):
                    logger.info(
                        "url ingest: static text %d chars, used rendered page "
                        "(%d chars) for %s", len(text), len(rendered), url,
                    )
                    text = rendered

    if len(text) < URL_MIN_TEXT_CHARS:
        raise ValueError(
            f"Too little readable text at the URL ({len(text)} characters). The "
            "page is probably built by JavaScript in the browser, or is empty — "
            "try saving it as a PDF and uploading that instead."
        )
    return [Document(page_content=text, metadata={"source": url})]


# [text](href) → text. Rendered markdown repeats URLs inline, which is pure noise
# for embeddings; images are dropped entirely.
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)\s]*(?:\s+\"[^\"]*\")?\)")


def _read_via_jina(url: str) -> str:
    """Render `url` with Jina Reader and return the main content as markdown
    (headings kept — the chunker splits on them), minus nav/header/footer."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Retain-Images": "none",
        "X-Remove-Selector": "nav, header, footer, aside",
    }
    key = os.getenv("JINA_API_KEY")
    if key:  # keyless works too, at a lower rate limit
        headers["Authorization"] = f"Bearer {key}"

    with requests.post(
        JINA_READER_URL,
        json={"url": url},
        headers=headers,
        timeout=JINA_READER_TIMEOUT,
        stream=True,
    ) as resp:
        resp.raise_for_status()
        body = resp.raw.read(URL_MAX_BYTES + 1, decode_content=True)
    if len(body) > URL_MAX_BYTES:
        raise ValueError("Rendered page is too large")

    data = json.loads(body).get("data") or {}
    content = _MD_IMAGE.sub("", data.get("content") or "")
    content = _MD_LINK.sub(r"\1", content).strip()
    title = (data.get("title") or "").strip()
    if title and title not in content[:500]:
        content = f"# {title}\n\n{content}"
    return content


def load_url2(url: str) -> List[Document]:

    loader = AsyncHtmlLoader([url])
    docs = loader.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    text = "\n".join([doc.page_content for doc in docs_transformed])
    return [Document(page_content=text, metadata={"source": url})]


def is_digital_pdf(path):
    reader = PdfReader(path)
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            return True  # has real text
    return False  # likely scanned


def _ocr_pdf(path: str) -> List[Document]:
    """
    OCR a scanned PDF: render each page to an image with PyMuPDF (300 DPI) and
    run it through Tesseract. PyMuPDF rasterises natively, so no poppler binary
    is needed — only the Tesseract engine (see `_ocr`).
    """
    page_texts: List[str] = []
    with pymupdf.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            with Image.open(io.BytesIO(pix.tobytes("png"))) as img:
                page_texts.append(_ocr(img))

    text = "\n".join(page_texts).strip()
    if not text:
        raise ValueError("OCR found no readable text in the scanned PDF")
    return [Document(page_content=text, metadata={"source": ""})]


def load_pdf(path: str) -> List[Document]:

    isDigital = is_digital_pdf(path)
    if isDigital:
        loader = PyPDFLoader(path)
        docs = loader.load()

        text = "\n".join([doc.page_content for doc in docs])
        return [Document(page_content=text, metadata={"source": ""})]
    else:
        # No embedded text layer → scanned/image PDF; fall back to OCR.
        return _ocr_pdf(path)


def load_txt(path: str) -> List[Document]:

    loader = TextLoader(path)
    docs = loader.load()

    text = "\n".join([doc.page_content for doc in docs])
    return [Document(page_content=text, metadata={"source": ""})]


def load_docx(path: str) -> List[Document]:

    loader = Docx2txtLoader(path)
    docs = loader.load()

    text = "\n".join([doc.page_content for doc in docs])
    return [Document(page_content=text, metadata={"source": ""})]


def _ocr(img: Image.Image) -> str:
    """
    Run Tesseract on one PIL image. Requires the Tesseract OCR *engine* on the
    host (the pip package is only a wrapper): macOS `brew install tesseract`,
    Debian/Ubuntu `apt-get install tesseract-ocr`.
    """
    try:
        return pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError as exc:
        raise ValueError(
            "Tesseract OCR engine is not installed on the server "
            "(install `tesseract-ocr` / `brew install tesseract`)"
        ) from exc


def load_image(path: str) -> List[Document]:
    """
    OCR an image into a single text Document. The upload size cap is enforced by
    the route before this runs.
    """
    with Image.open(path) as img:
        text = _ocr(img).strip()

    if not text:
        # No extractable text — surface it instead of ingesting an empty doc.
        raise ValueError("OCR found no readable text in the image")
    return [Document(page_content=text, metadata={"source": ""})]


SUPPORTED_FILE = {
    # Images (OCR via Tesseract; capped at 5 MB by the ingest route)
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    # "image/gif": ".gif",
    # "image/svg+xml": ".svg",
    # "image/x-icon": ".ico",
    # Documents & Applications
    "application/pdf": ".pdf",
    # "application/json": ".json",
    # "application/xml": ".xml",
    # "application/zip": ".zip",
    # "application/x-7z-compressed": ".7z",
    # "application/x-rar-compressed": ".rar",
    # "application/x-tar": ".tar",
    # "application/octet-stream": ".bin",
    # Microsoft Office / Open Office Documents
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    # "application/msword": ".doc",
    # "application/vnd.ms-excel": ".xls",
    # "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    # "application/vnd.ms-powerpoint": ".ppt",
    # "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    # Text Files
    "text/plain": ".txt",
    # "text/html": ".html",
    # "text/css": ".css",
    # "text/csv": ".csv",
    # "text/javascript": ".js",
    # "text/markdown": ".md",
    # Video (requires transcription)
    # "video/mp4": ".mp4",
    # "video/webm": ".webm",
    # "video/ogg": ".ogv",
    # "video/quicktime": ".mov",
    # "video/x-msvideo": ".avi",
    # Audio (requires transcription)
    # "audio/mpeg": ".mp3",
    # "audio/wav": ".wav",
    # "audio/webm": ".weba",
    # "audio/ogg": ".ogg",
    # "audio/midi": ".mid",
    # "audio/aac": ".aac",
}
