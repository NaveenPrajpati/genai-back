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
from typing import List

import requests
import pymupdf
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    AsyncHtmlLoader,
)
from langchain_community.document_transformers import Html2TextTransformer


def load_url(url: str) -> List[Document]:
    """
    Fetch a web page and return its visible text as a single Document.

    We strip script/style/nav/footer/header/aside so the embedding model sees
    article content, not menus and cookie banners. This noise removal at
    ingestion time directly improves retrieval quality downstream.
    """
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return [Document(page_content=text, metadata={"source": url})]


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
