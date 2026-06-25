"""
services/ingestion.py
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

from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import AsyncHtmlLoader
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


SUPPORTED_FILE_TYPES = {
    "application/pdf": {"suffix": ".pdf", "file_type": "pdf", "loader": PyPDFLoader},
    "text/plain": {"suffix": ".txt", "file_type": "text", "loader": TextLoader},
}
