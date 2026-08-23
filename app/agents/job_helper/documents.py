"""Resume files in, resume files out. No LLM calls live here.

IN  — `extract_text` turns an uploaded .docx/.pdf/.txt into plain text for the
      parsing prompt in service.py. Nothing more: layout is not preserved,
      because the structure the agent needs is recovered by the model, not by
      guessing at indentation.

OUT — `render_docx` / `render_pdf` build a fresh document from a `ResumeContent`.
      The user's original file is never edited. Rewriting a .docx in place means
      matching new text to existing runs, and a resume's runs are split at
      arbitrary points (bold fragments, spell-check boundaries), so any edit
      either loses formatting or corrupts the paragraph. Generating a clean
      single-column document instead is also the layout that survives a resume
      parser, which is the one reader guaranteed to see it.

Both renderers are pure-Python (python-docx, reportlab) on purpose: converting
via headless LibreOffice would be a ~500 MB system dependency in an image whose
whole design is a slim runtime with no compilers.
"""

import io
import logging
import os
from typing import List
from xml.sax.saxutils import escape

from docx import Document
from docx.document import Document as DocxDocument  # the class; `Document` is its factory
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from pypdf import PdfReader
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from .state import ResumeContent, ResumeExperience

logger = logging.getLogger(__name__)

# A resume is one to three pages. Anything past this is a different document
# (a thesis, a merged PDF) and only inflates the parsing prompt.
MAX_RESUME_CHARS = 30_000

SUPPORTED_UPLOADS = {
    ".docx": "docx",
    ".pdf": "pdf",
    ".txt": "txt",
    ".md": "txt",
}

ACCENT = "1F3864"  # deep navy; the only colour either renderer uses


class UnsupportedResumeFile(ValueError):
    """Raised for a file type neither renderer nor parser can read."""


# ── in ───────────────────────────────────────────────────────────────────────
def file_kind(filename: str) -> str:
    """The reader to use for `filename`, or raise for anything unsupported."""
    ext = os.path.splitext(filename or "")[1].lower()
    kind = SUPPORTED_UPLOADS.get(ext)
    if not kind:
        raise UnsupportedResumeFile(
            "Upload a .docx, .pdf or .txt resume — "
            f"{ext or 'that file'} cannot be read."
        )
    return kind


def _docx_text(path: str) -> str:
    """Paragraphs plus table cells.

    Tables matter: a large share of real resumes are laid out as an invisible
    two-column table, and reading only `document.paragraphs` returns an empty
    string for every one of them.
    """
    document = Document(path)
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            # A row's cells are read left to right; joining with a separator
            # keeps "Company | 2021-2023" from running into one word.
            if cells:
                parts.append("  |  ".join(dict.fromkeys(cells)))
    return "\n".join(parts)


def _pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:  # one unreadable page must not lose the rest
            logger.warning("resume pdf page extract failed: %s", e)
    return "\n".join(p.strip() for p in pages if p.strip())


def extract_text(path: str, filename: str) -> str:
    """Plain text of an uploaded resume, capped at MAX_RESUME_CHARS.

    Raises UnsupportedResumeFile for an unreadable type, and ValueError when a
    supported file yields nothing — a scanned-image PDF reaches here as a
    perfectly valid file containing zero characters, and the caller needs to say
    so rather than hand an empty string to the parser.
    """
    kind = file_kind(filename)
    if kind == "docx":
        text = _docx_text(path)
    elif kind == "pdf":
        text = _pdf_text(path)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()

    text = text.strip()
    if not text:
        raise ValueError(
            "No text could be read from that file. If it is a scanned or "
            "image-only PDF, upload the .docx or a text version instead."
        )
    return text[:MAX_RESUME_CHARS]


# ── out: shared shaping ──────────────────────────────────────────────────────
def _as_content(content) -> ResumeContent:
    """Accept either the model or the stored dict — routes hand over a Mongo
    document, the agent hands over what the LLM produced."""
    if isinstance(content, ResumeContent):
        return content
    return ResumeContent.model_validate(content or {})


def _contact_line(content: ResumeContent) -> str:
    contact = content.contact
    bits = [contact.email, contact.phone, contact.location, *contact.links]
    return "  •  ".join(b for b in bits if b)


def _date_range(job: ResumeExperience) -> str:
    if job.start and job.end:
        return f"{job.start} – {job.end}"
    return job.start or job.end or ""


def _role_line(job: ResumeExperience) -> str:
    left = " — ".join(b for b in (job.title, job.company) if b)
    return left or (job.company or job.title or "")


def suggest_filename(content: ResumeContent, label: str, extension: str) -> str:
    """`Naveen_Prajapati_Backend_Engineer.pdf` — a name a recruiter can file.

    Falls back to the version label when the resume carries no name, and to a
    generic stem when neither is usable.
    """
    stem = " ".join(p for p in (content.contact.name, label) if p) or "resume"
    safe = "".join(c if c.isalnum() or c in " -_" else "" for c in stem).strip()
    safe = "_".join(safe.split()) or "resume"
    return f"{safe[:80]}.{extension}"


# ── out: .docx ───────────────────────────────────────────────────────────────
def _docx_heading(document: DocxDocument, text: str) -> None:
    """A section rule: small caps-ish label with a bottom border via underline.

    python-docx exposes no paragraph border API, so the rule under a section
    heading would need raw XML. An underlined heading reads the same on screen
    and stays legible to a parser, which is the trade worth making here.
    """
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(10)
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(text.upper())
    run.bold = True
    run.underline = True
    run.font.size = Pt(10.5)
    run.font.color.rgb = RGBColor.from_string(ACCENT)


def _docx_bullets(document: DocxDocument, bullets: List[str]) -> None:
    for bullet in bullets:
        if not bullet:
            continue
        paragraph = document.add_paragraph(bullet, style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.paragraph_format.left_indent = Inches(0.25)


def render_docx(content, label: str = "Resume") -> bytes:
    """A single-column .docx built from the structured resume."""
    content = _as_content(content)
    document = Document()

    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    for section in document.sections:
        section.top_margin = section.bottom_margin = Inches(0.6)
        section.left_margin = section.right_margin = Inches(0.7)

    name = document.add_paragraph()
    name.alignment = WD_ALIGN_PARAGRAPH.CENTER
    name.paragraph_format.space_after = Pt(0)
    name_run = name.add_run(content.contact.name or label)
    name_run.bold = True
    name_run.font.size = Pt(20)
    name_run.font.color.rgb = RGBColor.from_string(ACCENT)

    if content.headline:
        headline = document.add_paragraph()
        headline.alignment = WD_ALIGN_PARAGRAPH.CENTER
        headline.paragraph_format.space_after = Pt(0)
        headline.add_run(content.headline).font.size = Pt(11)

    contact_line = _contact_line(content)
    if contact_line:
        contact = document.add_paragraph()
        contact.alignment = WD_ALIGN_PARAGRAPH.CENTER
        contact.add_run(contact_line).font.size = Pt(9)

    if content.summary:
        _docx_heading(document, "Summary")
        document.add_paragraph(content.summary)

    if content.skills:
        _docx_heading(document, "Skills")
        document.add_paragraph("  •  ".join(content.skills))

    if content.experience:
        _docx_heading(document, "Experience")
        for job in content.experience:
            row = document.add_paragraph()
            row.paragraph_format.space_after = Pt(0)
            row.add_run(_role_line(job)).bold = True
            meta = "  |  ".join(b for b in (job.location, _date_range(job)) if b)
            if meta:
                run = row.add_run(f"    {meta}")
                run.italic = True
                run.font.size = Pt(9)
            _docx_bullets(document, job.bullets)

    if content.projects:
        _docx_heading(document, "Projects")
        for project in content.projects:
            row = document.add_paragraph()
            row.paragraph_format.space_after = Pt(0)
            row.add_run(project.name or "Project").bold = True
            if project.link:
                run = row.add_run(f"    {project.link}")
                run.italic = True
                run.font.size = Pt(9)
            _docx_bullets(document, project.bullets)

    if content.education:
        _docx_heading(document, "Education")
        for school in content.education:
            row = document.add_paragraph()
            row.paragraph_format.space_after = Pt(0)
            row.add_run(
                " — ".join(b for b in (school.degree, school.institution) if b)
            ).bold = True
            if school.year:
                run = row.add_run(f"    {school.year}")
                run.italic = True
                run.font.size = Pt(9)
            if school.details:
                document.add_paragraph(school.details)

    if content.certifications:
        _docx_heading(document, "Certifications")
        _docx_bullets(document, content.certifications)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# ── out: .pdf ────────────────────────────────────────────────────────────────
def _styles():
    """Paragraph styles for the PDF, derived from reportlab's defaults."""
    base = getSampleStyleSheet()["BodyText"]
    body = ParagraphStyle(
        "ResumeBody", parent=base, fontName="Helvetica", fontSize=9.5, leading=12.5,
        spaceAfter=2,
    )
    return {
        "body": body,
        "name": ParagraphStyle(
            "ResumeName", parent=body, fontName="Helvetica-Bold", fontSize=19,
            leading=22, alignment=TA_CENTER, textColor=f"#{ACCENT}", spaceAfter=1,
        ),
        "headline": ParagraphStyle(
            "ResumeHeadline", parent=body, fontSize=10.5, leading=13,
            alignment=TA_CENTER, spaceAfter=1,
        ),
        "contact": ParagraphStyle(
            "ResumeContact", parent=body, fontSize=8.5, leading=11,
            alignment=TA_CENTER, spaceAfter=4,
        ),
        "section": ParagraphStyle(
            "ResumeSection", parent=body, fontName="Helvetica-Bold", fontSize=10,
            leading=12, textColor=f"#{ACCENT}", spaceBefore=9, spaceAfter=1,
        ),
        "role": ParagraphStyle(
            "ResumeRole", parent=body, fontName="Helvetica-Bold", fontSize=10,
            leading=12.5, spaceBefore=4, spaceAfter=0,
        ),
        "meta": ParagraphStyle(
            "ResumeMeta", parent=body, fontName="Helvetica-Oblique", fontSize=8.5,
            leading=11, spaceAfter=1,
        ),
    }


def _pdf_section(flow: list, styles: dict, title: str) -> None:
    flow.append(Paragraph(escape(title.upper()), styles["section"]))
    flow.append(
        HRFlowable(width="100%", thickness=0.6, color=f"#{ACCENT}", spaceAfter=3)
    )


def _pdf_bullets(flow: list, styles: dict, bullets: List[str]) -> None:
    items = [
        ListItem(Paragraph(escape(b), styles["body"]), leftIndent=12)
        for b in bullets
        if b
    ]
    if items:
        flow.append(
            ListFlowable(items, bulletType="bullet", bulletFontSize=6, leftIndent=12)
        )


def render_pdf(content, label: str = "Resume") -> bytes:
    """The same resume as a PDF. Every string is XML-escaped: reportlab parses
    Paragraph text as markup, so an unescaped `R&D` or `C++ <-> Python` in a
    bullet would raise mid-render."""
    content = _as_content(content)
    styles = _styles()
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        title=f"{content.contact.name or label} — Resume",
        author=content.contact.name or "",
    )

    flow: list = [Paragraph(escape(content.contact.name or label), styles["name"])]
    if content.headline:
        flow.append(Paragraph(escape(content.headline), styles["headline"]))
    contact_line = _contact_line(content)
    if contact_line:
        flow.append(Paragraph(escape(contact_line), styles["contact"]))

    if content.summary:
        _pdf_section(flow, styles, "Summary")
        flow.append(Paragraph(escape(content.summary), styles["body"]))

    if content.skills:
        _pdf_section(flow, styles, "Skills")
        flow.append(Paragraph(escape("  •  ".join(content.skills)), styles["body"]))

    if content.experience:
        _pdf_section(flow, styles, "Experience")
        for job in content.experience:
            flow.append(Paragraph(escape(_role_line(job)), styles["role"]))
            meta = "  |  ".join(b for b in (job.location, _date_range(job)) if b)
            if meta:
                flow.append(Paragraph(escape(meta), styles["meta"]))
            _pdf_bullets(flow, styles, job.bullets)

    if content.projects:
        _pdf_section(flow, styles, "Projects")
        for project in content.projects:
            flow.append(Paragraph(escape(project.name or "Project"), styles["role"]))
            if project.link:
                flow.append(Paragraph(escape(project.link), styles["meta"]))
            _pdf_bullets(flow, styles, project.bullets)

    if content.education:
        _pdf_section(flow, styles, "Education")
        for school in content.education:
            title = " — ".join(b for b in (school.degree, school.institution) if b)
            flow.append(Paragraph(escape(title or "Education"), styles["role"]))
            meta = "  |  ".join(b for b in (school.year, school.details) if b)
            if meta:
                flow.append(Paragraph(escape(meta), styles["meta"]))

    if content.certifications:
        _pdf_section(flow, styles, "Certifications")
        _pdf_bullets(flow, styles, content.certifications)

    if len(flow) == 1:  # name only — nothing was parsed into the resume
        flow.append(Spacer(1, 12))
        flow.append(Paragraph("This resume has no content yet.", styles["body"]))

    document.build(flow)
    return buffer.getvalue()


RENDERERS = {"docx": render_docx, "pdf": render_pdf}

MEDIA_TYPES = {
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    "pdf": "application/pdf",
}


def render(content, fmt: str, label: str = "Resume") -> tuple[bytes, str, str]:
    """(bytes, media type, filename) for `fmt`, which must be "docx" or "pdf"."""
    renderer = RENDERERS.get(fmt)
    if not renderer:
        raise UnsupportedResumeFile(f"Cannot render a resume as {fmt!r}.")
    payload = renderer(content, label)
    filename = suggest_filename(_as_content(content), label, fmt)
    return payload, MEDIA_TYPES[fmt], filename
