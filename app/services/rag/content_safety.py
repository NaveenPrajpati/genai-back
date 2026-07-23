"""
services/rag/content_safety.py
==============================
Ingestion-time defenses against INDIRECT PROMPT INJECTION carried by untrusted
documents (a PDF that says "ignore your instructions and…"). Two pure helpers,
no I/O, easy to unit-test:

  * sanitize_text      — normalise + strip characters used to HIDE instructions
                         from a human but not from a text extractor (zero-width,
                         control chars, compatibility/homoglyph forms), applied
                         before a chunk is embedded/stored.
  * scan_for_injection — a heuristic DETECTOR that returns the names of matched
                         injection patterns, so the caller can log/meter them.

Neither BLOCKS content: sanitisation only removes invisible/garbage characters,
and detection only reports. The actual refusing is done at the prompt layer
(core/prompts.py marks the context as untrusted data). Heuristics are bypassable
— this raises the bar and gives visibility, it is NOT a guarantee. Flagged
chunks are still ingested by design (log + metric), so a false positive never
silently drops legitimate document content.
"""

import re
import unicodedata

# Zero-width / BOM characters: invisible to a reader, still extracted from a PDF —
# a common way to smuggle instructions past human review. (ZWSP, ZWNJ, ZWJ,
# word-joiner, BOM.) Written as escapes so they stay visible/robust in source.
_ZERO_WIDTH = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF], None
)

# C0/C1 control characters, except the whitespace we want to keep (\t \n \r).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Pathological whitespace runs used to push visible text off-screen.
_LONG_SPACES_RE = re.compile(r"[ \t]{4,}")
_MANY_BLANKLINES_RE = re.compile(r"\n{4,}")


def sanitize_text(text: str) -> str:
    """Normalise and strip characters commonly used to smuggle hidden
    instructions, while leaving legitimate content intact. NFKC folds
    compatibility/homoglyph forms so both the model and the detector below see
    canonical text."""
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH)
    text = _CONTROL_RE.sub("", text)
    text = _LONG_SPACES_RE.sub("  ", text)
    text = _MANY_BLANKLINES_RE.sub("\n\n", text)
    return text


# Known indirect-injection phrasings, matched case-insensitively against
# sanitised text. Deliberately conservative wording to limit false positives;
# because flagging is log-only, an occasional false positive is cheap.
_INJECTION_PATTERNS = {
    "ignore_instructions": r"ignore\s+(?:all\s+|the\s+|your\s+)?(?:previous|prior|above|earlier|preceding)\s+instructions",
    "disregard_above": r"disregard\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above|earlier|preceding)\b",
    "reveal_prompt": r"(?:reveal|print|repeat|show|output|display)\s+(?:me\s+)?(?:your\s+|the\s+)?(?:system\s+)?(?:prompt|instructions)",
    "role_override": r"\byou\s+are\s+now\b|\bfrom\s+now\s+on,?\s+you\b|\bnew\s+instructions\s*:",
    "role_marker": r"^\s*(?:system|assistant)\s*:",
    "chat_template_token": r"<\|(?:im_start|im_end|system|user|assistant)\|>",
}
_COMPILED = {
    name: re.compile(pat, re.IGNORECASE | re.MULTILINE)
    for name, pat in _INJECTION_PATTERNS.items()
}


def scan_for_injection(text: str) -> list[str]:
    """Return the names of injection patterns present in `text` (empty = clean).
    Heuristic and bypassable — intended for logging/metrics, never for blocking."""
    if not text:
        return []
    return [name for name, rx in _COMPILED.items() if rx.search(text)]
