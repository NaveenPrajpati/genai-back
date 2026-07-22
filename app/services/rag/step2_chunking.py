"""
services/rag/step2_chunking.py
=================
STEP 2 OF THE RAG PIPELINE: CHUNKING (a.k.a. "splitting")

Chunking = cutting each loaded Document into smaller pieces. Why? Two reasons:
  1. Embedding models have a token limit and lose precision on long text.
  2. Retrieval returns *chunks*, so chunk size = the granularity of an answer.
     Too big → you retrieve a whole page to answer one sentence (noisy context,
     wasted tokens). Too small → you lose the surrounding context needed to
     understand the sentence.

──────────────────────────────────────────────────────────────────────────────
YOUR ORIGINAL CODE  (kept, with the bug fixed)
──────────────────────────────────────────────────────────────────────────────
Your factory had a subtle bug: it accepted a `SplitStrategy` enum but then
compared against raw strings ("recursive", "semantic"). So passing the enum
member `SplitStrategy.RECURSIVE` fell through to `None`. Fixed below by
comparing enum-to-enum and accepting strings too.

──────────────────────────────────────────────────────────────────────────────
WHICH CHUNKER, WHEN
──────────────────────────────────────────────────────────────────────────────
  FIXED (CharacterTextSplitter)
      Splits every N characters. Simplest, fastest, but blindly cuts mid-
      sentence. USE WHEN: text has no structure (logs, transcripts) or you just
      need a quick baseline.

  RECURSIVE (RecursiveCharacterTextSplitter)   ← your default, good choice
      Tries to split on paragraph → sentence → word boundaries in order, so
      chunks rarely cut mid-sentence. USE WHEN: general prose, articles, PDFs.
      This is the sensible default for ~90% of cases.

  SEMANTIC (SemanticChunker)
      Embeds sentences and starts a new chunk when meaning shifts (detected by
      a similarity drop). Produces topically-coherent chunks of variable size.
      USE WHEN: documents cover many topics and you want each chunk to be "one
      idea". COST: it embeds during chunking, so ingestion is slower/pricier.

──────────────────────────────────────────────────────────────────────────────
ADVANCED CHUNKING (beyond this factory — for when you go "pro")
──────────────────────────────────────────────────────────────────────────────
  • STRUCTURE-AWARE / DOCUMENT-AWARE
        Split on real document structure: Markdown headers, HTML tags, code
        functions, PDF sections. Keeps a table or a code block intact and can
        attach the heading path to metadata. Tools: MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter.from_language(...), Unstructured's
        "by_title" chunking. USE WHEN: docs have clear headings / code / tables.

  • PARENT–CHILD (a.k.a. "small-to-big" / ParentDocumentRetriever)
        Embed small child chunks (precise retrieval) but return the larger parent
        chunk to the LLM (full context). You get accurate matching AND enough
        surrounding context. USE WHEN: precise lookup matters but isolated small
        chunks are too thin to answer from. Very high ROI upgrade.

  • LATE CHUNKING
        Embed the WHOLE document first with a long-context embedding model, then
        pool token embeddings into chunk embeddings afterward. Each chunk's
        vector "remembers" the full-document context, fixing the classic problem
        where a chunk says "it increased 12%" with no idea what "it" is.
        USE WHEN: lots of cross-references / pronouns across a long document.

  • PROPOSITION / SENTENCE-WINDOW
        Index atomic facts or single sentences, then expand to a window of
        neighbours at retrieval time. USE WHEN: you need pinpoint factual recall.
"""

from enum import Enum

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


class SplitStrategy(str, Enum):
    """str-Enum so both `SplitStrategy.RECURSIVE` and the string "recursive" work."""
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


def get_splitter(
    strategy: SplitStrategy | str = SplitStrategy.RECURSIVE,
    embeddings=None,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Return a configured text splitter.

    Args:
        strategy:       FIXED | RECURSIVE | SEMANTIC (enum or string).
        embeddings:     required ONLY for SEMANTIC.
        chunk_size:     target characters per chunk (FIXED/RECURSIVE).
        chunk_overlap:  characters shared between neighbouring chunks. Overlap
                        preserves context that would otherwise be severed at a
                        boundary; ~10–20% of chunk_size is a good rule of thumb.
    """
    strategy = SplitStrategy(strategy)  # normalizes "recursive" → SplitStrategy.RECURSIVE

    if strategy is SplitStrategy.FIXED:
        return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if strategy is SplitStrategy.RECURSIVE:
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if strategy is SplitStrategy.SEMANTIC:
        if embeddings is None:
            raise ValueError("`embeddings` is required for the SEMANTIC strategy")
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95.0,
        )

    raise ValueError(f"Unknown split strategy: {strategy}")
