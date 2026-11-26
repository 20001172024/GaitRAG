"""Split section-labelled page text into retrieval chunks.

Strategy (justification in the README):
- Units are sentences, packed greedily up to MAX_TOKENS (300 tokens of the
  embedding model's tokenizer). bge-small reads at most 512 tokens; 300 leaves
  room for the "title > section" header added at embedding time, and is small
  enough that one chunk carries one or two claims, which keeps citations precise.
- Overlap: the next chunk repeats the previous chunk's last whole sentences, up
  to OVERLAP_TOKENS (50). A claim cut at a boundary is then still retrievable in
  full, and chunks never start mid-sentence.
- Hard boundaries at page and section changes: every chunk has one exact page
  for its citation and one topic.
- Leftovers shorter than MIN_TOKENS are merged into the previous chunk, so a
  chunk can slightly exceed MAX_TOKENS (largest in the corpus: 350, still well
  under the model's 512 limit).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import groupby

from app.rag.parsing import Segment

MAX_TOKENS = 300
OVERLAP_TOKENS = 50
MIN_TOKENS = 20
MIN_KEEP_TOKENS = 8  # a whole page/section shorter than this is noise (e.g. a stray label)

TokenCounter = Callable[[str], int]

_ABBREVIATIONS = (
    "et al.",
    "e.g.",
    "i.e.",
    "fig.",
    "figs.",
    "vs.",
    "eq.",
    "ref.",
    "approx.",
    "cf.",
    "resp.",
    "no.",
    "dr.",
    "tab.",
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk before embedding."""

    page: int
    section: str | None
    text: str
    token_count: int


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping abbreviations like "et al." intact."""
    pieces = _SENTENCE_END.split(text.strip())
    sentences: list[str] = []
    for piece in pieces:
        if sentences and sentences[-1].lower().endswith(_ABBREVIATIONS):
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    return [s for s in sentences if s]


def _split_long(sentence: str, count: TokenCounter, max_tokens: int) -> list[str]:
    """Fallback for a single sentence over the limit: split on word boundaries."""
    parts, current = [], []
    for word in sentence.split():
        if current and count(" ".join([*current, word])) > max_tokens:
            parts.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def _pack(
    sentences: list[str],
    count: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
) -> list[list[str]]:
    """Greedy sentence packing with whole-sentence overlap.

    A final chunk whose *new* (non-overlap) sentences total fewer than
    `min_tokens` is merged into the previous chunk instead of standing alone.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    n_overlap = 0  # leading sentences of `current` copied from the previous chunk
    for sent in sentences:
        if current and sum(count(s) for s in current) + count(sent) > max_tokens:
            chunks.append(current)
            overlap: list[str] = []
            budget = overlap_tokens
            for prev in reversed(current):
                budget -= count(prev)
                if budget < 0:
                    break
                overlap.insert(0, prev)
            current, n_overlap = overlap, len(overlap)
        current.append(sent)

    if current:
        new = current[n_overlap:]
        if chunks and sum(count(s) for s in new) < min_tokens:
            chunks[-1] = chunks[-1] + new
        else:
            chunks.append(current)
    return chunks


def chunk_segments(
    segments: Iterable[Segment],
    count_tokens: TokenCounter,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    min_tokens: int = MIN_TOKENS,
) -> list[ChunkDraft]:
    """Chunk segments; chunks never cross a page or section boundary.

    Args:
        segments: output of `app.rag.parsing.clean_pages` / `parse_pdf`.
        count_tokens: token counter (use the embedding model's tokenizer).
        max_tokens: maximum tokens per chunk.
        overlap_tokens: maximum tokens repeated from the previous chunk.
        min_tokens: shorter leftovers are merged into the previous chunk.
    """
    drafts: list[ChunkDraft] = []
    for (page, section), group in groupby(segments, key=lambda s: (s.page, s.section)):
        sentences: list[str] = []
        for seg in group:
            for sent in split_sentences(seg.text):
                if count_tokens(sent) > max_tokens:
                    sentences.extend(_split_long(sent, count_tokens, max_tokens))
                else:
                    sentences.append(sent)

        packed = _pack(sentences, count_tokens, max_tokens, overlap_tokens, min_tokens)
        for text in (" ".join(c) for c in packed):
            n = count_tokens(text)
            if n >= MIN_KEEP_TOKENS:
                drafts.append(ChunkDraft(page, section, text, n))
    return drafts
