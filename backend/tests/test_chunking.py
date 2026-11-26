"""Tests for sentence splitting and chunking."""

from app.rag.chunking import chunk_segments, split_sentences
from app.rag.parsing import Segment


def words(text: str) -> int:
    """Test token counter: one token per word."""
    return len(text.split())


def _sentences(n: int, length: int = 10) -> str:
    return " ".join(f"Sentence {i} " + "word " * (length - 3) + "end." for i in range(n))


def test_split_keeps_abbreviations_and_decimals() -> None:
    text = "Gait was slower (Hausdorff et al. 2005). Mean was 1.05 s vs. 1.10 s. See Fig. 2 here."
    assert split_sentences(text) == [
        "Gait was slower (Hausdorff et al. 2005).",
        "Mean was 1.05 s vs. 1.10 s.",
        "See Fig. 2 here.",
    ]


def test_chunks_respect_max_tokens_and_overlap() -> None:
    seg = Segment(page=1, section="Methods", text=_sentences(40))  # 400 words
    chunks = chunk_segments([seg], words, max_tokens=100, overlap_tokens=20, min_tokens=5)
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)
    for a, b in zip(chunks, chunks[1:], strict=False):
        last_two = " ".join(split_sentences(a.text)[-2:])
        assert b.text.startswith(last_two)  # 2 x 10-word sentences fit in 20 overlap tokens


def test_chunks_never_cross_pages_or_sections() -> None:
    segs = [
        Segment(1, "Methods", _sentences(3)),
        Segment(2, "Methods", _sentences(3)),
        Segment(2, "Results", _sentences(3)),
    ]
    chunks = chunk_segments(segs, words, max_tokens=300)
    assert [(c.page, c.section) for c in chunks] == [(1, "Methods"), (2, "Methods"), (2, "Results")]


def test_short_tail_is_merged_without_duplicating_overlap() -> None:
    text = _sentences(10) + " Tail."
    chunks = chunk_segments([Segment(1, None, text)], words, 50, 10, min_tokens=5)
    assert chunks[-1].text.endswith("end. Tail.")
    assert sum(c.text.count("Tail.") for c in chunks) == 1


def test_overlong_sentence_is_split_on_words() -> None:
    long_sentence = "word " * 250 + "end."
    chunks = chunk_segments([Segment(1, None, long_sentence)], words, 100, 0, min_tokens=1)
    assert all(c.token_count <= 100 for c in chunks)
    assert sum(c.token_count for c in chunks) == 251


def test_tiny_noise_is_dropped() -> None:
    assert chunk_segments([Segment(3, None, "Fig. 2")], words) == []
