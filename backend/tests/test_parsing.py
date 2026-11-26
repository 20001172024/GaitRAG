"""Tests for PDF text cleanup and parsing."""

from pathlib import Path

import pymupdf

from app.rag.parsing import clean_pages, join_lines, parse_pdf

HEADER = "Journal of Gait Studies | www.example.org"


def _page(n: int, body: list[str]) -> str:
    return "\n".join([HEADER, f"Page {n} of 3", *body, f"Volume 1 | Article {n}"])


def test_running_headers_and_page_numbers_are_removed() -> None:
    bodies = [
        ["Walking speed was recorded.", "Cadence was computed per trial."],
        ["Stride time was averaged.", "Variability was expressed as CV."],
        ["Stance phase was longer at slow speed.", "Symmetry was high."],
    ]
    pages = [_page(i, body) for i, body in enumerate(bodies, start=1)]
    text = " ".join(s.text for s in clean_pages(pages))
    assert "www.example.org" not in text
    assert "Page 2 of 3" not in text
    assert "Article" not in text
    assert "Stride time was averaged. Variability was expressed as CV." in text


def test_ligatures_are_normalized() -> None:
    segs = clean_pages(["The ﬁndings were signiﬁcant."])
    assert segs[0].text == "The findings were significant."


def test_hyphenated_words_are_rejoined() -> None:
    assert join_lines(["stride time vari-", "ability was low"]) == "stride time variability was low"
    # Capitalized continuation is a real hyphenated compound or a new sentence, keep it.
    assert join_lines(["Timed Up-", "And-Go"]) == "Timed Up- And-Go"


def test_sections_are_labelled_and_numbered_headings_recognized() -> None:
    page = "\n".join(
        [
            "Introduction",
            "Gait matters.",
            "2. Materials and Methods",
            "We recruited adults.",
            "3. The Pressure-Sensitive Insole",
            "It has sensors.",
            "2.1. Sample Size",
            "Twenty people.",
        ]
    )
    segs = clean_pages([page])
    assert [(s.section, s.text) for s in segs] == [
        ("Introduction", "Gait matters."),
        ("Methods", "We recruited adults."),
        ("The Pressure-Sensitive Insole", "It has sensors. 2.1. Sample Size Twenty people."),
    ]


def test_everything_after_references_is_dropped() -> None:
    pages = [
        "Results\nCadence was 110 steps/min.",
        "References\n1. Smith J, Doe A. Gait. 2010.\n12. Selles RW, Formanoy MA\nDiscussion\nnot real",
    ]
    segs = clean_pages(pages)
    assert [s.text for s in segs] == ["Cadence was 110 steps/min."]


def test_back_matter_is_skipped_but_later_sections_kept() -> None:
    page = "Funding\nThis work was funded by X.\nConclusion\nGait speed matters."
    segs = clean_pages([page])
    assert [(s.section, s.text) for s in segs] == [("Conclusion", "Gait speed matters.")]


def test_page_numbers_are_preserved() -> None:
    segs = clean_pages(["Methods\nFirst page.", "Second page."])
    assert [(s.page, s.section) for s in segs] == [(1, "Methods"), (2, "Methods")]


def test_parse_pdf_reads_a_real_pdf(tmp_path: Path) -> None:
    path = tmp_path / "t.pdf"
    doc = pymupdf.open()
    for body in ("Introduction\nWalking is studied.", "Results\nCadence rose with speed."):
        doc.new_page().insert_text((72, 72), body)
    doc.save(path)
    parsed = parse_pdf(path)
    assert parsed.n_pages == 2
    assert [(s.page, s.section, s.text) for s in parsed.segments] == [
        (1, "Introduction", "Walking is studied."),
        (2, "Results", "Cadence rose with speed."),
    ]
    assert parsed.tables == []
