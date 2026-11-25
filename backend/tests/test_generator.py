"""Tests for cited answer generation (scripted FakeLLM, no network)."""

from app.rag.generator import check_citations, generate_answer, sources_from_hits
from app.rag.llm import FakeLLM
from app.rag.norms import NormComparison
from app.rag.prompts import INSUFFICIENT_MARKER, SYSTEM_PROMPT
from app.rag.retriever import ChunkInfo, Hit


def _hit(cid: int, paper: str, page: int, text: str, table: str | None = None, rr=None) -> Hit:
    info = ChunkInfo(
        cid, paper, f"Title {paper}", page, "Results", text, "table" if table else "text", table
    )
    return Hit(chunk=info, score=1.0, rerank_score=rr)


HITS = [
    _hit(1, "P008", 3, "Stride time variability rose when speed decreased."),
    _hit(2, "P006", 5, "Cadence (step/min) — Young: 111.87 (9.27)", table="2"),
    _hit(3, "P008", 3, "This effect was significant (p < 0.05)."),
]

COMPARISON = NormComparison(
    feature="cadence_steps_per_min",
    value=112.0,
    units="steps/min",
    paper_id="P003",
    page=6,
    table_no="4",
    population="20–29 | M",
    norm_text="109.88 (6.46)",
    norm_mean=109.88,
    norm_sd=6.46,
    norm_low=None,
    norm_high=None,
    z_score=(112.0 - 109.88) / 6.46,
    within=True,
)


def test_sources_group_same_page_under_one_label() -> None:
    sources = sources_from_hits(HITS)
    assert [s.label for s in sources] == ["[P008, p.3]", "[P006, p.5, Table 2]"]
    assert len(sources[0].texts) == 2


def test_check_citations() -> None:
    allowed = {"[P008, p.3]", "[P006, p.5, Table 2]"}
    answer = "Variability rises at slow speed [P008, p.3]. Cadence is high [P099, p.1]. No cite."
    valid, invalid, uncited = check_citations(answer, allowed)
    assert valid == ["[P008, p.3]"]
    assert invalid == ["[P099, p.1]"]
    assert uncited == 1
    assert check_citations("Bad [P6, page 5].", allowed)[1] == ["[P6, page 5]"]


def test_valid_answer_is_returned_with_citations() -> None:
    llm = FakeLLM(
        lambda m: "Your cadence is within 1 SD of the group [P003, p.6, Table 4]. "
        "Variability rises at slower speed [P008, p.3]."
    )
    out = generate_answer(llm, "Is my cadence typical?", HITS, [COMPARISON])
    assert not out.insufficient_evidence
    assert out.citations == ["[P003, p.6, Table 4]", "[P008, p.3]"]
    assert out.attempts == 1 and out.uncited_sentences == 0
    system, user = llm.calls[0]
    assert system.content == SYSTEM_PROMPT
    assert "[P006, p.5, Table 2]" in user.content and "z = +0.33" in user.content


def test_invented_citation_triggers_one_retry_with_feedback() -> None:
    replies = iter(
        ["Cadence is normal [P042, p.9].", "Cadence is within 1 SD [P003, p.6, Table 4]."]
    )
    llm = FakeLLM(lambda m: next(replies))
    out = generate_answer(llm, "q?", HITS, [COMPARISON])
    assert out.attempts == 2 and not out.insufficient_evidence
    assert "[P042, p.9]" in llm.calls[1][-1].content  # feedback names the bad label


def test_still_invalid_after_retry_becomes_insufficient_evidence() -> None:
    llm = FakeLLM(lambda m: "Made up [P042, p.9].")
    out = generate_answer(llm, "q?", HITS, [])
    assert out.insufficient_evidence and out.reason == "citation_check_failed"
    assert out.citations == [] and out.invalid_citations == ["[P042, p.9]"]
    assert "[P042" not in out.answer


def test_uncited_answer_is_retried() -> None:
    replies = iter(["Cadence is fine.", "Cadence is within 1 SD [P003, p.6, Table 4]."])
    out = generate_answer(FakeLLM(lambda m: next(replies)), "q?", HITS, [COMPARISON])
    assert out.attempts == 2 and out.citations == ["[P003, p.6, Table 4]"]


def test_model_abstention_is_reported() -> None:
    out = generate_answer(FakeLLM(lambda m: INSUFFICIENT_MARKER), "q?", HITS, [])
    assert out.insufficient_evidence and out.reason == "model_abstained"


def test_no_sources_skips_the_llm() -> None:
    llm = FakeLLM(lambda m: "should not be called")
    out = generate_answer(llm, "q?", [], [])
    assert out.insufficient_evidence and out.reason == "no_sources" and llm.calls == []


def test_low_relevance_gate_only_when_calibrated() -> None:
    weak = [_hit(1, "P008", 3, "Unrelated text.", rr=-9.0)]
    llm = FakeLLM(lambda m: "Something [P008, p.3].")
    assert generate_answer(llm, "q?", weak, [], min_rerank_score=-2.0).reason == "low_relevance"
    assert generate_answer(llm, "q?", weak, [], min_rerank_score=None).reason == "answered"
