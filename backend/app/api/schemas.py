"""Response models shared by several endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.rag.norms import NormComparison


class NormComparisonOut(BaseModel):
    """A feature compared with one approved published norm (computed in code)."""

    feature: str
    value: float
    units: str
    citation: str = Field(description='e.g. "[P003, p.6, Table 4]"')
    paper_id: str
    page: int
    table_no: str | None
    population: str | None
    norm_text: str = Field(description="Verbatim table cell")
    norm_mean: float | None
    norm_sd: float | None
    norm_low: float | None
    norm_high: float | None
    z_score: float | None
    within: bool | None = Field(description="Within mean ± 1 SD (or inside the range)")

    @classmethod
    def from_comparison(cls, c: NormComparison) -> NormComparisonOut:
        """Build from the internal dataclass."""
        return cls(
            feature=c.feature,
            value=c.value,
            units=c.units,
            citation=c.citation,
            paper_id=c.paper_id,
            page=c.page,
            table_no=c.table_no,
            population=c.population,
            norm_text=c.norm_text,
            norm_mean=c.norm_mean,
            norm_sd=c.norm_sd,
            norm_low=c.norm_low,
            norm_high=c.norm_high,
            z_score=c.z_score,
            within=c.within,
        )
