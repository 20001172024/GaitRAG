import { render, screen } from "@testing-library/react";
import { AnswerView } from "../components/AnswerView";
import { FeatureCards } from "../components/FeatureCards";
import { ANALYSIS, ANSWER, NORM } from "./fixtures";

describe("FeatureCards", () => {
  it("shows the verdict with text, not colour only, and the citation", () => {
    render(<FeatureCards analysis={ANALYSIS} onOpenNorm={() => {}} />);
    expect(screen.getByText("✓ within 1 SD")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Open source [P003, p.6, Table 4]" }),
    ).toBeInTheDocument();
  });

  it("says so when no approved norm exists instead of inventing one", () => {
    render(<FeatureCards analysis={{ ...ANALYSIS, norms_comparison: [] }} onOpenNorm={() => {}} />);
    expect(screen.getAllByText("No approved published norm yet.").length).toBeGreaterThan(0);
  });

  it("marks values above the norm", () => {
    const above = { ...NORM, within: false, z_score: 2.1 };
    render(
      <FeatureCards analysis={{ ...ANALYSIS, norms_comparison: [above] }} onOpenNorm={() => {}} />,
    );
    expect(screen.getByText("▲ above 1 SD")).toBeInTheDocument();
  });
});

describe("AnswerView", () => {
  it("renders citations as clickable chips", () => {
    render(<AnswerView state={{ status: "done", response: ANSWER }} onOpenCitation={() => {}} />);
    expect(
      screen.getByRole("button", { name: "Open source [P003, p.6, Table 4]" }),
    ).toBeInTheDocument();
  });

  it("shows insufficient evidence without an answer", () => {
    const r = { ...ANSWER, insufficient_evidence: true, answer: "x", citations: [] };
    render(<AnswerView state={{ status: "done", response: r }} onOpenCitation={() => {}} />);
    expect(screen.getByText(/Insufficient evidence\./)).toBeInTheDocument();
  });

  it("explains a missing LLM key", () => {
    render(
      <AnswerView
        state={{ status: "error", message: "x", unavailable: true }}
        onOpenCitation={() => {}}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("no language-model API key");
  });
});
