import type { AnalyzeResponse, AskResponse, NormComparison } from "../api/types";
import { DISCLAIMER } from "../constants";

export const NORM: NormComparison = {
  feature: "cadence_steps_per_min",
  value: 112,
  units: "steps/min",
  citation: "[P003, p.6, Table 4]",
  paper_id: "P003",
  page: 6,
  table_no: "4",
  population: "20–29 | M",
  norm_text: "109.88 (6.46)",
  norm_mean: 109.88,
  norm_sd: 6.46,
  norm_low: null,
  norm_high: null,
  z_score: 0.33,
  within: true,
};

export const ANALYSIS: AnalyzeResponse = {
  prediction: "normal",
  confidence: 0.95,
  probabilities: { slow: 0.01, normal: 0.95, fast: 0.04 },
  features: {
    cadence_steps_per_min: 112,
    stride_time_s: 1.07,
    stride_time_cv_pct: 2.1,
    stance_pct_left: 61,
    stance_pct_right: 60,
    swing_pct_left: 39,
    swing_pct_right: 40,
    stance_swing_ratio: 1.55,
    asymmetry_stance_time_pct: 3.2,
    asymmetry_peak_pressure_pct: 6.4,
    n_strides_left: 23,
    n_strides_right: 24,
  },
  norms_comparison: [NORM],
  model_info: { trained_on: "synthetic", normalization: "trial_id", sklearn_version: "1.9.1" },
  disclaimer: DISCLAIMER,
};

export const ANSWER: AskResponse = {
  answer: "Your cadence is within 1 SD of the published group [P003, p.6, Table 4].",
  citations: [
    {
      label: "[P003, p.6, Table 4]",
      paper_id: "P003",
      title: "Reference values of gait using APDM",
      page: 6,
      table: "4",
      snippet: "20–29 | M: 109.88 (6.46)",
      source_url: "https://europepmc.org/article/PMC/PMC5792878",
    },
  ],
  insufficient_evidence: false,
  reason: "answered",
  norms_comparison: [NORM],
  model: "fake",
  latency_ms: 10,
  timings_ms: {},
  disclaimer: DISCLAIMER,
};
