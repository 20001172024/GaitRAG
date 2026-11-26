// Hand-written mirrors of the backend's Pydantic response models (backend/app/api/*.py).

export type SpeedClass = "slow" | "normal" | "fast";

export interface GaitFeatures {
  cadence_steps_per_min: number;
  stride_time_s: number;
  stride_time_cv_pct: number;
  stance_pct_left: number;
  stance_pct_right: number;
  swing_pct_left: number;
  swing_pct_right: number;
  stance_swing_ratio: number;
  asymmetry_stance_time_pct: number;
  asymmetry_peak_pressure_pct: number;
  n_strides_left: number;
  n_strides_right: number;
}

export interface NormComparison {
  feature: keyof GaitFeatures;
  value: number;
  units: string;
  citation: string;
  paper_id: string;
  page: number;
  table_no: string | null;
  population: string | null;
  norm_text: string;
  norm_mean: number | null;
  norm_sd: number | null;
  norm_low: number | null;
  norm_high: number | null;
  z_score: number | null;
  within: boolean | null;
}

export interface AnalyzeResponse {
  prediction: SpeedClass;
  confidence: number;
  probabilities: Record<string, number>;
  features: GaitFeatures;
  norms_comparison: NormComparison[];
  model_info: { trained_on: string; normalization: string; sklearn_version: string };
  disclaimer: string;
}

export interface Citation {
  label: string;
  paper_id: string;
  title: string;
  page: number;
  table: string | null;
  snippet: string;
  source_url: string | null;
}

export interface AskResponse {
  answer: string;
  citations: Citation[];
  insufficient_evidence: boolean;
  reason: string;
  norms_comparison: NormComparison[];
  model: string;
  latency_ms: number;
  timings_ms: Record<string, number>;
  disclaimer: string;
}

export interface Paper {
  paper_id: string;
  title: string;
  authors: string | null;
  year: number | null;
  doi: string | null;
  licence: string;
  source_url: string;
  n_pages: number | null;
  n_chunks: number;
}

export interface Sample {
  name: string;
  url: string;
  synthetic: boolean;
}
