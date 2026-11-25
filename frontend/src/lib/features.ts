import type { GaitFeatures, NormComparison } from "../api/types";

export interface FeatureInfo {
  key: keyof GaitFeatures;
  label: string;
  unit: string;
  digits: number;
  help: string;
}

/** Features shown as cards, in display order. */
export const FEATURES: FeatureInfo[] = [
  {
    key: "cadence_steps_per_min",
    label: "Cadence",
    unit: "steps/min",
    digits: 1,
    help: "Steps per minute (both feet).",
  },
  {
    key: "stride_time_s",
    label: "Stride time",
    unit: "s",
    digits: 3,
    help: "Time between two heel strikes of the same foot.",
  },
  {
    key: "stride_time_cv_pct",
    label: "Stride time variability",
    unit: "% CV",
    digits: 2,
    help: "Coefficient of variation of stride time (lower = more regular).",
  },
  {
    key: "stance_pct_left",
    label: "Stance (left)",
    unit: "% of cycle",
    digits: 1,
    help: "Share of the gait cycle the left foot is on the ground.",
  },
  {
    key: "stance_pct_right",
    label: "Stance (right)",
    unit: "% of cycle",
    digits: 1,
    help: "Share of the gait cycle the right foot is on the ground.",
  },
  {
    key: "asymmetry_stance_time_pct",
    label: "Stance-time asymmetry",
    unit: "% SI",
    digits: 1,
    help: "Symmetry index |L−R| / mean(L,R) × 100 (0 = symmetric).",
  },
  {
    key: "asymmetry_peak_pressure_pct",
    label: "Peak-pressure asymmetry",
    unit: "% SI",
    digits: 1,
    help: "Symmetry index of peak foot pressure (0 = symmetric).",
  },
];

export function comparisonsFor(
  key: keyof GaitFeatures,
  comparisons: NormComparison[],
): NormComparison[] {
  return comparisons.filter((c) => c.feature === key);
}
