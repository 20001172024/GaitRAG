import type { AnalyzeResponse, NormComparison } from "../api/types";
import { comparisonsFor, FEATURES } from "../lib/features";
import { CitationChip } from "./CitationChip";

interface Props {
  analysis: AnalyzeResponse;
  onOpenNorm: (comparison: NormComparison) => void;
}

const MAX_NORMS_PER_CARD = 3;

function Verdict({ c }: { c: NormComparison }) {
  // Icon + text, not colour alone.
  if (c.within === null) return <span className="text-slate-600">– not comparable</span>;
  if (c.within) return <span className="font-medium text-teal-800">✓ within 1 SD</span>;
  const above = (c.z_score ?? 0) > 0 || (c.norm_high !== null && c.value > c.norm_high);
  return <span className="font-medium text-amber-800">{above ? "▲ above" : "▼ below"} 1 SD</span>;
}

export function FeatureCards({ analysis, onOpenNorm }: Props) {
  return (
    <section aria-labelledby="features-heading">
      <h2 id="features-heading" className="text-lg font-semibold">
        3. Gait features vs published norms
      </h2>
      <p className="mt-1 text-sm text-slate-600">
        Detected {analysis.features.n_strides_left} left and {analysis.features.n_strides_right}{" "}
        right strides. Comparisons use only norm values approved by a reviewer.
      </p>
      <ul className="mt-3 grid gap-3 sm:grid-cols-2">
        {FEATURES.map((f) => {
          const norms = comparisonsFor(f.key, analysis.norms_comparison);
          return (
            <li key={f.key} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <p className="text-sm font-medium text-slate-700" title={f.help}>
                {f.label}
              </p>
              <p className="mt-1 text-2xl font-semibold tabular-nums text-slate-900">
                {analysis.features[f.key].toFixed(f.digits)}{" "}
                <span className="text-sm font-normal text-slate-500">{f.unit}</span>
              </p>
              {norms.length === 0 ? (
                <p className="mt-2 text-xs text-slate-500">No approved published norm yet.</p>
              ) : (
                <ul className="mt-2 space-y-1.5 text-xs">
                  {norms.slice(0, MAX_NORMS_PER_CARD).map((c, i) => (
                    <li key={i} className="leading-relaxed">
                      <Verdict c={c} />
                      {c.z_score !== null && (
                        <span className="text-slate-600"> (z = {c.z_score.toFixed(2)})</span>
                      )}{" "}
                      <span className="text-slate-600">
                        vs {c.population ?? "published group"}:
                      </span>{" "}
                      <span className="font-mono">{c.norm_text}</span>
                      <CitationChip label={c.citation} onOpen={() => onOpenNorm(c)} />
                    </li>
                  ))}
                  {norms.length > MAX_NORMS_PER_CARD && (
                    <li className="text-slate-500">
                      +{norms.length - MAX_NORMS_PER_CARD} more published groups
                    </li>
                  )}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
