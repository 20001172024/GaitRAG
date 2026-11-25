import type { AnalyzeResponse } from "../api/types";

const ORDER = ["slow", "normal", "fast"];

export function PredictionCard({ analysis }: { analysis: AnalyzeResponse }) {
  const synthetic = analysis.model_info.trained_on === "synthetic";
  return (
    <section
      aria-labelledby="prediction-heading"
      className="rounded-xl border bg-white p-5 shadow-sm"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="prediction-heading" className="text-lg font-semibold">
          2. Walking speed
        </h2>
        {synthetic && (
          <span className="rounded bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-900">
            Model trained on synthetic data
          </span>
        )}
      </div>
      <p className="mt-2 text-3xl font-bold capitalize text-slate-900">{analysis.prediction}</p>
      <p className="text-sm text-slate-600">Confidence {(analysis.confidence * 100).toFixed(0)}%</p>
      <dl className="mt-3 space-y-1.5">
        {ORDER.filter((c) => c in analysis.probabilities).map((c) => {
          const p = analysis.probabilities[c];
          return (
            <div key={c} className="flex items-center gap-2 text-sm">
              <dt className="w-16 capitalize text-slate-700">{c}</dt>
              <dd className="flex-1">
                <div
                  className="h-2.5 rounded bg-slate-200"
                  role="meter"
                  aria-label={`${c} probability`}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(p * 100)}
                >
                  <div className="h-2.5 rounded bg-teal-700" style={{ width: `${p * 100}%` }} />
                </div>
              </dd>
              <dd className="w-10 text-right tabular-nums text-slate-700">
                {Math.round(p * 100)}%
              </dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}
