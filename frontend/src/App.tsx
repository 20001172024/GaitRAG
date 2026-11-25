import { useCallback, useEffect, useState } from "react";
import {
  analyzeRecording,
  ApiError,
  ask,
  fetchSample,
  listPapers,
  listSamples,
} from "./api/client";
import type { AnalyzeResponse, AskResponse, NormComparison, Paper, Sample } from "./api/types";
import type { AnswerState } from "./components/AnswerView";
import { AnswerView } from "./components/AnswerView";
import { Disclaimer } from "./components/Disclaimer";
import { FeatureCards } from "./components/FeatureCards";
import { FollowUpBox } from "./components/FollowUpBox";
import { PredictionCard } from "./components/PredictionCard";
import { SignalPlots } from "./components/SignalPlots";
import { SourceDrawer, type SourceInfo } from "./components/SourceDrawer";
import { UploadPanel } from "./components/UploadPanel";
import { parseCsv, type Recording } from "./lib/csv";

export const EXPLAIN_QUESTION =
  "How do this recording's gait features compare with published values for healthy adults?";

interface Exchange {
  question: string;
  state: AnswerState;
}

function answerError(err: unknown): AnswerState {
  const status = err instanceof ApiError ? err.status : 0;
  const message = err instanceof Error ? err.message : String(err);
  return { status: "error", message, unavailable: status === 503 };
}

export default function App() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [papers, setPapers] = useState<Record<string, Paper>>({});
  const [recording, setRecording] = useState<Recording | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [source, setSource] = useState<SourceInfo | null>(null);

  useEffect(() => {
    listSamples()
      .then(setSamples)
      .catch(() => setSamples([]));
    listPapers()
      .then((ps) => setPapers(Object.fromEntries(ps.map((p) => [p.paper_id, p]))))
      .catch(() => setPapers({}));
  }, []);

  const askQuestion = useCallback((question: string, a: AnalyzeResponse | null) => {
    setExchanges((xs) => [...xs, { question, state: { status: "loading" } }]);
    const update = (state: AnswerState) =>
      setExchanges((xs) => xs.map((x, i) => (i === xs.length - 1 ? { ...x, state } : x)));
    ask(question, a?.features, a?.prediction)
      .then((response) => update({ status: "done", response }))
      .catch((err) => update(answerError(err)));
  }, []);

  const analyze = useCallback(
    async (file: Blob, name: string) => {
      setAnalyzing(true);
      setAnalyzeError(null);
      setAnalysis(null);
      setExchanges([]);
      try {
        setRecording(parseCsv(await file.text()));
        const result = await analyzeRecording(file, name);
        setAnalysis(result);
        askQuestion(EXPLAIN_QUESTION, result);
      } catch (err) {
        setRecording(null);
        setAnalyzeError(err instanceof Error ? err.message : String(err));
      } finally {
        setAnalyzing(false);
      }
    },
    [askQuestion],
  );

  const openCitation = (label: string, response: AskResponse) => {
    const c = response.citations.find((x) => x.label === label);
    if (c) setSource({ ...c });
  };

  const openNorm = (c: NormComparison) =>
    setSource({
      label: c.citation,
      paper_id: c.paper_id,
      title: papers[c.paper_id]?.title ?? c.paper_id,
      page: c.page,
      table: c.table_no,
      snippet: `${c.population ?? "Published group"}: ${c.norm_text}`,
      source_url: papers[c.paper_id]?.source_url ?? null,
    });

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <main className="mx-auto max-w-5xl space-y-6 p-4 sm:p-6">
        <header>
          <h1 className="text-3xl font-bold">GaitRAG</h1>
          <p className="text-slate-600">
            Evidence-grounded gait analysis: speed classification, gait features, and explanations
            cited to open-access papers.
          </p>
        </header>
        <Disclaimer />

        <UploadPanel
          samples={samples}
          busy={analyzing}
          onFile={analyze}
          onSample={async (s) => {
            try {
              await analyze(await fetchSample(s), `${s.name}.csv`);
            } catch (err) {
              setAnalyzeError(err instanceof Error ? err.message : String(err));
            }
          }}
        />
        {analyzing && (
          <p role="status" className="text-sm text-slate-600">
            Analysing recording…
          </p>
        )}
        {analyzeError && (
          <p role="alert" className="rounded-md bg-red-50 p-3 text-sm text-red-800">
            Could not analyse this recording: {analyzeError}
          </p>
        )}

        {analysis && recording && (
          <>
            <div className="grid gap-6 lg:grid-cols-[1fr_2fr]">
              <PredictionCard analysis={analysis} />
              <SignalPlots recording={recording} />
            </div>
            <FeatureCards analysis={analysis} onOpenNorm={openNorm} />

            <section
              aria-labelledby="answer-heading"
              className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
            >
              <h2 id="answer-heading" className="text-lg font-semibold">
                4. What the literature says
              </h2>
              <div className="mt-3 space-y-5">
                {exchanges.map((x, i) => (
                  <div key={i}>
                    <p className="text-sm font-medium text-slate-600">{x.question}</p>
                    <div className="mt-1">
                      <AnswerView state={x.state} onOpenCitation={openCitation} />
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-5">
                <FollowUpBox
                  busy={exchanges.some((x) => x.state.status === "loading")}
                  onAsk={(q) => askQuestion(q, analysis)}
                />
              </div>
            </section>
          </>
        )}
      </main>
      <SourceDrawer
        source={source}
        paper={source ? papers[source.paper_id] : undefined}
        onClose={() => setSource(null)}
      />
    </div>
  );
}
