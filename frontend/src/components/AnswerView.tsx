import type { AskResponse } from "../api/types";
import { splitCitations } from "../lib/citations";
import { CitationChip } from "./CitationChip";

export type AnswerState =
  | { status: "loading" }
  | { status: "error"; message: string; unavailable: boolean }
  | { status: "done"; response: AskResponse };

interface Props {
  state: AnswerState;
  onOpenCitation: (label: string, response: AskResponse) => void;
}

export function AnswerView({ state, onOpenCitation }: Props) {
  if (state.status === "loading") {
    return (
      <p className="text-sm text-slate-500" role="status">
        Searching the papers and writing an answer…
      </p>
    );
  }
  if (state.status === "error") {
    return (
      <p className="rounded-md bg-slate-100 p-3 text-sm text-slate-700" role="alert">
        {state.unavailable
          ? "Explanations are unavailable: the server has no language-model API key configured. The analysis above still works."
          : `Could not get an answer: ${state.message}`}
      </p>
    );
  }
  const r = state.response;
  if (r.insufficient_evidence) {
    return (
      <p className="rounded-md border border-slate-300 bg-slate-50 p-3 text-sm text-slate-700">
        <strong>Insufficient evidence.</strong> The indexed papers do not support an answer to this
        question, so none is given.
      </p>
    );
  }
  return (
    <p className="leading-relaxed text-slate-800">
      {splitCitations(r.answer).map((s, i) =>
        s.kind === "text" ? (
          <span key={i}>{s.text}</span>
        ) : (
          <CitationChip key={i} label={s.label} onOpen={(l) => onOpenCitation(l, r)} />
        ),
      )}
    </p>
  );
}
