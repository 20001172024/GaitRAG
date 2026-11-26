import { useState } from "react";

interface Props {
  busy: boolean;
  onAsk: (question: string) => void;
}

export function FollowUpBox({ busy, onAsk }: Props) {
  const [question, setQuestion] = useState("");
  const ok = question.trim().length >= 3 && !busy;
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (!ok) return;
        onAsk(question.trim());
        setQuestion("");
      }}
      className="flex gap-2"
    >
      <label htmlFor="followup" className="sr-only">
        Ask a follow-up question
      </label>
      <input
        id="followup"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        maxLength={1000}
        placeholder="Ask a follow-up, e.g. How does stride variability change with age?"
        className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-teal-700 focus:outline-none focus:ring-1 focus:ring-teal-700"
      />
      <button
        type="submit"
        disabled={!ok}
        className="rounded-md bg-teal-800 px-4 py-2 text-sm font-medium text-white hover:bg-teal-700 disabled:opacity-50"
      >
        Ask
      </button>
    </form>
  );
}
