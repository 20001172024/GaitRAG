import { useEffect, useRef } from "react";
import type { Paper } from "../api/types";

/** Everything shown for one cited source. */
export interface SourceInfo {
  label: string;
  paper_id: string;
  title: string;
  page: number;
  table: string | null;
  snippet: string;
  source_url: string | null;
}

interface Props {
  source: SourceInfo | null;
  paper?: Paper;
  onClose: () => void;
}

export function SourceDrawer({ source, paper, onClose }: Props) {
  const closeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!source) return;
    closeButton.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [source, onClose]);

  if (!source) return null;
  const url = paper?.source_url ?? source.source_url;
  return (
    <div className="fixed inset-0 z-20 flex justify-end bg-slate-900/30" onClick={onClose}>
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="source-title"
        onClick={(e) => e.stopPropagation()}
        className="h-full w-full max-w-md overflow-y-auto bg-white p-6 shadow-xl"
      >
        <div className="flex items-start justify-between gap-4">
          <p className="font-mono text-sm text-teal-800">{source.label}</p>
          <button
            ref={closeButton}
            type="button"
            onClick={onClose}
            className="rounded px-2 py-1 text-sm text-slate-600 hover:bg-slate-100"
          >
            Close
          </button>
        </div>
        <h2 id="source-title" className="mt-2 text-lg font-semibold leading-snug">
          {source.title}
        </h2>
        {paper?.authors && <p className="mt-1 text-sm text-slate-600">{paper.authors}</p>}
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          {paper?.year && (
            <>
              <dt className="text-slate-500">Year</dt>
              <dd>{paper.year}</dd>
            </>
          )}
          <dt className="text-slate-500">Page</dt>
          <dd>{source.page}</dd>
          {source.table && (
            <>
              <dt className="text-slate-500">Table</dt>
              <dd>{source.table}</dd>
            </>
          )}
          {paper?.licence && (
            <>
              <dt className="text-slate-500">Licence</dt>
              <dd>{paper.licence}</dd>
            </>
          )}
        </dl>
        <h3 className="mt-5 text-sm font-medium text-slate-700">Supporting passage</h3>
        <blockquote className="mt-1 whitespace-pre-line border-l-4 border-teal-700 bg-slate-50 p-3 text-sm text-slate-800">
          {source.snippet}
        </blockquote>
        {url && (
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            className="mt-5 inline-block text-sm font-medium text-teal-800 underline"
          >
            Open the paper (open access)
          </a>
        )}
      </aside>
    </div>
  );
}
