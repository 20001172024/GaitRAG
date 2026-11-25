import { useRef, useState } from "react";
import type { Sample } from "../api/types";

interface Props {
  samples: Sample[];
  busy: boolean;
  onFile: (file: Blob, name: string) => void;
  onSample: (sample: Sample) => void;
}

const SAMPLE_LABEL: Record<string, string> = {
  synthetic_slow: "Slow walk",
  synthetic_normal: "Normal walk",
  synthetic_fast: "Fast walk",
};

export function UploadPanel({ samples, busy, onFile, onSample }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const pick = (files: FileList | null) => {
    const file = files?.[0];
    if (file) onFile(file, file.name);
  };

  return (
    <section aria-labelledby="upload-heading" className="rounded-xl border bg-white p-5 shadow-sm">
      <h2 id="upload-heading" className="text-lg font-semibold">
        1. Load a walking recording
      </h2>
      <p className="mt-1 text-sm text-slate-600">
        Insole CSV at 40 Hz (pressure + 4 IMUs). Files are processed in memory and not stored.
      </p>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pick(e.dataTransfer.files);
        }}
        className={`mt-4 flex flex-col items-center gap-2 rounded-lg border-2 border-dashed p-6 text-center ${
          dragging ? "border-teal-600 bg-teal-50" : "border-slate-300"
        }`}
      >
        <p className="text-sm text-slate-700">Drag a CSV file here, or</p>
        <button
          type="button"
          disabled={busy}
          onClick={() => input.current?.click()}
          className="rounded-md bg-slate-800 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
        >
          Choose file
        </button>
        <input
          ref={input}
          type="file"
          accept=".csv,text/csv"
          className="sr-only"
          aria-label="Upload recording CSV"
          onChange={(e) => pick(e.target.files)}
        />
      </div>

      {samples.length > 0 && (
        <div className="mt-4">
          <p className="text-sm font-medium text-slate-700">
            Or try a sample recording{" "}
            <span className="rounded bg-amber-100 px-1.5 py-0.5 text-xs text-amber-900">
              synthetic data
            </span>
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {samples.map((s) => (
              <button
                key={s.name}
                type="button"
                disabled={busy}
                onClick={() => onSample(s)}
                className="rounded-md border border-teal-700 px-3 py-1.5 text-sm font-medium text-teal-800 hover:bg-teal-50 disabled:opacity-50"
              >
                {SAMPLE_LABEL[s.name] ?? s.name}
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
