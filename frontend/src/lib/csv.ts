// Minimal CSV parsing for plotting. The upload itself is analysed by the backend;
// this only extracts numeric columns to draw the signals.

export interface Recording {
  columns: string[];
  rows: number;
  data: Record<string, number[]>;
}

export function parseCsv(text: string): Recording {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length < 2) return { columns: [], rows: 0, data: {} };
  const columns = lines[0].split(",").map((c) => c.trim());
  const data: Record<string, number[]> = Object.fromEntries(columns.map((c) => [c, []]));
  for (const line of lines.slice(1)) {
    const cells = line.split(",");
    columns.forEach((c, i) => data[c].push(Number(cells[i])));
  }
  return { columns, rows: lines.length - 1, data };
}

/** Keep every k-th index so a series has at most `maxPoints` points. */
export function downsampleIndices(n: number, maxPoints: number): number[] {
  const step = Math.max(1, Math.ceil(n / maxPoints));
  const out: number[] = [];
  for (let i = 0; i < n; i += step) out.push(i);
  return out;
}
