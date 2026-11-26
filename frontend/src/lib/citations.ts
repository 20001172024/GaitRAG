// Split an answer into text and citation segments, e.g. "... [P006, p.5, Table 2]."
// Same label format the backend validates (app/rag/generator.py CITATION).

export type Segment = { kind: "text"; text: string } | { kind: "citation"; label: string };

const CITATION = /\[P\d{3}, p\.\d+(?:, Table [A-Za-z]?\d+|, Table [IVX]+)?\]/g;

export function splitCitations(answer: string): Segment[] {
  const segments: Segment[] = [];
  let last = 0;
  for (const match of answer.matchAll(CITATION)) {
    const start = match.index ?? 0;
    if (start > last) segments.push({ kind: "text", text: answer.slice(last, start) });
    segments.push({ kind: "citation", label: match[0] });
    last = start + match[0].length;
  }
  if (last < answer.length) segments.push({ kind: "text", text: answer.slice(last) });
  return segments;
}
