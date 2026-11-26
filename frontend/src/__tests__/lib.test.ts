import { splitCitations } from "../lib/citations";
import { downsampleIndices, parseCsv } from "../lib/csv";

describe("splitCitations", () => {
  it("separates text and citation labels", () => {
    expect(
      splitCitations("Cadence was typical [P006, p.5, Table 2]. Variability rose [P008, p.3]."),
    ).toEqual([
      { kind: "text", text: "Cadence was typical " },
      { kind: "citation", label: "[P006, p.5, Table 2]" },
      { kind: "text", text: ". Variability rose " },
      { kind: "citation", label: "[P008, p.3]" },
      { kind: "text", text: "." },
    ]);
  });

  it("leaves malformed labels as text", () => {
    expect(splitCitations("See [P6, page 5].")).toEqual([
      { kind: "text", text: "See [P6, page 5]." },
    ]);
  });
});

describe("csv helpers", () => {
  it("parses numeric columns", () => {
    const rec = parseCsv("timestamp_ms,fsr_l_heel\n0,1.5\n25,2\n");
    expect(rec.rows).toBe(2);
    expect(rec.data.fsr_l_heel).toEqual([1.5, 2]);
  });

  it("downsamples to at most maxPoints", () => {
    expect(downsampleIndices(10, 4)).toEqual([0, 3, 6, 9]);
    expect(downsampleIndices(3, 10)).toEqual([0, 1, 2]);
  });
});
