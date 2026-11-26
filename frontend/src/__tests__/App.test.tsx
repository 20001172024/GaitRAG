import { fireEvent, render, screen, within } from "@testing-library/react";
import App from "../App";
import { DISCLAIMER } from "../constants";
import { ANALYSIS, ANSWER } from "./fixtures";

const CSV =
  "timestamp_ms,fsr_l_heel,fsr_r_heel,imu1_ax,imu1_ay,imu1_az\n0,1,2,0,0,9.8\n25,3,4,0,0,9.8\n";

function mockApi(ask: () => Response = () => Response.json(ANSWER)) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/samples"))
        return Response.json([
          { name: "synthetic_normal", url: "/samples/synthetic_normal.csv", synthetic: true },
        ]);
      if (url.endsWith("/samples/synthetic_normal.csv")) return new Response(CSV);
      if (url.endsWith("/papers"))
        return Response.json([
          {
            paper_id: "P003",
            title: "Reference values of gait using APDM",
            authors: "Doe J",
            year: 2018,
            doi: null,
            licence: "CC-BY",
            source_url: "https://europepmc.org/article/PMC/PMC5792878",
            n_pages: 10,
            n_chunks: 5,
          },
        ]);
      if (url.endsWith("/analyze")) return Response.json(ANALYSIS);
      if (url.endsWith("/ask")) return ask();
      return new Response("not found", { status: 404 });
    }),
  );
}

afterEach(() => vi.unstubAllGlobals());

it("always shows the disclaimer", () => {
  mockApi();
  render(<App />);
  expect(screen.getByRole("note", { name: "Disclaimer" })).toHaveTextContent(DISCLAIMER);
});

it("runs the sample flow: prediction, cited answer, source panel", async () => {
  mockApi();
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Normal walk" }));

  expect(await screen.findByText("Model trained on synthetic data")).toBeInTheDocument();
  const answer = await screen.findByText(/Your cadence is within 1 SD/);
  fireEvent.click(within(answer.closest("p")!).getByRole("button", { name: /P003, p.6, Table 4/ }));

  const dialog = await screen.findByRole("dialog");
  expect(dialog).toHaveTextContent("Reference values of gait using APDM");
  expect(dialog).toHaveTextContent("CC-BY");
  expect(dialog).toHaveTextContent("20–29 | M: 109.88 (6.46)");
  fireEvent.keyDown(window, { key: "Escape" });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it("keeps the analysis when explanations are unavailable (no LLM key)", async () => {
  mockApi(() => Response.json({ detail: "No API key configured" }, { status: 503 }));
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Normal walk" }));
  expect(await screen.findByText(/no language-model API key/)).toBeInTheDocument();
  expect(screen.getByText("Cadence")).toBeInTheDocument();
});
