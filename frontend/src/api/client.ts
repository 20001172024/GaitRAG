import type {
  AnalyzeResponse,
  AskResponse,
  GaitFeatures,
  Paper,
  Sample,
  SpeedClass,
} from "./types";

export const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

/** An HTTP error with the backend's `detail` message. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail))
        detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      // non-JSON error body: keep statusText
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export function analyzeRecording(file: Blob, filename = "recording.csv"): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file, filename);
  return request<AnalyzeResponse>("/analyze", { method: "POST", body: form });
}

export function ask(
  question: string,
  features?: GaitFeatures,
  prediction?: SpeedClass,
): Promise<AskResponse> {
  return request<AskResponse>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, features, prediction }),
  });
}

export const listPapers = (): Promise<Paper[]> => request<Paper[]>("/papers");
export const listSamples = (): Promise<Sample[]> => request<Sample[]>("/samples");

export async function fetchSample(sample: Sample): Promise<File> {
  const res = await fetch(`${API_BASE}${sample.url}`);
  if (!res.ok) throw new ApiError(res.status, "Could not load the sample recording");
  return new File([await res.text()], `${sample.name}.csv`, { type: "text/csv" });
}
