"""End-to-end latency (p50 / p95) of /analyze, /search and /ask against a running API.

    docker compose up            # or uvicorn app.main:app
    python eval/run_latency_eval.py --base-url http://localhost:8000 --rounds 5

Warm-up calls (model loading) are excluded. /ask is skipped, with a note, when the
server has no LLM key (HTTP 503).
"""

from __future__ import annotations

import argparse
import time

import httpx
from _common import add_question_args, questions_from_args

from app.evaluation.metrics import percentile
from app.evaluation.report import write_report


def timed(client: httpx.Client, method: str, url: str, **kw) -> tuple[float, httpx.Response]:
    """Wall-clock milliseconds for one request."""
    t = time.perf_counter()
    r = client.request(method, url, **kw)
    return (time.perf_counter() - t) * 1000, r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_question_args(parser)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--rounds", type=int, default=5, help="repetitions per request")
    args = parser.parse_args()
    questions, placeholder = questions_from_args(args)

    times: dict[str, list[float]] = {"/analyze": [], "/search": [], "/ask": []}
    notes = []
    with httpx.Client(base_url=args.base_url, timeout=300) as c:
        samples = c.get("/samples").json()
        files = {s["name"]: c.get(s["url"]).content for s in samples}
        # Warm-up: load the classifier, embedding model, reranker and BM25 index.
        c.post("/analyze", files={"file": ("w.csv", next(iter(files.values())), "text/csv")})
        c.post("/search", json={"query": "warm up"})
        ask_ok = c.post("/ask", json={"question": "warm up"}).status_code != 503
        if not ask_ok:
            notes.append("/ask skipped: the server has no LLM API key (HTTP 503).")

        for _ in range(args.rounds):
            for name, content in files.items():
                ms, r = timed(
                    c, "POST", "/analyze", files={"file": (f"{name}.csv", content, "text/csv")}
                )
                r.raise_for_status()
                times["/analyze"].append(ms)
            for q in questions:
                ms, r = timed(c, "POST", "/search", json={"query": q.question})
                r.raise_for_status()
                times["/search"].append(ms)
                if ask_ok:
                    body = {
                        "question": q.question,
                        "features": q.features,
                        "prediction": q.prediction,
                    }
                    ms, r = timed(c, "POST", "/ask", json=body)
                    r.raise_for_status()
                    times["/ask"].append(ms)

    summary = [
        {
            "endpoint": ep,
            "requests": len(v),
            "p50 ms": percentile(v, 50),
            "p95 ms": percentile(v, 95),
        }
        for ep, v in times.items()
        if v
    ]
    for row in summary:
        print(
            f"{row['endpoint']:9} n={row['requests']:3} p50={row['p50 ms']:.0f} ms p95={row['p95 ms']:.0f} ms"
        )
    notes += [
        f"{args.rounds} rounds against {args.base_url}; warm-up calls excluded.",
        "/analyze uses the synthetic sample recordings; /search and /ask use the question set "
        "with default retrieval settings.",
        "Numbers depend on the machine (CPU-only inference); record the hardware with them.",
    ]
    md, _ = write_report("latency", "Latency", summary, [], notes, placeholder)
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
