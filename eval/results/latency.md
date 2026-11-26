# Latency

Generated 2026-11-26T15:21:33+00:00

| endpoint | requests | p50 ms | p95 ms |
|---|---|---|---|
| /analyze | 9 | 104.112 | 122.805 |
| /search | 150 | 953.656 | 1225.159 |

- /ask skipped: the server has no LLM API key (HTTP 503).
- 3 rounds against http://localhost:8000; warm-up calls excluded.
- /analyze uses the synthetic sample recordings; /search and /ask use the question set with default retrieval settings.
- Numbers depend on the machine (CPU-only inference); record the hardware with them.
