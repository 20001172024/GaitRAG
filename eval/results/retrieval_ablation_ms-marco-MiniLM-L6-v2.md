# Retrieval ablation

Generated 2026-11-26T15:05:47+00:00

| config | questions | Recall@5 | Recall@10 | MRR | median ms |
|---|---|---|---|---|---|
| 1 vector | 40 | 0.804 | 0.825 | 0.749 | 23.812 |
| 2 bm25 | 40 | 0.821 | 0.883 | 0.749 | 1.550 |
| 3 hybrid | 40 | 0.829 | 0.900 | 0.749 | 25.068 |
| 4 hybrid+rerank | 40 | 0.817 | 0.854 | 0.776 | 860.945 |
| 5 hybrid+rerank+tables | 40 | 0.904 | 0.917 | 0.857 | 870.737 |

- 40 answerable and 10 unanswerable questions; reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`; embedding model: `BAAI/bge-small-en-v1.5`.
- Recall@k: share of a question's gold papers among the top-k retrieved chunks. MRR: 1 / rank of the first chunk from a gold paper.
- Latency is per query on the machine that ran the script.
- Gate calibration (4 hybrid+rerank): MIN_RERANK_SCORE = 1.397 gives abstention precision 0.83, recall 1.00. Fitted on this question set; indicative only.
- Gate calibration (5 hybrid+rerank+tables): MIN_RERANK_SCORE = 1.761 gives abstention precision 0.83, recall 1.00. Fitted on this question set; indicative only.
