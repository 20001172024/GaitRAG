# Retrieval ablation

Generated 2026-11-26T15:18:45+00:00

| config | questions | Recall@5 | Recall@10 | MRR | median ms |
|---|---|---|---|---|---|
| 1 vector | 40 | 0.804 | 0.825 | 0.749 | 20.716 |
| 2 bm25 | 40 | 0.821 | 0.883 | 0.749 | 1.932 |
| 3 hybrid | 40 | 0.829 | 0.900 | 0.749 | 22.756 |
| 4 hybrid+rerank | 40 | 0.863 | 0.887 | 0.760 | 6758.184 |
| 5 hybrid+rerank+tables | 40 | 0.896 | 0.917 | 0.838 | 6686.672 |

- 40 answerable and 10 unanswerable questions; reranker: `BAAI/bge-reranker-base`; embedding model: `BAAI/bge-small-en-v1.5`.
- Recall@k: share of a question's gold papers among the top-k retrieved chunks. MRR: 1 / rank of the first chunk from a gold paper.
- Latency is per query on the machine that ran the script.
- Gate calibration (4 hybrid+rerank): MIN_RERANK_SCORE = 0.493 gives abstention precision 0.75, recall 0.90. Fitted on this question set; indicative only.
- Gate calibration (5 hybrid+rerank+tables): MIN_RERANK_SCORE = 0.528 gives abstention precision 0.75, recall 0.90. Fitted on this question set; indicative only.
