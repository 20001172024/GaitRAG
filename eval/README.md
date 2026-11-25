# Evaluation

`questions.jsonl` is written by the project owner (50–100 questions). It is not generated.

```json
{"id": "q001", "question": "...", "gold_paper_ids": ["P012"], "gold_answer": "...", "type": "norm_lookup",
 "features": {"cadence_steps_per_min": 112.0}, "prediction": "normal"}
```

- `type` is one of `norm_lookup`, `method_comparison`, `feature_interpretation` or `unanswerable`.
- `gold_paper_ids` must be paper IDs from `data/papers/manifest.csv`, and must be empty for `unanswerable` questions.
- `features` and `prediction` are optional and simulate a question asked about a recording.

`questions.example.jsonl` holds 8 **placeholder** questions used only to test the scripts. Any run on it writes `results/PLACEHOLDER_*` files, which are git-ignored and carry a warning banner. They are never results.

| Script | Measures | Needs |
|---|---|---|
| `run_retrieval_eval.py` | Recall@5, Recall@10, MRR per ablation config, plus calibration of the insufficient-evidence gate | database |
| `run_generation_eval.py` | abstention precision/recall, citation checks, a sheet for manual citation accuracy | database + LLM key |
| `run_latency_eval.py` | p50/p95 of `/analyze`, `/search`, `/ask` | running API |
