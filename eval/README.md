# Evaluation

`questions.jsonl` is written by the project owner (50–100 questions). It is not generated.

```json
{"id": "q001", "question": "...", "gold_paper_ids": ["P012"], "gold_answer": "...", "type": "norm_lookup",
 "features": {"cadence_steps_per_min": 112.0}, "prediction": "normal"}
```

- `type` is one of `norm_lookup`, `method_comparison`, `feature_interpretation` or `unanswerable`.
- `gold_paper_ids` must be paper IDs from `data/papers/manifest.csv`, and must be empty for `unanswerable` questions.
- `features` and `prediction` are optional and simulate a question asked about a recording.

**Provenance of `questions.jsonl`:** the 50 questions were drafted by the AI assistant (Claude) from the papers' own text, at the owner's request. Gold papers were chosen from the text, not from retrieval output, and gold answers quote figures stated in the papers. The set becomes official when the owner approves it by merging the pull request that adds it. Because the system's builder wrote the questions, results on this set may be optimistic; an independently written set would be a stronger test.

`questions.example.jsonl` holds 8 **placeholder** questions used only to test the scripts. Any run on it writes `results/PLACEHOLDER_*` files, which are git-ignored and carry a warning banner. They are never results.

| Script | Measures | Needs |
|---|---|---|
| `run_retrieval_eval.py` | Recall@5, Recall@10, MRR per ablation config, plus calibration of the insufficient-evidence gate | database |
| `run_generation_eval.py` | abstention precision/recall, citation checks, a sheet for manual citation accuracy | database + LLM key |
| `run_latency_eval.py` | p50/p95 of `/analyze`, `/search`, `/ask` | running API |
