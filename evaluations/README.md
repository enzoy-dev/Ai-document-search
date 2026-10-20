# Retrieval benchmark

Four fictional Portuguese PDFs, eight pages (including related distractor passages), and 20 bilingual questions: 16 with labeled evidence and four intentionally unanswerable. Run from the repository root:

```bash
python -m experiments.evaluate_retrieval --output evaluations/results/baseline.json
```

This command uses the **real embedding model** and actual PDF extraction, builds an isolated SQLite/FAISS index and writes an auditable report without calling Gemini or modifying the application index. Model download/network access or a populated cache is required. No benchmark score is claimed until this command completes.

Metrics:

- `recall_at_k`: mean fraction of expected evidence pages present in the top-k results, before thresholding.
- `accepted_recall_at_k`: same calculation after applying the distance threshold.
- `mrr_at_k`: mean reciprocal rank of the first correct evidence page for answerable questions.
- `unanswerable_rejection_rate`: fraction of deliberately unanswerable questions for which the threshold accepts no chunks. This is a diagnostic of threshold behavior, not a measure of Gemini's ability to abstain: a related passage may be retrieved even when a particular fact is absent.

The report records model identifier, configured revision, actual resolved weight identity, dimension, normalized distance metric, threshold, k, dataset hash, per-question ranking and accepted sources. Pin `EMBEDDING_REVISION` to compare model versions reliably. The API resolves and records the actual remote weight commit and rejects different weights against an existing index. For fully reproducible comparisons, also configure a fixed revision.

Compare several thresholds and model configurations, including a Portuguese/multilingual model if needed, using separate reports. Do not tune and report performance on only these same 20 examples: reserve additional PDFs and questions as a holdout before claiming general quality. This small benchmark does not validate answer grounding or prompt-injection resistance.
