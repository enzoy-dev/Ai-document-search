# Retrieval benchmark

For the user-reported local Gemini smoke test, see the
[manual session record](manual/2026-10-21-local-gemini.md). It is kept separate from
the automated reports and includes the failed clarification case.

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

## Additional holdout and precision

The [separate holdout](holdout/README.md) adds three document styles, 24 questions,
absent facts and ambiguity. Its committed real-model reports preserve the existing
models, threshold and k. New reports also include precision on distinct evidence
pages and per-query retrieval time. Ambiguous cases have their own diagnostic and
are excluded from absent-fact rejection metrics.

## Answer and abstention evaluation

Generation is evaluated separately, using the retrieval results as input. This
command makes real Gemini calls and can incur charges; it requires a configured
key, an available model and an explicit opt-in:

```bash
python -m experiments.evaluate_answers --allow-paid-generation --max-cases 24 --output evaluations/results/answers.json
```

The default dataset is the additional holdout. The evaluator builds an isolated
index and records the model/data identities, per-question answers, expected and
actual status, sources, errors and timings. Missing accepted evidence produces
`not_found` without a provider request. At most one generation request is attempted
per selected case. Errors stay in the denominator rather than silently improving
the scores. No Gemini results are committed until a live run is performed.

Automatic diagnostics:

- Status accuracy across `answered`, `not_found`, and `needs_clarification`.
- Answerable-response, absent-fact abstention and ambiguity clarification rates.
- Valid citation-ID rate among generated answers.
- Whether an answered response cites at least one labeled evidence page.
- Mean generation latency across attempted provider calls, excluding retrieval
  and queries that skip generation, plus per-question retrieval time.

These metrics do not establish semantic answer correctness. A model can cite the
right page while making an unsupported claim. Citation validity and expected-page
hits are structural diagnostics, not an answer-grounding score.

### Human review rubric

For each answer, open the cited PDF pages and record:

1. **Correctness:** correct / partially correct / incorrect.
2. **Support:** every factual claim supported / partially supported / unsupported.
3. **Behavior:** appropriate answer, abstention or clarification / inappropriate.
4. **Notes:** missing facts, misleading wording, incorrect page, or a clarification
   that should name the relevant alternatives.

Review abstentions too: a high refusal rate can hide missed answerable questions.
Keep reviewer judgments separate from the automatic metrics and report their
sample size. The offline tests simulate provider responses to check integration
and scoring; they do not substitute for this live/human evaluation.
