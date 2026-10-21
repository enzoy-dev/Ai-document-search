# Additional evaluation: policy, runbook and schedule

Three original fictional Portuguese PDFs, six pages and 24 bilingual questions:
16 answerable, four with absent facts, and four deliberately ambiguous. The
documents use different topics/styles from the original 20-question development
set. Regenerate them with `python -m evaluations.build_holdout`.

This set was authored after the original model comparison. The two existing
models were evaluated at the unchanged top-k 4 and distance threshold 1.2. No
model/threshold tuning was performed on this set. It is a small synthetic
diagnostic, not evidence of general performance; future tuning requires another
held-out collection.

```bash
python -m experiments.evaluate_retrieval --dataset evaluations/holdout/retrieval.json --output evaluations/results/holdout.json
```

## Real-model results

| Model | Recall@4 before threshold | Expected pages accepted | MRR@4 | Absent-fact queries rejected | Ambiguous queries accepting any passage |
| --- | ---: | ---: | ---: | ---: | ---: |
| all-MiniLM-L6-v2 | 100% | 13/16 | 0.906 | 1/4 | 2/4 |
| paraphrase-multilingual-MiniLM-L12-v2 | 100% | 16/16 | 1.000 | 0/4 | 3/4 |

See the [full reports](baselines), including actual model commits, data hashes,
per-question rankings and observed retrieval times. Timing includes query
embedding/search after corpus indexing/model loading; it excludes cold startup,
generation and network. These timings describe this machine, not a service SLA.

Precision is computed on distinct `(filename, page)` pairs, matching the page-level
labels. A top-k search can return several related pages when just one is labeled;
precision and recall therefore answer different questions. Labels identify the
intended evidence pages, not every possible useful contextual passage.

**Interpretation:** both models can find the labeled answer pages. The threshold
does not reliably reject questions about absent facts, especially when the
question mentions an existing topic. An ambiguous question should often retrieve
candidate evidence and then request clarification; it is not counted as an
unanswerable rejection case. The ambiguity acceptance metric is diagnostic only.

These reports do not call Gemini or establish whether a response is correct,
grounded, appropriately abstains, or requests clarification. Use the separate
[generation evaluation](../README.md#answer-and-abstention-evaluation) and inspect
its answers manually. No generation score is claimed here.
