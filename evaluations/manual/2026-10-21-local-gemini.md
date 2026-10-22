# Local Gemini smoke test — user-reported observations

These observations were reported by Enzo in the project conversation on
**21 October 2026 (America/Sao_Paulo)**, using the local Docker demonstration.
They are a small manual smoke test, not a benchmark run by the automated evaluator.
The configuration was corrected to use `gemini-3.5-flash-lite`; a complete runtime
manifest, raw provider responses, request IDs, latencies and model weight hashes
were not captured during this session. The model name is configuration evidence,
not a provider-side attestation. No keys or personal documents are included.

## Reported results

| Mode | Question | Observed result | Assessment |
| --- | --- | --- | --- |
| Retrieval | Qual é o prazo para devolver os livros? | `biblioteca.pdf`, page 1, contained 14 days; `museu.pdf`, page 1, also appeared with a seven-day loan for museum pieces. | Relevant evidence retrieved alongside a related distractor. Retrieval alone does not select an answer. |
| Generation | Qual é o prazo para devolver os livros? | “O prazo de empréstimo e devolução dos livros é de 14 dias [S1].” Source: `biblioteca.pdf`, page 1. | Correct for this example, with the expected reference. |
| Generation | Qual foi o custo total do projeto Aurora? | “Não encontrei essa informação nos documentos indexados.” | Appropriate refusal for an absent fact in this example. |
| Generation | Qual é o prazo de devolução? | Information not found. | Did not demonstrate a clarification question. |
| Retrieval | Qual é o prazo de devolução? | No nearby passage accepted. | The threshold rejected all results. Under the current pipeline, this bypasses Gemini, so the preceding refusal cannot be attributed to Gemini's ambiguity handling. |
| Generation | Qual é o prazo de devolução dos livros da biblioteca? | “O prazo de empréstimo dos livros é de 14 dias [S1].” Source: `biblioteca.pdf`, page 1. | Specifying the subject restored the expected answer. |

Additional interface/PDF/English-query checks were reported as successful in
aggregate, but their individual outputs were not supplied. They are not counted
as independently documented cases here. Some rows repeat a question with a
different mode; this is not six independent accuracy samples.

## What this establishes

The local integration reached Gemini, returned an evidenced answer, and declined
one absent fact. It also exposed a retrieval limitation for a short vague query.
It does not establish general answer accuracy, a calibrated confidence threshold,
or successful ambiguity clarification. The UI currently groups a retrieval miss
and a model refusal under `not_found`.

No threshold/model change is justified by this one vague query. Next, evaluate
additional specific/vague pairs, record accepted passages and whether Gemini was
called, then score clarification separately from missing-fact abstention.
The [24-question evaluator](../README.md#answer-and-abstention-evaluation) and
human rubric remain the reproducible path to broader evaluation. A complete live
generation report is still pending.

## Reproduce after configuring the local demo

1. Select **Gerar resposta** and run the four generation rows above.
2. Save response status, answer, exact cited sources and timings for each row.
3. For a refusal, repeat with **Buscar trechos** to distinguish a retrieval miss
   from an abstention after context retrieval.
4. Check each factual claim in the original PDF, rather than judging citation
   syntax alone. Record failures as well as successes.
