"""Opt-in generation diagnostic; citation checks do not establish factual support."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.errors import ServiceUnavailableError
from app.llm import generate_answer_draft
from app.retrieval import evidence_from_results, make_context, cited_sources
from experiments.evaluate_retrieval import evaluate


def score_answers(rows):
    if not rows:
        raise ValueError("At least one evaluated answer is required")
    answered = [row for row in rows if row.get("status") == "answered"]
    expected_answers = [row for row in rows if row["expected_status"] == "answered"]
    absent = [row for row in rows if row["expected_status"] == "not_found"]
    ambiguous = [row for row in rows if row["expected_status"] == "needs_clarification"]
    tracked_answers = [
        row for row in expected_answers if row.get("decision_reason") is not None
    ]
    mean = lambda values: sum(values) / len(values) if values else None
    return {
        "cases": len(rows),
        "errors": sum(row.get("status") == "error" for row in rows),
        "attempted_generations": sum(
            row.get("generation_attempted", False) for row in rows
        ),
        "answerable_cases_with_stage_diagnostic": len(tracked_answers),
        "retrieval_miss_on_answerable_rate": mean(
            [
                row.get("decision_reason") == "no_accepted_passages"
                for row in tracked_answers
            ]
        ),
        "abstention_after_context_count": sum(
            row.get("decision_reason") == "model_abstained" for row in rows
        ),
        "clarification_after_context_count": sum(
            row.get("decision_reason") == "model_requested_clarification"
            for row in rows
        ),
        "status_accuracy": mean(
            [row.get("status") == row["expected_status"] for row in rows]
        ),
        "answerable_response_rate": mean(
            [row.get("status") == "answered" for row in expected_answers]
        ),
        "unanswerable_abstention_rate": mean(
            [row.get("status") == "not_found" for row in absent]
        ),
        "ambiguity_clarification_rate": mean(
            [row.get("status") == "needs_clarification" for row in ambiguous]
        ),
        "citation_id_validity_on_answered": mean(
            [row.get("citation_ids_valid", False) for row in answered]
        ),
        "expected_page_hit_on_answered": mean(
            [row.get("expected_page_hit", False) for row in answered]
        ),
        "mean_generation_ms": mean(
            [row["generation_ms"] for row in rows if row.get("generation_attempted")]
        ),
    }


def evaluate_answers(dataset, max_cases=24, retrieval_mode="semantic"):
    dataset = Path(dataset)
    labels = json.loads(dataset.read_text(encoding="utf-8"))["cases"][:max_cases]
    retrieval = evaluate(dataset, retrieval_mode=retrieval_mode)
    rows = []
    for case, retrieved in zip(labels, retrieval["results"]):
        sources = evidence_from_results(retrieved["ranked"], retrieval["threshold"])
        row = {
            "question": case["question"],
            "expected_status": case.get(
                "expected_status", "answered" if case["expected"] else "not_found"
            ),
            "expected": case["expected"],
            "retrieval_ms": retrieved["retrieval_ms"],
            "generation_attempted": False,
            "retrieved_sources": sources,
        }
        if not sources:
            row.update(
                decision_stage="retrieval",
                decision_reason="no_accepted_passages",
                status="not_found",
                answer="No evidence passed the retrieval threshold.",
                sources=[],
                generation_ms=0,
            )
        else:
            started = perf_counter()
            row["generation_attempted"] = True
            try:
                draft = generate_answer_draft(case["question"], make_context(sources))
                row.update(
                    status=draft.status,
                    answer=draft.answer,
                    decision_stage="generation",
                    decision_reason={
                        "answered": "cited_response",
                        "not_found": "model_abstained",
                        "needs_clarification": "model_requested_clarification",
                    }[draft.status],
                )
                if draft.status == "answered":
                    try:
                        selected = cited_sources(draft.answer, sources)
                        row["citation_ids_valid"] = bool(selected)
                    except ServiceUnavailableError:
                        selected = []
                        row["citation_ids_valid"] = False
                    expected = {(e["filename"], e["page"]) for e in case["expected"]}
                    row["expected_page_hit"] = bool(
                        expected & {(s["filename"], s["page"]) for s in selected}
                    )
                    row["sources"] = selected
                else:
                    row["sources"] = []
            except ServiceUnavailableError as exc:
                row.update(
                    status="error",
                    error=str(exc),
                    decision_stage="generation",
                    decision_reason="provider_or_contract_error",
                )
            row["generation_ms"] = round((perf_counter() - started) * 1000, 3)
        rows.append(row)
    return {
        "dataset_sha256": retrieval["dataset_sha256"],
        "embedding_model": retrieval["model"],
        "generation_model": GEMINI_MODEL,
        "retrieval_mode": retrieval_mode,
        "resolved_model_identity": retrieval["resolved_model_identity"],
        "note": "Status and citation/page diagnostics only. Factual correctness and claim support require human review.",
        **score_answers(rows),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "evaluations/holdout/retrieval.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--retrieval-mode", choices=["semantic", "hybrid"], default="semantic"
    )
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--allow-paid-generation", action="store_true")
    args = parser.parse_args()
    if not args.allow_paid_generation or not GEMINI_API_KEY:
        parser.error(
            "Set GEMINI_API_KEY and pass --allow-paid-generation to make provider requests."
        )
    if not 1 <= args.max_cases <= 100:
        parser.error("--max-cases must be between 1 and 100")
    report = evaluate_answers(args.dataset, args.max_cases, args.retrieval_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "results"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
