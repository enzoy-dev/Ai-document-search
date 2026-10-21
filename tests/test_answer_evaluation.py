import hashlib
import json
from pathlib import Path

import pytest

from app.document_processor import process_document
from deployment.prepare_space import prepare
from experiments.evaluate_answers import score_answers
from experiments.evaluate_retrieval import score_cases


def test_status_and_citation_diagnostics_do_not_hide_answer_failures():
    rows = [
        {
            "expected_status": "answered",
            "status": "answered",
            "citation_ids_valid": True,
            "expected_page_hit": False,
            "generation_ms": 10,
        },
        {
            "expected_status": "not_found",
            "status": "answered",
            "citation_ids_valid": False,
        },
        {"expected_status": "needs_clarification", "status": "needs_clarification"},
        {"expected_status": "answered", "status": "error"},
    ]
    metrics = score_answers(rows)
    assert metrics["status_accuracy"] == 0.5
    assert metrics["answerable_response_rate"] == 0.5
    assert metrics["unanswerable_abstention_rate"] == 0
    assert metrics["ambiguity_clarification_rate"] == 1
    assert metrics["citation_id_validity_on_answered"] == 0.5
    assert metrics["expected_page_hit_on_answered"] == 0
    assert metrics["errors"] == 1


def test_ambiguous_questions_are_not_scored_as_unanswerable_retrieval():
    cases = [
        {"question": "Qual limite?", "kind": "ambiguous", "expected": []},
        {"question": "Missing fact", "kind": "unanswerable", "expected": []},
    ]
    ranked = [[{"source": "a.pdf", "page": 1, "distance": 0.1}], []]
    result = score_cases(cases, ranked, threshold=1.2, top_k=4)
    assert result["ambiguous_cases"] == result["unanswerable_cases"] == 1
    assert result["unanswerable_rejection_rate"] == 1
    assert result["ambiguous_evidence_acceptance_rate"] == 1


def test_generation_latency_excludes_queries_that_skip_the_provider():
    metrics = score_answers(
        [
            {
                "expected_status": "not_found",
                "status": "not_found",
                "generation_attempted": False,
                "generation_ms": 0,
            },
            {
                "expected_status": "answered",
                "status": "answered",
                "generation_attempted": True,
                "generation_ms": 40,
            },
        ]
    )
    assert metrics["attempted_generations"] == 1
    assert metrics["mean_generation_ms"] == 40


def test_holdout_labels_and_reports_reference_real_pages():
    root = Path(__file__).resolve().parents[1] / "evaluations/holdout"
    path = root / "retrieval.json"
    data = json.loads(path.read_text())
    pages = {
        (Path(name).name, chunk["page"])
        for name in data["documents"]
        for chunk in process_document(str(root / name))["chunks"]
    }
    assert len(data["cases"]) == 24 and len(pages) == 6
    assert {case["kind"] for case in data["cases"]} == {
        "answerable",
        "unanswerable",
        "ambiguous",
    }
    assert len({case["id"] for case in data["cases"]}) == 24
    digest = hashlib.sha256(path.read_bytes())
    for name in data["documents"]:
        digest.update((root / name).read_bytes())
    assert all(
        (item["filename"], item["page"]) in pages
        for case in data["cases"]
        for item in case["expected"]
    )
    reports = list((root / "baselines").glob("*.json"))
    assert {path.name for path in reports} == {"minilm.json", "multilingual.json"}
    for report_path in reports:
        report = json.loads(report_path.read_text())
        assert report["dataset_sha256"] == digest.hexdigest()
        assert len(report["resolved_model_identity"]) == 40
        recalculated = score_cases(
            data["cases"],
            [row["ranked"] for row in report["results"]],
            report["threshold"],
            report["top_k"],
        )
        for name in [
            "recall_at_k",
            "accepted_recall_at_k",
            "precision_at_k",
            "accepted_precision_at_k",
            "mrr_at_k",
            "unanswerable_rejection_rate",
            "ambiguous_evidence_acceptance_rate",
        ]:
            assert report[name] == pytest.approx(recalculated[name])


def test_space_bundle_contains_only_public_runtime_files(tmp_path):
    output = prepare(tmp_path / "space")
    assert (output / "Dockerfile").is_file()
    assert "sdk: docker" in (output / "README.md").read_text()
    assert (output / "app/static/index.html").is_file()
    assert len(list((output / "evaluations/documents").glob("*.pdf"))) == 4
    for name in [".env", ".git", "data", "uploads", "tests", "evaluations/holdout"]:
        assert not (output / name).exists()
    assert not list(output.rglob("*.pyc"))
    with pytest.raises(FileExistsError):
        prepare(output)
