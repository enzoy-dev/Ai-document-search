"""Evaluate real retrieval separately from Gemini; no generation credentials needed."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
from time import perf_counter

from app.config import EMBEDDING_MODEL, EMBEDDING_REVISION, SIMILARITY_THRESHOLD, TOP_K
from app.document_processor import process_document
from app.embedding_service import generate_embeddings, get_model_identity
from app.vector_store import VectorStore


def score_cases(cases, ranked, threshold, top_k):
    if (
        len(cases) != len(ranked)
        or not cases
        or top_k <= 0
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError(
            "Nonempty aligned cases, a positive top_k and a finite nonnegative threshold are required"
        )
    recalls, accepted_recalls, reciprocals, rejections, details = [], [], [], [], []
    precisions, accepted_precisions, ambiguous_hits = [], [], []
    for case, results in zip(cases, ranked):
        expected = {(item["filename"], item["page"]) for item in case["expected"]}
        selected = results[:top_k]
        actual = {(item["source"], item["page"]) for item in selected}
        accepted = {
            (item["source"], item["page"])
            for item in selected
            if item["distance"] <= threshold
        }
        kind = case.get("kind", "answerable" if expected else "unanswerable")
        if expected:
            recalls.append(len(expected & actual) / len(expected))
            accepted_recalls.append(len(expected & accepted) / len(expected))
            reciprocals.append(
                next(
                    (
                        1 / rank
                        for rank, item in enumerate(selected, 1)
                        if (item["source"], item["page"]) in expected
                    ),
                    0,
                )
            )
            precisions.append(len(expected & actual) / len(actual) if actual else 0)
            accepted_precisions.append(
                len(expected & accepted) / len(accepted) if accepted else 0
            )
        elif kind == "ambiguous":
            ambiguous_hits.append(int(bool(accepted)))
        else:
            rejections.append(int(not accepted))
        details.append(
            {
                "question": case["question"],
                "expected": case["expected"],
                "kind": kind,
                "ranked": selected,
                "accepted": [r for r in selected if r["distance"] <= threshold],
            }
        )
    mean = lambda values: sum(values) / len(values) if values else None
    return {
        "cases": len(cases),
        "answerable_cases": len(recalls),
        "unanswerable_cases": len(rejections),
        "ambiguous_cases": len(ambiguous_hits),
        "recall_at_k": mean(recalls),
        "accepted_recall_at_k": mean(accepted_recalls),
        "precision_at_k": mean(precisions),
        "accepted_precision_at_k": mean(accepted_precisions),
        "mrr_at_k": mean(reciprocals),
        "unanswerable_rejection_rate": mean(rejections),
        "ambiguous_evidence_acceptance_rate": mean(ambiguous_hits),
        "results": details,
    }


def evaluate(dataset_path, threshold=SIMILARITY_THRESHOLD, top_k=TOP_K):
    dataset_path = Path(dataset_path)
    data = json.loads(dataset_path.read_text())
    digest = hashlib.sha256(dataset_path.read_bytes())
    chunks, filenames = [], []
    for name in data["documents"]:
        path = (dataset_path.parent / name).resolve()
        if not path.is_relative_to(dataset_path.parent.resolve()):
            raise ValueError("Dataset PDFs must be inside the dataset directory")
        digest.update(path.read_bytes())
        processed = process_document(str(path))["chunks"]
        chunks.extend(processed)
        filenames.extend([path.name] * len(processed))
    if not chunks:
        raise ValueError("Dataset has no extractable text")
    available = {(name, chunk["page"]) for name, chunk in zip(filenames, chunks)}
    for case in data["cases"]:
        if any(
            (item["filename"], item["page"]) not in available
            for item in case["expected"]
        ):
            raise ValueError("Expected evidence is absent from the dataset PDFs")
    vectors = generate_embeddings([chunk["text"] for chunk in chunks])
    with tempfile.TemporaryDirectory(prefix="retrieval-evaluation-") as directory:
        store = VectorStore(dimension=len(vectors[0]), directory=directory)
        for name in dict.fromkeys(filenames):
            positions = [i for i, filename in enumerate(filenames) if filename == name]
            store.add(
                [chunks[i] for i in positions],
                [vectors[i] for i in positions],
                name,
                encoder_identity=get_model_identity(),
            )
        ranked, latencies = [], []
        for case in data["cases"]:
            started = perf_counter()
            ranked.append(
                store.search(
                    generate_embeddings([case["question"]])[0],
                    k=top_k,
                    encoder_identity=get_model_identity(),
                )
            )
            latencies.append((perf_counter() - started) * 1000)
    metrics = score_cases(data["cases"], ranked, threshold, top_k)
    for detail, latency in zip(metrics["results"], latencies):
        detail["retrieval_ms"] = round(latency, 3)
    return {
        "dataset_sha256": digest.hexdigest(),
        "model": EMBEDDING_MODEL,
        "model_revision": EMBEDDING_REVISION,
        "resolved_model_identity": get_model_identity(),
        "dimension": len(vectors[0]),
        "metric": "normalized_squared_l2",
        "threshold": threshold,
        "top_k": top_k,
        "mean_retrieval_ms": round(sum(latencies) / len(latencies), 3),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluations/retrieval.json",
    )
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k <= 0 or not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--top-k must be positive and --threshold finite and nonnegative")
    report = evaluate(args.dataset, args.threshold, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "results"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
