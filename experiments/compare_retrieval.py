"""Compare lexical/semantic/RRF using a verified archived real-model ranking.

This replay cannot recover unrecorded semantic candidates or fresh latency.
Use evaluate_retrieval --retrieval-mode hybrid for the actual runtime benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
from app.document_processor import process_document
from app.hybrid import lexical_rank, fuse
from experiments.evaluate_retrieval import score_cases


def compare_archived(dataset, archived):
    dataset, archived = Path(dataset), Path(archived)
    data = json.loads(dataset.read_text(encoding="utf-8"))
    original = json.loads(archived.read_text(encoding="utf-8"))
    digest = hashlib.sha256(dataset.read_bytes())
    chunks = []
    for name in data["documents"]:
        path = (dataset.parent / name).resolve()
        if not path.is_relative_to(dataset.parent.resolve()):
            raise ValueError("Dataset PDFs must be inside its directory")
        digest.update(path.read_bytes())
        chunks.extend(
            {**chunk, "source": path.name}
            for chunk in process_document(str(path))["chunks"]
        )
    if digest.hexdigest() != original["dataset_sha256"]:
        raise ValueError("Archived report does not match dataset/PDF hashes")
    cases, recorded = data["cases"], original["results"]
    if len(cases) != len(recorded) or not cases:
        raise ValueError("Archived cases do not match dataset")
    lookup = {(r["source"], r["page"], r["text"]): i for i, r in enumerate(chunks)}
    if len(lookup) != len(chunks):
        raise ValueError("Duplicate chunks need explicit stable IDs before replay")
    rankings = {"semantic": [], "lexical": [], "hybrid_replay": []}
    for case, result in zip(cases, recorded):
        if (
            case["question"] != result["question"]
            or case["expected"] != result["expected"]
        ):
            raise ValueError("Archived questions/labels do not match")
        semantic = []
        for r in result["ranked"]:
            key = (r["source"], r["page"], r["text"])
            if key not in lookup:
                raise ValueError("Recorded chunk is absent from PDF extraction")
            semantic.append({**r, "chunk_id": lookup[key]})
        lexical = [
            {
                **r,
                "distance": None,
                "retrieval_mode": "hybrid",
                "fusion_score": 1 / (60 + rank),
            }
            for rank, r in enumerate(lexical_rank(case["question"], chunks), 1)
        ]
        rankings["semantic"].append(semantic)
        rankings["lexical"].append(lexical)
        rankings["hybrid_replay"].append(fuse(semantic, lexical, original["top_k"]))
    return {
        "implementation_sha256": {
            name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / name).read_bytes()
            ).hexdigest()
            for name in (
                "app/hybrid.py",
                "app/retrieval.py",
                "experiments/compare_retrieval.py",
                "experiments/evaluate_retrieval.py",
            )
        },
        "dataset_sha256": digest.hexdigest(),
        "archived_report_sha256": hashlib.sha256(archived.read_bytes()).hexdigest(),
        "model": original["model"],
        "resolved_model_identity": original["resolved_model_identity"],
        "semantic_candidates_per_query": original["top_k"],
        "top_k": original["top_k"],
        "threshold": original["threshold"],
        "execution": "archived_ranking_replay",
        "limitations": "Only archived top-k semantic candidates; lexical-only distances unknown. No fresh embedding/latency/Gemini run. Runtime hybrid uses a larger candidate pool. Lexical match does not prove answerability.",
        "comparisons": {
            mode: score_cases(cases, ranked, original["threshold"], original["top_k"])
            for mode, ranked in rankings.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--archived-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_archived(args.dataset, args.archived_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for name, row in report["comparisons"].items():
        print(name, {k: v for k, v in row.items() if k != "results"})


if __name__ == "__main__":
    main()
