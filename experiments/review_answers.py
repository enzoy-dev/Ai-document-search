"""Human factual review bound to exact report content; blank reviews are pending."""

import argparse
import hashlib
import json
from pathlib import Path

CHOICES = {
    "correctness": {
        "correct",
        "partial",
        "incorrect",
        "not_applicable",
        "not_reviewed",
    },
    "support": {
        "supported",
        "partial",
        "unsupported",
        "not_applicable",
        "not_reviewed",
    },
    "behavior": {"appropriate", "inappropriate", "not_reviewed"},
}


def report_hash(report):
    return hashlib.sha256(
        json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def review_template(report):
    return {
        "report_sha256": report_hash(report),
        "reviewer": "",
        "reviews": [
            {
                "case_index": i,
                "question": r["question"],
                "correctness": "not_reviewed",
                "support": "not_reviewed",
                "behavior": "not_reviewed",
                "notes": "",
            }
            for i, r in enumerate(report["results"])
        ],
    }


def score_reviews(report, review):
    if review.get("report_sha256") != report_hash(report):
        raise ValueError("Reviews belong to a different answer report")
    items = review.get("reviews", [])
    if not isinstance(items, list) or any(
        not isinstance(r, dict) or type(r.get("case_index")) is not int for r in items
    ):
        raise ValueError("Reviews require integer case indexes")
    indexes = [r["case_index"] for r in items]
    if len(indexes) != len(set(indexes)) or set(indexes) != set(
        range(len(report["results"]))
    ):
        raise ValueError("One unique review per reported case is required")
    completed, answers = [], []
    for item in items:
        row = report["results"][item["case_index"]]
        if item.get("question") != row["question"]:
            raise ValueError("Review question differs from report")
        if any(item.get(key) not in choices for key, choices in CHOICES.items()):
            raise ValueError("Unknown human review label")
        if any(item[key] == "not_reviewed" for key in CHOICES):
            continue
        if row.get("status") == "answered":
            if (
                item["correctness"] == "not_applicable"
                or item["support"] == "not_applicable"
            ):
                raise ValueError("Answered cases require factual and support review")
            answers.append(item)
        completed.append(item)
    mean = lambda values: sum(values) / len(values) if values else None
    return {
        "reviewer": review.get("reviewer", ""),
        "cases": len(items),
        "reviewed_cases": len(completed),
        "pending_cases": len(items) - len(completed),
        "reviewed_answered_cases": len(answers),
        "factual_correctness_on_reviewed_answers": mean(
            [r["correctness"] == "correct" for r in answers]
        ),
        "fully_supported_reviewed_answers": mean(
            [r["support"] == "supported" for r in answers]
        ),
        "appropriate_behavior_on_reviewed_cases": mean(
            [r["behavior"] == "appropriate" for r in completed]
        ),
        "note": "Human judgments, not automatic proof or population accuracy. Report reviewer and sample size.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output = (
        score_reviews(report, json.loads(args.reviews.read_text(encoding="utf-8")))
        if args.reviews
        else review_template(report)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Choose a new output to preserve existing human review")
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
