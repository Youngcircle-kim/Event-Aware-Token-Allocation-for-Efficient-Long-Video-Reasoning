# scripts/analysis/compare_by_category.py
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--baseline-name", type=str, default="Baseline")
    parser.add_argument("--method-name", type=str, default="Method")
    parser.add_argument(
        "--category-key",
        type=str,
        default="task_type_original",
        choices=["domain", "sub_category", "task_type_original"],
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path}:{line_no}") from exc
    return rows


def index_by_sample_id(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        result[sample_id] = row
    return result


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def is_correct(row: dict[str, Any]) -> bool:
    if "correct" in row:
        return bool(row["correct"])

    prediction = normalize_label(row.get("prediction"))
    answer = normalize_label(row.get("answer"))

    return (
        prediction is not None
        and answer is not None
        and prediction == answer
    )


def get_category(
    annotation: dict[str, Any],
    category_key: str,
) -> str:
    metadata = annotation.get("metadata", {})
    if not isinstance(metadata, dict):
        return "unknown"

    value = metadata.get(category_key)
    return "unknown" if value is None else str(value)


def main() -> None:
    args = parse_args()

    annotation_index = index_by_sample_id(
        read_jsonl(args.annotation)
    )
    baseline_index = index_by_sample_id(
        read_jsonl(args.baseline)
    )
    method_index = index_by_sample_id(
        read_jsonl(args.method)
    )

    common_ids = sorted(
        set(annotation_index)
        & set(baseline_index)
        & set(method_index)
    )

    if not common_ids:
        raise RuntimeError("No common sample IDs found.")

    stats = defaultdict(
        lambda: {
            "total": 0,
            "baseline_correct": 0,
            "method_correct": 0,
            "both_correct": 0,
            "baseline_only": 0,
            "method_only": 0,
            "both_wrong": 0,
        }
    )

    for sample_id in common_ids:
        annotation = annotation_index[sample_id]
        baseline_row = baseline_index[sample_id]
        method_row = method_index[sample_id]

        category = get_category(
            annotation,
            args.category_key,
        )

        baseline_correct = is_correct(baseline_row)
        method_correct = is_correct(method_row)

        row = stats[category]
        row["total"] += 1
        row["baseline_correct"] += int(baseline_correct)
        row["method_correct"] += int(method_correct)

        if baseline_correct and method_correct:
            row["both_correct"] += 1
        elif baseline_correct:
            row["baseline_only"] += 1
        elif method_correct:
            row["method_only"] += 1
        else:
            row["both_wrong"] += 1

    results = []

    for category, row in stats.items():
        total = row["total"]
        baseline_acc = row["baseline_correct"] / total
        method_acc = row["method_correct"] / total

        results.append(
            {
                "category": category,
                "num_samples": total,
                "baseline_correct": row["baseline_correct"],
                "method_correct": row["method_correct"],
                "baseline_accuracy": baseline_acc,
                "method_accuracy": method_acc,
                "delta": method_acc - baseline_acc,
                "both_correct": row["both_correct"],
                "baseline_only_correct": row["baseline_only"],
                "method_only_correct": row["method_only"],
                "both_wrong": row["both_wrong"],
            }
        )

    results.sort(
        key=lambda x: (
            -x["num_samples"],
            x["category"],
        )
    )

    print()
    print(
        f"===== Category Analysis: {args.category_key} ====="
    )
    print(
        f"Baseline: {args.baseline_name} | "
        f"Method: {args.method_name}"
    )
    print()

    header = (
        f"{'Category':35} "
        f"{'N':>5} "
        f"{'Base':>8} "
        f"{'Method':>8} "
        f"{'Delta':>8} "
        f"{'BaseOnly':>9} "
        f"{'MethodOnly':>11}"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row['category'][:35]:35} "
            f"{row['num_samples']:5d} "
            f"{row['baseline_accuracy']:8.4f} "
            f"{row['method_accuracy']:8.4f} "
            f"{row['delta']:+8.4f} "
            f"{row['baseline_only_correct']:9d} "
            f"{row['method_only_correct']:11d}"
        )

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with args.output.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "category_key": args.category_key,
                    "baseline_name": args.baseline_name,
                    "method_name": args.method_name,
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()