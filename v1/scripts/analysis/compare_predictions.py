from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two prediction JSONL files using paired outcomes "
            "and McNemar's exact test."
        )
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--method",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="Baseline",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default="Method",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected an object at {path}:{line_number}."
                )

            records.append(record)

    return records


def index_by_sample_id(
    records: list[dict[str, Any]],
    source_name: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}

    for record in records:
        sample_id = record.get("sample_id")

        if sample_id is None:
            raise ValueError(
                f"{source_name} contains a record without sample_id."
            )

        sample_id = str(sample_id)

        if sample_id in indexed:
            raise ValueError(
                f"Duplicate sample_id in {source_name}: {sample_id}"
            )

        indexed[sample_id] = record

    return indexed


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().upper()
    return text or None


def get_correct(record: dict[str, Any]) -> bool:
    if "correct" in record:
        return bool(record["correct"])

    prediction = normalize_label(record.get("prediction"))
    answer = normalize_label(record.get("answer"))

    return (
        prediction is not None
        and answer is not None
        and prediction == answer
    )


def exact_mcnemar_p_value(
    baseline_only: int,
    method_only: int,
) -> float:
    """
    Exact two-sided McNemar test.

    Under H0, discordant pairs follow Binomial(n, 0.5).
    """
    discordant = baseline_only + method_only

    if discordant == 0:
        return 1.0

    smaller = min(
        baseline_only,
        method_only,
    )

    lower_tail = sum(
        math.comb(discordant, k)
        for k in range(smaller + 1)
    ) / (2 ** discordant)

    return min(
        1.0,
        2.0 * lower_tail,
    )


def corrected_mcnemar(
    baseline_only: int,
    method_only: int,
) -> tuple[float, float]:
    discordant = baseline_only + method_only

    if discordant == 0:
        return 0.0, 1.0

    statistic = (
        (abs(baseline_only - method_only) - 1.0) ** 2
        / discordant
    )

    # Chi-square distribution with one degree of freedom
    p_value = math.erfc(
        math.sqrt(statistic / 2.0)
    )

    return statistic, p_value


def main() -> None:
    args = parse_args()

    baseline_records = read_jsonl(args.baseline)
    method_records = read_jsonl(args.method)

    baseline_index = index_by_sample_id(
        baseline_records,
        args.baseline_name,
    )
    method_index = index_by_sample_id(
        method_records,
        args.method_name,
    )

    baseline_ids = set(baseline_index)
    method_ids = set(method_index)
    common_ids = sorted(
        baseline_ids & method_ids
    )

    if not common_ids:
        raise RuntimeError(
            "No common sample IDs were found."
        )

    missing_from_method = sorted(
        baseline_ids - method_ids
    )
    missing_from_baseline = sorted(
        method_ids - baseline_ids
    )

    both_correct = 0
    baseline_only = 0
    method_only = 0
    both_wrong = 0

    changed_samples: list[dict[str, Any]] = []

    for sample_id in common_ids:
        baseline_record = baseline_index[sample_id]
        method_record = method_index[sample_id]

        baseline_correct = get_correct(
            baseline_record
        )
        method_correct = get_correct(
            method_record
        )

        if baseline_correct and method_correct:
            both_correct += 1
        elif baseline_correct:
            baseline_only += 1
        elif method_correct:
            method_only += 1
        else:
            both_wrong += 1

        if baseline_correct != method_correct:
            changed_samples.append(
                {
                    "sample_id": sample_id,
                    "answer": baseline_record.get("answer"),
                    "baseline_prediction": (
                        baseline_record.get("prediction")
                    ),
                    "method_prediction": (
                        method_record.get("prediction")
                    ),
                    "baseline_correct": baseline_correct,
                    "method_correct": method_correct,
                }
            )

    total = len(common_ids)

    baseline_correct_count = (
        both_correct + baseline_only
    )
    method_correct_count = (
        both_correct + method_only
    )

    baseline_accuracy = (
        baseline_correct_count / total
    )
    method_accuracy = (
        method_correct_count / total
    )

    exact_p = exact_mcnemar_p_value(
        baseline_only,
        method_only,
    )

    chi_square, chi_square_p = corrected_mcnemar(
        baseline_only,
        method_only,
    )

    result: dict[str, Any] = {
        "baseline_name": args.baseline_name,
        "method_name": args.method_name,
        "num_common_samples": total,
        "baseline_correct": baseline_correct_count,
        "method_correct": method_correct_count,
        "baseline_accuracy": baseline_accuracy,
        "method_accuracy": method_accuracy,
        "accuracy_delta": (
            method_accuracy - baseline_accuracy
        ),
        "pairwise": {
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only,
            "method_only_correct": method_only,
            "both_wrong": both_wrong,
        },
        "mcnemar": {
            "discordant_pairs": (
                baseline_only + method_only
            ),
            "exact_two_sided_p_value": exact_p,
            "continuity_corrected_chi_square": (
                chi_square
            ),
            "chi_square_p_value": chi_square_p,
        },
        "missing": {
            "missing_from_method": missing_from_method,
            "missing_from_baseline": (
                missing_from_baseline
            ),
        },
        "changed_samples": changed_samples,
    }

    print("\n===== Paired Prediction Comparison =====")
    print(f"Baseline : {args.baseline_name}")
    print(f"Method   : {args.method_name}")
    print(f"Samples  : {total}")

    print(
        f"\n{args.baseline_name} accuracy: "
        f"{baseline_accuracy:.4f} "
        f"({baseline_correct_count}/{total})"
    )
    print(
        f"{args.method_name} accuracy: "
        f"{method_accuracy:.4f} "
        f"({method_correct_count}/{total})"
    )
    print(
        f"Delta: "
        f"{method_accuracy - baseline_accuracy:+.4f}"
    )

    print("\n===== Pairwise Outcomes =====")
    print(f"Both correct          : {both_correct}")
    print(
        f"{args.baseline_name} only correct : "
        f"{baseline_only}"
    )
    print(
        f"{args.method_name} only correct : "
        f"{method_only}"
    )
    print(f"Both wrong            : {both_wrong}")

    print("\n===== McNemar Test =====")
    print(
        f"Discordant pairs      : "
        f"{baseline_only + method_only}"
    )
    print(
        f"Exact two-sided p     : "
        f"{exact_p:.6f}"
    )
    print(
        f"Corrected chi-square  : "
        f"{chi_square:.6f}"
    )
    print(
        f"Chi-square p          : "
        f"{chi_square_p:.6f}"
    )

    if missing_from_method:
        print(
            f"\n[WARN] {len(missing_from_method)} samples "
            "are missing from method predictions."
        )

    if missing_from_baseline:
        print(
            f"\n[WARN] {len(missing_from_baseline)} samples "
            "are missing from baseline predictions."
        )

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()