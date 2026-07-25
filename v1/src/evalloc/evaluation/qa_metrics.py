# src/evalloc/evaluation/qa_metrics.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


def normalize_choice_label(value: str | None) -> str | None:
    """
    Normalize a multiple-choice label.

    Examples:
        "a" -> "A"
        " B " -> "B"
        None -> None
    """
    if value is None:
        return None

    value = value.strip().upper()

    if len(value) != 1 or not value.isalpha():
        return None

    return value


def is_correct_choice(
    prediction: str | None,
    answer: str | None,
) -> bool:
    normalized_prediction = normalize_choice_label(prediction)
    normalized_answer = normalize_choice_label(answer)

    if normalized_prediction is None or normalized_answer is None:
        return False

    return normalized_prediction == normalized_answer


def compute_accuracy(
    records: Iterable[dict[str, Any]],
) -> float:
    """
    Compute multiple-choice accuracy from result records.

    Each record must contain:
        prediction
        answer
    """
    total = 0
    correct = 0

    for record in records:
        total += 1

        if is_correct_choice(
            record.get("prediction"),
            record.get("answer"),
        ):
            correct += 1

    if total == 0:
        return 0.0

    return correct / total


@dataclass
class AccuracyMeter:
    """
    Incremental accuracy accumulator.

    Useful for displaying running accuracy while an experiment is running.
    """

    total: int = 0
    correct: int = 0
    parse_failures: int = 0
    failed_samples: list[str] = field(default_factory=list)

    def update(
        self,
        *,
        prediction: str | None,
        answer: str | None,
        sample_id: str | None = None,
    ) -> bool:
        self.total += 1

        if prediction is None:
            self.parse_failures += 1

            if sample_id is not None:
                self.failed_samples.append(sample_id)

        correct = is_correct_choice(prediction, answer)

        if correct:
            self.correct += 1

        return correct

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0

        return self.correct / self.total

    @property
    def parse_failure_rate(self) -> float:
        if self.total == 0:
            return 0.0

        return self.parse_failures / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.total,
            "num_correct": self.correct,
            "accuracy": self.accuracy,
            "num_parse_failures": self.parse_failures,
            "parse_failure_rate": self.parse_failure_rate,
            "failed_sample_ids": self.failed_samples,
        }