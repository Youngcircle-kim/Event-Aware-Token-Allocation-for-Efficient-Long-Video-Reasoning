# src/evalloc/inference/answer_parser.py

from __future__ import annotations

import re
from dataclasses import dataclass


OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class ParsedAnswer:
    """
    Parsed model answer.

    Attributes:
        answer:
            Normalized answer label such as "A", "B", "C", or None.
        raw_output:
            Original model output.
        matched_pattern:
            Name of the parsing rule that succeeded.
    """

    answer: str | None
    raw_output: str
    matched_pattern: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.answer is not None


class MultipleChoiceAnswerParser:
    """
    Parse an option label from a model-generated response.

    Parsing rules are applied from strict to permissive:
    1. Single-letter output
    2. Explicit expressions such as "Answer: B"
    3. Parenthesized label such as "(B)"
    4. Option expression such as "Option B"
    5. Standalone label occurring in the text
    """

    def parse(
        self,
        text: str,
        *,
        num_options: int,
    ) -> ParsedAnswer:
        if num_options <= 0:
            raise ValueError(
                f"num_options must be positive, got {num_options}."
            )

        if num_options > len(OPTION_LABELS):
            raise ValueError(
                f"num_options exceeds supported range: {num_options}."
            )

        raw_output = text
        normalized = self._normalize_text(text)

        if not normalized:
            return ParsedAnswer(
                answer=None,
                raw_output=raw_output,
                matched_pattern=None,
            )

        valid_labels = OPTION_LABELS[:num_options]
        label_group = re.escape(valid_labels)

        # Case 1: the entire output is only one option label.
        exact_match = re.fullmatch(
            rf"[\(\[\{{]?\s*([{label_group}])\s*[\)\]\}}]?[.!]?",
            normalized,
            flags=re.IGNORECASE,
        )

        if exact_match:
            return ParsedAnswer(
                answer=exact_match.group(1).upper(),
                raw_output=raw_output,
                matched_pattern="exact_label",
            )

        patterns = [
            (
                "explicit_answer",
                rf"\b(?:final\s+answer|answer|correct\s+answer)\s*"
                rf"(?:is|:|-)?\s*[\(\[]?([{label_group}])[\)\]]?\b",
            ),
            (
                "option_expression",
                rf"\b(?:option|choice)\s*[\(\[]?([{label_group}])[\)\]]?\b",
            ),
            (
                "parenthesized_label",
                rf"[\(\[]([{label_group}])[\)\]]",
            ),
            (
                "label_with_period",
                rf"\b([{label_group}])\s*[\.\):]",
            ),
        ]

        for pattern_name, pattern in patterns:
            match = re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )

            if match:
                return ParsedAnswer(
                    answer=match.group(1).upper(),
                    raw_output=raw_output,
                    matched_pattern=pattern_name,
                )

        # Last fallback: accept a standalone valid option label only when
        # exactly one unique candidate appears.
        standalone_labels = re.findall(
            rf"\b([{label_group}])\b",
            normalized,
            flags=re.IGNORECASE,
        )

        unique_labels = {
            label.upper()
            for label in standalone_labels
        }

        if len(unique_labels) == 1:
            return ParsedAnswer(
                answer=next(iter(unique_labels)),
                raw_output=raw_output,
                matched_pattern="single_standalone_label",
            )

        return ParsedAnswer(
            answer=None,
            raw_output=raw_output,
            matched_pattern=None,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().split())


def parse_multiple_choice_answer(
    text: str,
    *,
    num_options: int,
) -> str | None:
    """
    Convenience function returning only the normalized label.
    """
    parser = MultipleChoiceAnswerParser()
    return parser.parse(
        text,
        num_options=num_options,
    ).answer