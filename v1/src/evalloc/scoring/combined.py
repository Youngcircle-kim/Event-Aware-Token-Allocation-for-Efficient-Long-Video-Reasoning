from __future__ import annotations

from dataclasses import dataclass, field

import torch

from evalloc.scoring.base import (
    BaseEventScorer,
    EventScore,
)
from evalloc.scoring.complexity import (
    ComplexityScorer,
)
from evalloc.scoring.relevance import (
    RelevanceScorer,
)
from evalloc.segmentation.base import Event


@dataclass
class CombinedScorer(BaseEventScorer):
    """
    Weighted combination of question relevance and visual complexity.
    """

    relevance_weight: float = 0.5
    complexity_weight: float = 0.5

    relevance_scorer: RelevanceScorer = field(
        default_factory=RelevanceScorer
    )

    complexity_scorer: ComplexityScorer = field(
        default_factory=ComplexityScorer
    )

    def __post_init__(self) -> None:
        total_weight = (
            self.relevance_weight
            + self.complexity_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "The sum of combined weights must be positive."
            )

        self.relevance_weight /= total_weight
        self.complexity_weight /= total_weight

    def score(
        self,
        *,
        events: list[Event],
        features: torch.Tensor,
        question_feature: torch.Tensor | None = None,
    ) -> list[EventScore]:
        relevance_scores = self.relevance_scorer.score(
            events=events,
            features=features,
            question_feature=question_feature,
        )

        complexity_scores = self.complexity_scorer.score(
            events=events,
            features=features,
            question_feature=None,
        )

        if len(relevance_scores) != len(
            complexity_scores
        ):
            raise RuntimeError(
                "Relevance and complexity scorer results "
                "have different lengths."
            )

        combined_scores: list[EventScore] = []

        for relevance, complexity in zip(
            relevance_scores,
            complexity_scores,
        ):
            if relevance.event_id != complexity.event_id:
                raise RuntimeError(
                    "Event ID mismatch between scorers."
                )

            score = (
                self.relevance_weight
                * relevance.score
                + self.complexity_weight
                * complexity.score
            )

            components = {
                **relevance.components,
                **complexity.components,
                "importance": score,
            }

            combined_scores.append(
                EventScore(
                    event_id=relevance.event_id,
                    score=score,
                    components=components,
                )
            )

        return combined_scores