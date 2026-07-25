from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from evalloc.scoring.base import (
    BaseEventScorer,
    EventScore,
)
from evalloc.scoring.normalization import (
    minmax_normalize,
)
from evalloc.segmentation.base import Event


@dataclass
class ComplexityScorer(BaseEventScorer):
    """
    Visual complexity based on three feature-space signals:

    motion:
        Average semantic change between consecutive candidate frames.

    diversity:
        Average pairwise semantic distance inside the event.

    variance:
        Average squared distance from the event representation.
    """

    motion_weight: float = 1.0 / 3.0
    diversity_weight: float = 1.0 / 3.0
    variance_weight: float = 1.0 / 3.0

    normalize_components: bool = True

    def __post_init__(self) -> None:
        total_weight = (
            self.motion_weight
            + self.diversity_weight
            + self.variance_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "The sum of complexity weights must be positive."
            )

        self.motion_weight /= total_weight
        self.diversity_weight /= total_weight
        self.variance_weight /= total_weight

    def score(
        self,
        *,
        events: list[Event],
        features: torch.Tensor,
        question_feature: torch.Tensor | None = None,
    ) -> list[EventScore]:
        del question_feature

        if not events:
            return []

        normalized_features = F.normalize(
            features.float(),
            dim=-1,
        )

        raw_motion: list[float] = []
        raw_diversity: list[float] = []
        raw_variance: list[float] = []

        for event in events:
            event_features = normalized_features[
                event.start_idx : event.end_idx + 1
            ]

            raw_motion.append(
                self._compute_motion(event_features)
            )

            raw_diversity.append(
                self._compute_diversity(event_features)
            )

            raw_variance.append(
                self._compute_variance(event_features)
            )

        if self.normalize_components:
            motion_values = minmax_normalize(
                raw_motion
            )
            diversity_values = minmax_normalize(
                raw_diversity
            )
            variance_values = minmax_normalize(
                raw_variance
            )
        else:
            motion_values = raw_motion
            diversity_values = raw_diversity
            variance_values = raw_variance

        complexity_values = [
            (
                self.motion_weight
                * motion_values[index]
                + self.diversity_weight
                * diversity_values[index]
                + self.variance_weight
                * variance_values[index]
            )
            for index in range(len(events))
        ]

        return [
            EventScore(
                event_id=event.event_id,
                score=complexity_values[index],
                components={
                    "raw_motion": raw_motion[index],
                    "raw_diversity": (
                        raw_diversity[index]
                    ),
                    "raw_variance": raw_variance[index],
                    "motion": motion_values[index],
                    "diversity": diversity_values[index],
                    "variance": variance_values[index],
                    "complexity": complexity_values[index],
                },
            )
            for index, event in enumerate(events)
        ]

    @staticmethod
    def _compute_motion(
        event_features: torch.Tensor,
    ) -> float:
        if event_features.shape[0] <= 1:
            return 0.0

        adjacent_similarities = (
            event_features[:-1]
            * event_features[1:]
        ).sum(dim=-1)

        adjacent_distances = (
            1.0 - adjacent_similarities
        ).clamp(min=0.0)

        return adjacent_distances.mean().item()

    @staticmethod
    def _compute_diversity(
        event_features: torch.Tensor,
    ) -> float:
        num_frames = event_features.shape[0]

        if num_frames <= 1:
            return 0.0

        similarity_matrix = (
            event_features
            @ event_features.transpose(0, 1)
        )

        upper_triangle_mask = torch.triu(
            torch.ones_like(
                similarity_matrix,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        pairwise_similarities = similarity_matrix[
            upper_triangle_mask
        ]

        pairwise_distances = (
            1.0 - pairwise_similarities
        ).clamp(min=0.0)

        return pairwise_distances.mean().item()

    @staticmethod
    def _compute_variance(
        event_features: torch.Tensor,
    ) -> float:
        if event_features.shape[0] <= 1:
            return 0.0

        event_representation = (
            event_features.mean(dim=0)
        )

        squared_distances = (
            event_features
            - event_representation
        ).pow(2).sum(dim=-1)

        return squared_distances.mean().item()