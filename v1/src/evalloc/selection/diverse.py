from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from evalloc.features.clip_extractor import VideoFeatures
from evalloc.segmentation.base import Event


@dataclass
class RelevanceDiverseSelector:
    """
    Greedy relevance-aware diverse frame selector.

    score =
        relevance_weight × question relevance
        + diversity_weight × visual diversity
        + temporal_weight × temporal diversity
    """

    relevance_weight: float = 1.0
    diversity_weight: float = 0.3
    temporal_weight: float = 0.1

    def select(
        self,
        *,
        event: Event,
        video_features: VideoFeatures,
        question_feature: torch.Tensor,
        budget: int,
    ) -> list[int]:
        if budget <= 0:
            return []

        event_features = video_features.features[
            event.start_idx : event.end_idx + 1
        ].float()

        event_features = F.normalize(
            event_features,
            dim=-1,
        )

        question_feature = F.normalize(
            question_feature.float(),
            dim=0,
        )

        event_frame_indices = video_features.frame_indices[
            event.start_idx : event.end_idx + 1
        ]

        event_timestamps = video_features.timestamps[
            event.start_idx : event.end_idx + 1
        ]

        budget = min(
            budget,
            len(event_frame_indices),
        )

        if budget == len(event_frame_indices):
            return sorted(event_frame_indices)

        relevance_scores = (
            event_features @ question_feature
        )

        selected_positions: list[int] = []

        first_position = torch.argmax(
            relevance_scores
        ).item()

        selected_positions.append(first_position)

        while len(selected_positions) < budget:
            best_position: int | None = None
            best_score = float("-inf")

            for position in range(len(event_features)):
                if position in selected_positions:
                    continue

                selected_features = event_features[
                    selected_positions
                ]

                max_similarity = torch.max(
                    selected_features
                    @ event_features[position]
                ).item()

                visual_diversity = 1.0 - max_similarity

                temporal_diversity = self._temporal_diversity(
                    position=position,
                    selected_positions=selected_positions,
                    timestamps=event_timestamps,
                )

                score = (
                    self.relevance_weight
                    * relevance_scores[position].item()
                    + self.diversity_weight
                    * visual_diversity
                    + self.temporal_weight
                    * temporal_diversity
                )

                if score > best_score:
                    best_score = score
                    best_position = position

            if best_position is None:
                break

            selected_positions.append(best_position)

        selected_indices = [
            event_frame_indices[position]
            for position in selected_positions
        ]

        return sorted(selected_indices)

    @staticmethod
    def _temporal_diversity(
        *,
        position: int,
        selected_positions: list[int],
        timestamps: list[float],
    ) -> float:
        if not selected_positions:
            return 1.0

        event_duration = (
            max(timestamps) - min(timestamps)
        )

        if event_duration <= 0:
            return 0.0

        minimum_distance = min(
            abs(
                timestamps[position]
                - timestamps[selected_position]
            )
            for selected_position in selected_positions
        )

        return minimum_distance / event_duration