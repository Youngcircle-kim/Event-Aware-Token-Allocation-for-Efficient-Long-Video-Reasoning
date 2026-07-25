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
class RelevanceScorer(BaseEventScorer):
    """
    Event-question relevance based on cosine similarity.

    Each event is represented by mean pooling its frame features.
    """

    normalize: bool = True

    def score(
        self,
        *,
        events: list[Event],
        features: torch.Tensor,
        question_feature: torch.Tensor | None = None,
    ) -> list[EventScore]:
        if question_feature is None:
            raise ValueError(
                "RelevanceScorer requires question_feature."
            )

        if not events:
            return []

        normalized_features = F.normalize(
            features.float(),
            dim=-1,
        )

        normalized_question = F.normalize(
            question_feature.float(),
            dim=0,
        )

        raw_scores: list[float] = []

        for event in events:
            event_features = normalized_features[
                event.start_idx : event.end_idx + 1
            ]

            if event_features.shape[0] == 0:
                raw_scores.append(0.0)
                continue

            event_representation = F.normalize(
                event_features.mean(dim=0),
                dim=0,
            )

            relevance = torch.dot(
                event_representation,
                normalized_question,
            ).item()

            raw_scores.append(relevance)

        final_scores = (
            minmax_normalize(raw_scores)
            if self.normalize
            else raw_scores
        )

        return [
            EventScore(
                event_id=event.event_id,
                score=final_scores[index],
                components={
                    "raw_relevance": raw_scores[index],
                    "relevance": final_scores[index],
                },
            )
            for index, event in enumerate(events)
        ]