from __future__ import annotations

from dataclasses import dataclass

import torch

from evalloc.scoring.base import (
    BaseEventScorer,
    EventScore,
)
from evalloc.scoring.normalization import (
    minmax_normalize,
)
from evalloc.segmentation.base import Event


@dataclass
class DurationScorer(BaseEventScorer):
    """
    Assign larger scores to longer events.
    """

    normalize: bool = True

    def score(
        self,
        *,
        events: list[Event],
        features: torch.Tensor,
        question_feature: torch.Tensor | None = None,
    ) -> list[EventScore]:
        del features
        del question_feature

        raw_durations = [
            event.duration
            for event in events
        ]

        final_scores = (
            minmax_normalize(raw_durations)
            if self.normalize
            else raw_durations
        )

        return [
            EventScore(
                event_id=event.event_id,
                score=final_scores[index],
                components={
                    "duration": raw_durations[index],
                    "normalized_duration": (
                        final_scores[index]
                    ),
                },
            )
            for index, event in enumerate(events)
        ]