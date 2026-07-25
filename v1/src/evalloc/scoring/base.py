from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from evalloc.segmentation.base import Event


@dataclass(frozen=True)
class EventScore:
    """
    Generic score result for one event.

    score:
        Final scalar used by an allocator.

    components:
        Individual values used to construct the final score.

        Examples:
            {"relevance": 0.8}
            {"motion": 0.3, "diversity": 0.6}
            {"relevance": 0.8, "complexity": 0.5}
    """

    event_id: int
    score: float
    components: dict[str, float] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "score": self.score,
            "components": self.components,
        }


class BaseEventScorer(ABC):
    """
    Base interface shared by all event scorers.
    """

    @abstractmethod
    def score(
        self,
        *,
        events: list[Event],
        features: torch.Tensor,
        question_feature: torch.Tensor | None = None,
    ) -> list[EventScore]:
        """
        Args:
            events:
                Temporally segmented events.

            features:
                Candidate-frame features with shape [N, D].

            question_feature:
                Optional question embedding with shape [D].

        Returns:
            One EventScore per event, in the same order as events.
        """
        raise NotImplementedError