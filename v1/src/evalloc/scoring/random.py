from __future__ import annotations

from dataclasses import dataclass

import torch

from evalloc.scoring.base import (
    BaseEventScorer,
    EventScore,
)
from evalloc.segmentation.base import Event


@dataclass
class RandomScorer(BaseEventScorer):
    """
    Deterministic random event scorer for baseline experiments.
    """

    seed: int = 42

    def score(
        self,
        *,
        events: list[Event],
        features: torch.Tensor,
        question_feature: torch.Tensor | None = None,
    ) -> list[EventScore]:
        del features
        del question_feature

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        values = torch.rand(
            len(events),
            generator=generator,
        ).tolist()

        return [
            EventScore(
                event_id=event.event_id,
                score=values[index],
                components={
                    "random": values[index],
                    "seed": float(self.seed),
                },
            )
            for index, event in enumerate(events)
        ]