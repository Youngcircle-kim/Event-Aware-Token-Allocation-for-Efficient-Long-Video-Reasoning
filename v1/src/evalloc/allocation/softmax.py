from __future__ import annotations

from dataclasses import dataclass

import torch

from evalloc.scoring.base import EventScore
from evalloc.segmentation.base import Event


@dataclass
class SoftmaxAllocator:
    """
    Allocate the total frame budget across events according to
    softmax-normalized event scores.

    The returned allocation never exceeds each event's candidate-frame
    capacity, and the sum equals min(total_budget, total_capacity).
    """

    temperature: float = 0.3

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {self.temperature}."
            )

    def allocate(
        self,
        *,
        events: list[Event],
        scores: list[EventScore],
        total_budget: int,
    ) -> dict[int, int]:
        if total_budget < 0:
            raise ValueError(
                f"total_budget must be non-negative, got {total_budget}."
            )

        if not events:
            return {}

        if len(events) != len(scores):
            raise ValueError(
                "events and scores must have the same length: "
                f"{len(events)} != {len(scores)}"
            )

        event_ids = [event.event_id for event in events]
        score_ids = [score.event_id for score in scores]

        if event_ids != score_ids:
            raise ValueError(
                "events and scores must be ordered by the same event IDs: "
                f"{event_ids} != {score_ids}"
            )

        capacities = torch.tensor(
            [event.num_frames for event in events],
            dtype=torch.long,
        )

        total_capacity = int(capacities.sum().item())
        effective_budget = min(total_budget, total_capacity)

        if effective_budget == 0:
            return {
                event.event_id: 0
                for event in events
            }

        score_tensor = torch.tensor(
            [score.score for score in scores],
            dtype=torch.float32,
        )

        if not torch.isfinite(score_tensor).all():
            raise ValueError(
                f"Non-finite event scores detected: {score_tensor.tolist()}"
            )

        probabilities = torch.softmax(
            score_tensor / self.temperature,
            dim=0,
        )

        ideal_allocation = probabilities * effective_budget

        allocation_tensor = torch.floor(
            ideal_allocation
        ).to(torch.long)

        allocation_tensor = torch.minimum(
            allocation_tensor,
            capacities,
        )

        remaining = (
            effective_budget
            - int(allocation_tensor.sum().item())
        )

        fractional_parts = (
            ideal_allocation
            - torch.floor(ideal_allocation)
        )

        while remaining > 0:
            available_indices = [
                index
                for index in range(len(events))
                if allocation_tensor[index] < capacities[index]
            ]

            if not available_indices:
                break

            best_index = max(
                available_indices,
                key=lambda index: (
                    fractional_parts[index].item(),
                    probabilities[index].item(),
                    -index,
                ),
            )

            allocation_tensor[best_index] += 1
            fractional_parts[best_index] = -1.0
            remaining -= 1

        allocation = {
            event.event_id: int(
                allocation_tensor[index].item()
            )
            for index, event in enumerate(events)
        }

        if sum(allocation.values()) != effective_budget:
            raise RuntimeError(
                "Allocation sum mismatch: "
                f"expected={effective_budget}, "
                f"actual={sum(allocation.values())}"
            )

        for event in events:
            allocated = allocation[event.event_id]

            if allocated > event.num_frames:
                raise RuntimeError(
                    f"Event {event.event_id} received {allocated} frames, "
                    f"but contains only {event.num_frames} candidates."
                )

        return allocation