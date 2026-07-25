from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Event:
    """
    A temporal event composed of consecutive candidate frames.

    Index convention:
        start_idx and end_idx are inclusive indices in the candidate
        feature sequence.

    Example:
        start_idx=3, end_idx=5
        means candidate positions 3, 4, and 5 belong to the event.
    """

    event_id: int

    # Candidate feature-sequence indices
    start_idx: int
    end_idx: int

    # Time boundaries in seconds
    start_time: float
    end_time: float

    # Original video frame indices belonging to this event
    frame_indices: list[int]

    # Semantic-change score at the beginning of the event
    boundary_score: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)

    def is_empty(self) -> bool:
        return self.num_frames == 0

    def contains_time(self, timestamp: float) -> bool:
        return self.start_time <= timestamp <= self.end_time

    def validate(self) -> None:
        if self.event_id < 0:
            raise ValueError(
                f"event_id must be non-negative, got {self.event_id}."
            )

        if self.start_idx < 0:
            raise ValueError(
                f"start_idx must be non-negative, got {self.start_idx}."
            )

        if self.end_idx < self.start_idx:
            raise ValueError(
                f"end_idx must be >= start_idx, got "
                f"{self.start_idx} to {self.end_idx}."
            )

        expected_num_frames = self.end_idx - self.start_idx + 1

        if self.num_frames != expected_num_frames:
            raise ValueError(
                f"Event {self.event_id} index range contains "
                f"{expected_num_frames} positions, but frame_indices "
                f"contains {self.num_frames} frames."
            )

        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time must be >= start_time, got "
                f"{self.start_time} to {self.end_time}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "num_frames": self.num_frames,
            "frame_indices": self.frame_indices,
            "boundary_score": self.boundary_score,
            "metadata": self.metadata,
        }


class BaseSegmenter(ABC):
    """
    Base interface for temporal event segmenters.
    """

    @abstractmethod
    def segment(
        self,
        frame_indices: list[int],
        timestamps: list[float],
        features: torch.Tensor,
    ) -> list[Event]:
        """
        Args:
            frame_indices:
                Original video frame indices of candidate frames.

            timestamps:
                Candidate-frame timestamps in seconds.

            features:
                Candidate-frame features with shape [N, D].

        Returns:
            Temporally ordered list of Events.
        """
        raise NotImplementedError