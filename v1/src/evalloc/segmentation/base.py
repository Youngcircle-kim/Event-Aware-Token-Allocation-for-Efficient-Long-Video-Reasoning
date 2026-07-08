from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    event_id: int
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    fram_indices: list[int]
    boundary_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def num_frames(self) -> int:
        return len(self.fram_indices)
    
    def is_empty(self) -> bool:
        return len(self.fram_indices) == 0
    
    def contains_time(self, timestamp: float) -> bool:
        return self.start_time <= timestamp <= self.end_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_frames": self.num_frames,
            "fram_indices": self.fram_indices,
            "boundary_score": self.boundary_score,
            "metadata": self.metadata,
        }
    
class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, frame_indices: list[int], timestamps:list[float], features:Any) -> list[Event]:
        raise NotImplementedError
    