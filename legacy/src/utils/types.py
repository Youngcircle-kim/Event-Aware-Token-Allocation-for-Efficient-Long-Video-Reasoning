"""
Core data types used across the codebase.

All modules consume/produce these standardized objects so that
mock and real implementations are interchangeable.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# =============================================================================
# Dataset-level objects
# =============================================================================

@dataclass
class VideoQAExample:
    """
    A single video QA example, normalized across datasets.
    """
    video_id: str
    video_path: str               # path to video file (or mock id)
    duration: float               # seconds
    question: str
    options: List[str]            # e.g., ["A. Onions", "B. Garlic", ...]
    answer: str                   # ground truth letter, e.g., "A"
    gt_timestamp: Optional[Tuple[float, float]] = None  # seconds (start, end) if available
    task_type: str = "general"    # e.g., "temporal_reasoning", "spatial"
    dataset: str = "unknown"

    def __repr__(self):
        return f"VideoQAExample(id={self.video_id}, dur={self.duration:.1f}s, q='{self.question[:40]}...')"


# =============================================================================
# Frame & embedding objects
# =============================================================================

@dataclass
class FrameSet:
    """
    A set of sampled frames from a video.

    frames: numpy array of shape (N, H, W, 3) in uint8, or mock placeholder
    timestamps: array of shape (N,) in seconds
    frame_indices: original frame indices in the video
    """
    frames: np.ndarray            # (N, H, W, 3) or (N, D) for mock
    timestamps: np.ndarray        # (N,)
    frame_indices: np.ndarray     # (N,)

    def __len__(self):
        return len(self.timestamps)

    @property
    def num_frames(self):
        return len(self.timestamps)


@dataclass
class FrameEmbeddings:
    """CLIP/SigLIP-style frame embeddings."""
    embeddings: np.ndarray        # (N, D)
    timestamps: np.ndarray        # (N,)
    frame_indices: np.ndarray     # (N,)

    def __len__(self):
        return len(self.timestamps)


# =============================================================================
# Event structure
# =============================================================================

@dataclass
class Event:
    """
    A semantically coherent segment of a video.

    start_idx, end_idx: indices into the Stage-1 sparse frame sequence.
    start_time, end_time: in seconds.
    """
    event_id: int
    start_idx: int
    end_idx: int                  # exclusive
    start_time: float
    end_time: float
    # Filled in by importance estimation:
    complexity: Optional[float] = None
    relevance: Optional[float] = None
    importance: Optional[float] = None
    # Filled in by allocation:
    allocated_frames: Optional[int] = None

    @property
    def num_frames(self):
        return self.end_idx - self.start_idx

    @property
    def duration(self):
        return self.end_time - self.start_time


@dataclass
class EventSet:
    """Collection of events for one video."""
    events: List[Event]
    video_id: str

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def __getitem__(self, idx):
        return self.events[idx]

    @property
    def total_frames(self):
        return sum(e.num_frames for e in self.events)

    @property
    def total_duration(self):
        return sum(e.duration for e in self.events)


# =============================================================================
# Allocation output
# =============================================================================

@dataclass
class AllocationResult:
    """
    Result of token budget allocation.

    allocations: list of int, one per event. Sum should equal T (within rounding).
    selected_indices: list of arrays, one per event, indicating chosen frame indices
                      (into the Stage-1 sparse frame sequence).
    """
    allocations: List[int]
    selected_indices: List[np.ndarray]
    total_budget: int
    events: EventSet

    def __post_init__(self):
        assert len(self.allocations) == len(self.events), \
            f"Allocation length {len(self.allocations)} != num events {len(self.events)}"

    @property
    def actual_total(self):
        return sum(self.allocations)

    @property
    def final_frame_count(self):
        return sum(len(arr) for arr in self.selected_indices)


# =============================================================================
# Prediction result
# =============================================================================

@dataclass
class Prediction:
    """Output of a full QA pipeline on one example."""
    example_id: str
    predicted_answer: str
    gt_answer: str
    is_correct: bool
    # Instrumentation
    num_frames_used: int
    num_events: int
    allocation: Optional[List[int]] = None
    latency_seconds: float = 0.0
    gpu_memory_mb: float = 0.0
    raw_output: str = ""           # raw model output, for debugging
