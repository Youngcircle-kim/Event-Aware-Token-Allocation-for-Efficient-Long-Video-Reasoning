"""
Module 1: Event Segmentation.

Partitions a long video into semantically coherent events using
CLIP-based semantic change detection.

Pipeline:
  frame embeddings -> adjacent cosine similarity -> sliding-window average
    -> threshold -> boundaries -> events

All logic here is pure numpy; no GPU required.
"""
from typing import List, Optional
import numpy as np

from src.utils.types import Event, EventSet


def compute_adjacent_cosine_sims(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between every pair of adjacent frame embeddings.

    Args:
        embeddings: (N, D), assumed L2-normalized
    Returns:
        (N-1,) similarities
    """
    if len(embeddings) < 2:
        return np.array([])
    return np.sum(embeddings[:-1] * embeddings[1:], axis=-1)


def sliding_window_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth a 1-D signal with a centered moving average.

    Edges are handled with replicate-padding, so output length == input length.

    Args:
        x: (M,)
        window: odd window size preferred; even also works
    Returns:
        (M,) smoothed signal
    """
    if len(x) == 0:
        return x
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')


def detect_boundaries(
    smoothed_sims: np.ndarray,
    strategy: str = "percentile",
    threshold: Optional[float] = None,
    percentile: float = 10.0,
) -> List[int]:
    """
    Identify event boundary indices from smoothed similarity signal.

    A boundary at index i means: "the transition between frame i and frame i+1
    is an event boundary."

    Args:
        smoothed_sims: (M,) smoothed adjacency similarities
        strategy: "absolute" (use fixed threshold) or "percentile" (adaptive)
        threshold: used if strategy == "absolute"
        percentile: used if strategy == "percentile"; points below the
                    p-th percentile become boundaries
    Returns:
        list of boundary indices (sorted ascending, may be empty)
    """
    if len(smoothed_sims) == 0:
        return []

    if strategy == "absolute":
        if threshold is None:
            raise ValueError("`threshold` required for absolute strategy")
        mask = smoothed_sims < threshold
    elif strategy == "percentile":
        cutoff = np.percentile(smoothed_sims, percentile)
        mask = smoothed_sims < cutoff
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    boundary_indices = np.where(mask)[0].tolist()
    return boundary_indices


def build_events_from_boundaries(
    boundaries: List[int],
    num_frames: int,
    timestamps: np.ndarray,
    min_event_frames: int = 3,
) -> List[Event]:
    """
    Convert boundary indices into Event objects.

    A boundary at index i means event_k ends at frame i (exclusive)
    and event_{k+1} starts at frame i+1.

    Args:
        boundaries: indices where a boundary occurs (into adjacency array)
        num_frames: total number of sparse frames
        timestamps: (num_frames,) timestamps in seconds
        min_event_frames: discard events shorter than this (merge with neighbor)
    """
    # Convert adjacency boundary to frame-space boundary
    # adjacency index i is between frame i and i+1
    # so event boundary in frame space = i+1
    frame_boundaries = sorted(set(b + 1 for b in boundaries))
    # Filter duplicates and out-of-range
    frame_boundaries = [b for b in frame_boundaries if 0 < b < num_frames]

    # Build raw events: [0, b_1), [b_1, b_2), ..., [b_K, num_frames)
    start_indices = [0] + frame_boundaries
    end_indices = frame_boundaries + [num_frames]

    raw_events = []
    for k, (s, e) in enumerate(zip(start_indices, end_indices)):
        if e <= s:
            continue
        raw_events.append(Event(
            event_id=k,
            start_idx=s,
            end_idx=e,
            start_time=float(timestamps[s]),
            end_time=float(timestamps[min(e, num_frames - 1)]),
        ))

    # Merge too-short events into the neighbor with higher contrast
    merged = _merge_short_events(raw_events, min_event_frames)

    # Reassign event_ids
    for i, ev in enumerate(merged):
        ev.event_id = i

    return merged


def _merge_short_events(events: List[Event], min_frames: int) -> List[Event]:
    """Greedy merging: merge short events into their temporal neighbor."""
    if not events:
        return events
    changed = True
    while changed:
        changed = False
        for i, ev in enumerate(events):
            if ev.num_frames >= min_frames:
                continue
            # Merge into neighbor
            if i == 0 and len(events) > 1:
                # merge with right neighbor
                events[1].start_idx = ev.start_idx
                events[1].start_time = ev.start_time
                events.pop(0)
            elif i == len(events) - 1 and len(events) > 1:
                # merge with left neighbor
                events[-2].end_idx = ev.end_idx
                events[-2].end_time = ev.end_time
                events.pop(-1)
            elif 0 < i < len(events) - 1:
                # merge with whichever neighbor is smaller (ties broken left)
                left, right = events[i - 1], events[i + 1]
                if left.num_frames <= right.num_frames:
                    left.end_idx = ev.end_idx
                    left.end_time = ev.end_time
                else:
                    right.start_idx = ev.start_idx
                    right.start_time = ev.start_time
                events.pop(i)
            else:
                # only one event, can't merge
                break
            changed = True
            break
    return events


# =============================================================================
# Main entry point
# =============================================================================

def segment_events(
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    video_id: str,
    window: int = 5,
    strategy: str = "percentile",
    percentile: float = 10.0,
    threshold: Optional[float] = None,
    min_event_frames: int = 3,
    max_num_events: Optional[int] = 64,
) -> EventSet:
    """
    Full event segmentation pipeline.

    Args:
        embeddings: (N, D) frame embeddings (L2-normalized)
        timestamps: (N,) timestamps in seconds
        video_id: identifier
        window: sliding-window size for smoothing
        strategy: boundary detection strategy ("percentile" or "absolute")
        percentile: percentile cutoff if strategy == "percentile"
        threshold: threshold value if strategy == "absolute"
        min_event_frames: minimum frames per event
        max_num_events: if too many events, merge the lowest-contrast neighbors
                        until we are under the cap. None = no cap.
    """
    sims = compute_adjacent_cosine_sims(embeddings)
    smoothed = sliding_window_average(sims, window=window)
    boundaries = detect_boundaries(
        smoothed, strategy=strategy, threshold=threshold, percentile=percentile
    )
    events = build_events_from_boundaries(
        boundaries, len(embeddings), timestamps, min_event_frames=min_event_frames
    )

    if max_num_events is not None and len(events) > max_num_events:
        events = _cap_num_events(events, smoothed, max_num_events)

    return EventSet(events=events, video_id=video_id)


def _cap_num_events(
    events: List[Event],
    smoothed_sims: np.ndarray,
    max_events: int,
) -> List[Event]:
    """
    Reduce event count by merging adjacent events at the weakest boundary.
    Weakest = highest similarity at the boundary (= least semantic change).
    """
    while len(events) > max_events:
        # Boundary at position i is between events[i] and events[i+1]
        # Its strength is smoothed_sims[events[i].end_idx - 1]
        boundary_strengths = []
        for i in range(len(events) - 1):
            boundary_idx = events[i].end_idx - 1
            if 0 <= boundary_idx < len(smoothed_sims):
                boundary_strengths.append(smoothed_sims[boundary_idx])
            else:
                boundary_strengths.append(1.0)
        # Merge at max (weakest boundary)
        weakest = int(np.argmax(boundary_strengths))
        # Merge events[weakest+1] into events[weakest]
        events[weakest].end_idx = events[weakest + 1].end_idx
        events[weakest].end_time = events[weakest + 1].end_time
        events.pop(weakest + 1)

    # Reassign ids
    for i, ev in enumerate(events):
        ev.event_id = i
    return events
