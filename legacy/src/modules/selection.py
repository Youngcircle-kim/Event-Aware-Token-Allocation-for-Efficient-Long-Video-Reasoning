"""
Module 4: Frame selection within an event.

Once we've decided how many frames each event gets (n_k), we choose
which specific frames from the event to actually use.

Three strategies:
  1. uniform  - linearly spaced frames within the event
  2. topk     - highest-importance frames (requires per-frame scores)
  3. hybrid   - uniform anchors with importance-based refinement in local windows

All return indices into the Stage-1 sparse frame sequence.
"""
from typing import Optional
import numpy as np

from src.utils.types import Event


def uniform_select(event: Event, n_k: int) -> np.ndarray:
    """Linearly spaced selection within [start_idx, end_idx)."""
    L = event.num_frames
    if n_k <= 0:
        return np.array([], dtype=int)
    if n_k >= L:
        return np.arange(event.start_idx, event.end_idx)
    # linspace inside the event
    local = np.linspace(0, L - 1, n_k).round().astype(int)
    return event.start_idx + np.unique(local)


def topk_select(
    event: Event,
    n_k: int,
    frame_scores: np.ndarray,
) -> np.ndarray:
    """
    Select the n_k frames within the event with highest scores.
    Results are returned in temporal (ascending index) order.

    Args:
        event: the event
        n_k: number to select
        frame_scores: (N,) per-frame importance for the full sparse sequence
    """
    L = event.num_frames
    if n_k <= 0:
        return np.array([], dtype=int)
    if n_k >= L:
        return np.arange(event.start_idx, event.end_idx)
    local_scores = frame_scores[event.start_idx:event.end_idx]
    top_local = np.argsort(-local_scores)[:n_k]
    return np.sort(event.start_idx + top_local)


def hybrid_select(
    event: Event,
    n_k: int,
    frame_scores: np.ndarray,
    window_ratio: float = 0.5,
) -> np.ndarray:
    """
    Hybrid: place n_k uniform anchors; replace each with the best-scoring
    frame within a local window.

    Args:
        window_ratio: window half-width relative to the spacing between
                      anchors. 0.5 = windows just touch; smaller = tighter.
    """
    L = event.num_frames
    if n_k <= 0:
        return np.array([], dtype=int)
    if n_k >= L:
        return np.arange(event.start_idx, event.end_idx)

    anchors = np.linspace(0, L - 1, n_k)
    spacing = (L - 1) / max(n_k, 1)
    half_win = int(max(1, round(spacing * window_ratio)))

    local_scores = frame_scores[event.start_idx:event.end_idx]
    selected_local = []
    used = set()
    for a in anchors:
        a = int(round(a))
        start = max(0, a - half_win)
        end = min(L, a + half_win + 1)
        # Pick best frame in window not already used
        candidates = [(i, local_scores[i]) for i in range(start, end) if i not in used]
        if not candidates:
            # fallback to the anchor itself even if already used
            selected_local.append(a)
            used.add(a)
        else:
            best = max(candidates, key=lambda x: x[1])[0]
            selected_local.append(best)
            used.add(best)

    return np.sort(event.start_idx + np.unique(selected_local))


# =============================================================================
# Dispatcher
# =============================================================================

def select_frames(
    event: Event,
    n_k: int,
    strategy: str = "uniform",
    frame_scores: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Select n_k frames from an event using the specified strategy."""
    if strategy == "uniform":
        return uniform_select(event, n_k)
    if strategy == "topk":
        if frame_scores is None:
            raise ValueError("topk strategy requires frame_scores")
        return topk_select(event, n_k, frame_scores)
    if strategy == "hybrid":
        if frame_scores is None:
            raise ValueError("hybrid strategy requires frame_scores")
        return hybrid_select(event, n_k, frame_scores, **kwargs)
    raise ValueError(f"Unknown strategy: {strategy}")
