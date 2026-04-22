"""Unit tests for event segmentation module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.modules.segmentation import (
    compute_adjacent_cosine_sims,
    sliding_window_average,
    detect_boundaries,
    build_events_from_boundaries,
    segment_events,
)


def test_cosine_sims_basic():
    """Two identical vectors have cosine sim 1."""
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sims = compute_adjacent_cosine_sims(emb)
    assert sims.shape == (2,)
    assert np.isclose(sims[0], 1.0)  # identical
    assert np.isclose(sims[1], 0.0)  # orthogonal


def test_cosine_sims_empty():
    """Single embedding gives no adjacency."""
    emb = np.array([[1.0, 0.0]])
    sims = compute_adjacent_cosine_sims(emb)
    assert len(sims) == 0


def test_sliding_window_preserves_length():
    """Output length should equal input length (edge-padding)."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sliding_window_average(x, window=3)
    assert len(out) == len(x)


def test_sliding_window_reduces_noise():
    """Uniform input yields uniform output; noise gets attenuated."""
    n = 100
    x = np.ones(n) + np.random.RandomState(0).normal(0, 1, size=n)
    smoothed = sliding_window_average(x, window=11)
    assert smoothed.std() < x.std()


def test_detect_boundaries_percentile():
    """Percentile strategy: bottom 20% become boundaries."""
    sims = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.8, 0.9])
    boundaries = detect_boundaries(sims, strategy="percentile", percentile=30)
    # Bottom 30%: 0.1 and 0.2 indices are 2 and 4
    assert 2 in boundaries
    assert 4 in boundaries


def test_detect_boundaries_absolute():
    """Absolute threshold strategy."""
    sims = np.array([0.9, 0.5, 0.3, 0.8])
    boundaries = detect_boundaries(sims, strategy="absolute", threshold=0.6)
    assert 1 in boundaries
    assert 2 in boundaries
    assert 0 not in boundaries


def test_build_events_basic():
    """Boundaries produce (K+1) events."""
    timestamps = np.arange(10, dtype=float)
    events = build_events_from_boundaries(
        boundaries=[2, 5],  # adjacency boundaries -> frame boundaries at 3 and 6
        num_frames=10,
        timestamps=timestamps,
        min_event_frames=1,
    )
    assert len(events) == 3
    assert events[0].start_idx == 0 and events[0].end_idx == 3
    assert events[1].start_idx == 3 and events[1].end_idx == 6
    assert events[2].start_idx == 6 and events[2].end_idx == 10


def test_build_events_merges_short():
    """Short events get merged into neighbors."""
    timestamps = np.arange(20, dtype=float)
    # Boundaries produce events of sizes: 3, 1, 16
    events = build_events_from_boundaries(
        boundaries=[2, 3],
        num_frames=20,
        timestamps=timestamps,
        min_event_frames=3,
    )
    # 1-frame event should be merged away
    for e in events:
        assert e.num_frames >= 3


def test_segment_events_end_to_end():
    """End-to-end: embeddings with two cluster centers -> 2 events."""
    # Create 20 frames: first 10 at [1,0], last 10 at [0,1], with tiny noise
    rng = np.random.RandomState(42)
    center1 = np.array([1.0, 0.0])
    center2 = np.array([0.0, 1.0])
    emb = np.vstack([
        center1 + rng.normal(0, 0.01, (10, 2)),
        center2 + rng.normal(0, 0.01, (10, 2)),
    ])
    # Normalize
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    timestamps = np.arange(20, dtype=float)

    events = segment_events(
        embeddings=emb,
        timestamps=timestamps,
        video_id="test",
        window=3,
        strategy="percentile",
        percentile=10,
        min_event_frames=3,
    )

    # Should find ~2 events (the transition at index 9->10)
    assert 1 <= len(events) <= 4  # allow some flexibility
    # Union should cover all frames
    total = sum(e.num_frames for e in events)
    assert total == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
