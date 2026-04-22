"""Unit tests for adaptive token allocation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.utils.types import Event, EventSet
from src.modules.allocation import (
    largest_remainder_round,
    cap_and_redistribute,
    reduce_events_if_needed,
    allocate_frames,
)


# =============================================================================
# Integer rounding
# =============================================================================

def test_largest_remainder_exact_sum():
    """Rounded values must sum exactly to total."""
    real = np.array([10.3, 10.4, 10.3])
    out = largest_remainder_round(real, total=31)
    assert out.sum() == 31


def test_largest_remainder_handles_zero():
    """Zero total is OK."""
    real = np.array([0.0, 0.0, 0.0])
    out = largest_remainder_round(real, total=0)
    assert out.sum() == 0


def test_largest_remainder_deterministic():
    """Larger remainders go first."""
    real = np.array([1.1, 1.9, 1.0])
    out = largest_remainder_round(real, total=4)
    # 1.9 should get +1, so out = [1, 2, 1]
    assert out.tolist() == [1, 2, 1]


# =============================================================================
# Capacity capping
# =============================================================================

def test_cap_no_overflow():
    """If no overflow, nothing changes."""
    alloc = np.array([5, 10, 3])
    cap = np.array([10, 20, 10])
    imp = np.array([0.5, 0.3, 0.2])
    out = cap_and_redistribute(alloc, cap, imp)
    assert out.tolist() == [5, 10, 3]


def test_cap_redistributes_to_high_importance():
    """Overflow goes to highest-importance event with headroom."""
    alloc = np.array([20, 2, 2])         # 20 overflows by 15 (cap=5)
    cap = np.array([5, 100, 100])
    imp = np.array([0.1, 0.8, 0.1])      # event 1 has highest importance
    out = cap_and_redistribute(alloc, cap, imp)
    assert out[0] == 5                   # capped
    assert out[1] > 2                    # got redistribution
    assert out.sum() <= cap.sum()        # respects total capacity


def test_cap_stops_when_no_headroom():
    """If nobody has headroom, overflow is dropped."""
    alloc = np.array([20, 20, 20])
    cap = np.array([5, 5, 5])
    imp = np.array([0.3, 0.3, 0.4])
    out = cap_and_redistribute(alloc, cap, imp)
    # All capped at 5
    assert out.tolist() == [5, 5, 5]


# =============================================================================
# Event reduction
# =============================================================================

def _make_eventset(n_events: int, frames_per_event: int = 10):
    """Helper to build an EventSet with importance=1/n for each event."""
    events = []
    for i in range(n_events):
        e = Event(
            event_id=i,
            start_idx=i * frames_per_event,
            end_idx=(i + 1) * frames_per_event,
            start_time=i * 10.0,
            end_time=(i + 1) * 10.0,
            importance=1.0 / n_events,
        )
        events.append(e)
    return EventSet(events=events, video_id="test")


def test_reduce_events_no_op_when_feasible():
    """If K*n_min <= T, no reduction."""
    events = _make_eventset(n_events=4)
    reduced = reduce_events_if_needed(events, total_budget=100, n_min=2)
    assert len(reduced) == 4


def test_reduce_events_merges_when_infeasible():
    """If K*n_min > T, merge down to fit."""
    events = _make_eventset(n_events=100)  # 100 events, n_min=2 -> needs 200
    reduced = reduce_events_if_needed(events, total_budget=64, n_min=2)
    # After reduction, K*n_min <= T
    assert len(reduced) * 2 <= 64


# =============================================================================
# Full allocation
# =============================================================================

def test_allocate_sums_to_budget():
    """Allocation sums to total budget (or less if capped)."""
    events = _make_eventset(n_events=4, frames_per_event=100)
    # Give different importances
    for i, e in enumerate(events):
        e.importance = [0.4, 0.3, 0.2, 0.1][i]

    _, alloc = allocate_frames(events, total_budget=100, n_min=2)
    assert alloc.sum() <= 100
    assert alloc.sum() >= 100 - 3  # allow small rounding slack


def test_allocate_respects_n_min():
    """Every event gets at least n_min, if capacity allows."""
    events = _make_eventset(n_events=4, frames_per_event=100)
    for i, e in enumerate(events):
        e.importance = [0.97, 0.01, 0.01, 0.01][i]

    _, alloc = allocate_frames(events, total_budget=128, n_min=4)
    assert (alloc >= 4).all()


def test_allocate_respects_capacity():
    """No event gets more than its length."""
    events = _make_eventset(n_events=3, frames_per_event=10)
    # One event is "super important" but only has 10 frames
    events[0].importance = 0.98
    events[1].importance = 0.01
    events[2].importance = 0.01

    _, alloc = allocate_frames(events, total_budget=100, n_min=2)
    for i, e in enumerate(events):
        assert alloc[i] <= e.num_frames


def test_allocate_concentrates_on_high_importance():
    """High-importance event should receive more frames."""
    events = _make_eventset(n_events=3, frames_per_event=100)
    events[0].importance = 0.8
    events[1].importance = 0.15
    events[2].importance = 0.05

    _, alloc = allocate_frames(events, total_budget=60, n_min=2, tau_allocation=0.3)
    assert alloc[0] > alloc[1]
    assert alloc[1] >= alloc[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
