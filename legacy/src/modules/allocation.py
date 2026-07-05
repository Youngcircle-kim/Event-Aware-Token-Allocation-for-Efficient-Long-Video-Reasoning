"""
Module 3: Adaptive Token Allocation.

Given importance scores and event lengths, allocate a total frame budget T
across events. Handles the two edge cases we discussed:

  1. K * n_min > T  (too many events)
     -> merge low-importance adjacent events; if still too many, prune.

  2. n_k > L_k      (allocated more than event can provide)
     -> cap at L_k and redistribute overflow by importance.

Also solves rounding: the real-valued softmax-proportional allocation
is rounded using the Largest Remainder Method so that sum exactly equals T.
"""
from typing import List, Optional, Tuple
import numpy as np

from src.utils.types import Event, EventSet, AllocationResult
from src.modules.importance import softmax


# =============================================================================
# Integer rounding with exact-sum guarantee
# =============================================================================

def largest_remainder_round(
    real_values: np.ndarray,
    total: int,
) -> np.ndarray:
    """
    Round a real vector to integers that sum exactly to `total`.

    Strategy: floor everything, then distribute remaining units (total - sum_floor)
    to entries with the largest fractional parts.
    """
    floored = np.floor(real_values).astype(int)
    deficit = total - int(floored.sum())
    if deficit == 0:
        return floored
    if deficit > 0:
        # give +1 to largest remainders
        remainders = real_values - floored
        indices = np.argsort(-remainders)[:deficit]
        floored[indices] += 1
    else:
        # overshot; take 1 from smallest remainders (rare if all non-neg)
        remainders = real_values - floored
        indices = np.argsort(remainders)[:(-deficit)]
        floored[indices] -= 1
    return floored


# =============================================================================
# Edge case: too many events
# =============================================================================

def reduce_events_if_needed(
    events: EventSet,
    total_budget: int,
    n_min: int,
    strategy: str = "merge_then_prune",
) -> EventSet:
    """
    If K * n_min > T, reduce the number of events.

    merge_then_prune:
      1. Merge adjacent low-importance events until K*n_min <= T.
      2. If still violating, drop lowest-importance events.

    Returns a new EventSet with fewer events. Importance scores are
    recomputed for merged events (weighted by duration).
    """
    K = len(events)
    if K * n_min <= total_budget:
        return events

    # Convert to mutable list
    ev_list = list(events.events)

    # --- Step 1: Merge low-importance adjacent pairs ---
    target_K = total_budget // n_min
    while len(ev_list) > target_K:
        # For each adjacent pair, sum of importances (lower = better to merge)
        pair_scores = [
            (ev_list[i].importance or 0.0) + (ev_list[i + 1].importance or 0.0)
            for i in range(len(ev_list) - 1)
        ]
        i_merge = int(np.argmin(pair_scores))
        # Merge ev_list[i_merge] and ev_list[i_merge + 1]
        left, right = ev_list[i_merge], ev_list[i_merge + 1]
        merged = Event(
            event_id=left.event_id,  # reassigned at end
            start_idx=left.start_idx,
            end_idx=right.end_idx,
            start_time=left.start_time,
            end_time=right.end_time,
            complexity=_weighted_avg(left, right, "complexity"),
            relevance=(left.relevance or 0.0) + (right.relevance or 0.0),
            importance=_weighted_avg(left, right, "importance"),
        )
        ev_list[i_merge] = merged
        ev_list.pop(i_merge + 1)

    # Reassign ids
    for i, e in enumerate(ev_list):
        e.event_id = i

    return EventSet(events=ev_list, video_id=events.video_id)


def _weighted_avg(left: Event, right: Event, attr: str) -> Optional[float]:
    lv = getattr(left, attr)
    rv = getattr(right, attr)
    if lv is None or rv is None:
        return lv if lv is not None else rv
    lw = left.num_frames
    rw = right.num_frames
    return (lv * lw + rv * rw) / (lw + rw)


# =============================================================================
# Edge case: overflow (n_k > L_k) — water-filling redistribution
# =============================================================================

def cap_and_redistribute(
    allocations: np.ndarray,
    capacities: np.ndarray,
    importances: np.ndarray,
) -> np.ndarray:
    """
    Cap each allocation at its event's capacity, redistribute overflow
    to events with remaining headroom by importance priority.

    Args:
        allocations: (K,) proposed integer frame counts
        capacities: (K,) event lengths L_k
        importances: (K,) importance scores (for priority ordering)

    Returns:
        (K,) final allocations, each <= capacities
    """
    alloc = allocations.copy()
    cap = capacities.copy()

    # First pass: cap
    overflow = 0
    for k in range(len(alloc)):
        if alloc[k] > cap[k]:
            overflow += alloc[k] - cap[k]
            alloc[k] = cap[k]

    # Second pass: distribute overflow to events with headroom,
    # in order of decreasing importance
    if overflow > 0:
        priority = np.argsort(-importances)
        while overflow > 0:
            gave = False
            for k in priority:
                if alloc[k] < cap[k]:
                    alloc[k] += 1
                    overflow -= 1
                    gave = True
                    if overflow == 0:
                        break
            if not gave:
                # Nobody can accept more; remaining overflow is lost
                break

    return alloc


# =============================================================================
# Main allocation entry point
# =============================================================================

def allocate_frames(
    events: EventSet,
    total_budget: int,
    n_min: int = 2,
    tau_allocation: float = 1.0,
    reduce_events: bool = True,
) -> Tuple[EventSet, np.ndarray]:
    """
    Allocate `total_budget` frames across events in proportion to importance.

    Args:
        events: EventSet with importance scores populated
        total_budget: T, total frames for the MLLM
        n_min: minimum per event
        tau_allocation: sharpness temperature for softmax
        reduce_events: if True, handle K*n_min > T by merging/pruning

    Returns:
        possibly-modified EventSet, and (K,) allocation array.
    """
    # Edge case 1: too many events
    if reduce_events:
        events = reduce_events_if_needed(events, total_budget, n_min)

    K = len(events)
    if K == 0:
        return events, np.array([], dtype=int)

    # If sum of minimums still exceeds budget, n_min is infeasible.
    if K * n_min > total_budget:
        # Fall back: give each event 1 frame, up to budget
        alloc = np.zeros(K, dtype=int)
        alloc[:min(K, total_budget)] = 1
        return events, alloc

    # Standard allocation
    importances = np.array([e.importance or 0.0 for e in events])

    # Real-valued allocation (subtract n_min before distribution, then add back)
    remaining_budget = total_budget - K * n_min
    weights = softmax(importances, tau=tau_allocation)
    real_extra = weights * remaining_budget

    extra_int = largest_remainder_round(real_extra, remaining_budget)
    alloc = extra_int + n_min  # shape (K,)

    # Edge case 2: cap at event capacity and redistribute
    capacities = np.array([e.num_frames for e in events])
    alloc = cap_and_redistribute(alloc, capacities, importances)

    # Write back to events
    for i, e in enumerate(events):
        e.allocated_frames = int(alloc[i])

    return events, alloc
