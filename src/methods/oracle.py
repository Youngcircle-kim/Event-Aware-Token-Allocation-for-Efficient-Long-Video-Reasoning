"""
Oracle allocation — cheats by using the ground-truth answer event.

This is used ONLY for the pilot study to measure the upper bound of
event-aware allocation. If oracle - uniform gap is small, our method
has limited headroom.

Strategy: give the GT answer event 70% of the budget uniformly, and
distribute the remaining 30% uniformly across all other events.
"""
import numpy as np

from src.methods.base import BaseMethod
from src.utils.types import VideoQAExample
from src.data.mock_generator import MockVideoSpec
from src.models.mock_mllm import MockMLLM


class OracleAllocation(BaseMethod):

    def __init__(
        self,
        mllm: MockMLLM,
        total_budget: int = 32,
        gt_event_share: float = 0.7,
    ):
        super().__init__(mllm, name=f"oracle_T{total_budget}")
        self.total_budget = total_budget
        self.gt_event_share = gt_event_share

    def select_frames(
        self,
        spec: MockVideoSpec,
        example: VideoQAExample,
    ) -> np.ndarray:
        T = self.total_budget
        K = spec.num_events
        gt_k = spec.answer_event_idx

        n_gt = int(round(T * self.gt_event_share))
        n_other = T - n_gt
        K_other = K - 1
        if K_other <= 0:
            # only one event; all budget goes there
            start, end = spec.event_boundaries[0], spec.event_boundaries[1]
            return np.linspace(start, end, T, endpoint=False)

        per_other = n_other // K_other
        leftover = n_other - per_other * K_other

        timestamps_list = []
        for k in range(K):
            start = spec.event_boundaries[k]
            end = spec.event_boundaries[k + 1]
            if k == gt_k:
                n_k = n_gt
            else:
                n_k = per_other + (1 if leftover > 0 else 0)
                if leftover > 0:
                    leftover -= 1
            if n_k > 0:
                timestamps_list.append(
                    np.linspace(start, end, n_k, endpoint=False)
                )

        ts = np.concatenate(timestamps_list) if timestamps_list else np.array([])
        self._last_num_events = K
        self._last_allocation = [
            n_gt if k == gt_k else (n_other // K_other) for k in range(K)
        ]
        return ts
