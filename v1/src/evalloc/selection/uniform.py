from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def uniform_sample_indices(
    total_frames: int,
    budget: int,
    *,
    include_last: bool = True,
) -> list[int]:
    """
    Uniformly sample frame indices from a video.

    Args:
        total_frames:
            Total number of frames in the video.
        budget:
            Number of frames to sample.
        include_last:
            If True, the first and last frames are included when budget > 1.
            Example:
                total_frames=100, budget=4 -> [0, 33, 66, 99]

            If False, samples are taken from bin centers.
            Example:
                total_frames=100, budget=4 -> [12, 38, 62, 88]

    Returns:
        Sorted list of unique frame indices.

    Notes:
        - Returned indices are original video frame indices.
        - The function always tries to return exactly `budget` indices,
          unless `budget >= total_frames`, in which case all frames are returned.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")

    if budget <= 0:
        return []

    if budget >= total_frames:
        return list(range(total_frames))

    if budget == 1:
        if include_last:
            return [total_frames // 2]
        return [total_frames // 2]

    if include_last:
        indices = [
            round(i * (total_frames - 1) / (budget - 1))
            for i in range(budget)
        ]
    else:
        step = total_frames / budget
        indices = [
            round((i + 0.5) * step)
            for i in range(budget)
        ]

    indices = [
        min(max(int(idx), 0), total_frames - 1)
        for idx in indices
    ]

    return _ensure_unique_budget(
        indices=indices,
        total_frames=total_frames,
        budget=budget,
    )


def uniform_sample_from_candidates(
    candidate_indices: Sequence[int],
    budget: int,
    *,
    include_last: bool = True,
) -> list[int]:
    """
    Uniformly sample from a given list of candidate frame indices.

    This is useful when features are extracted every N seconds and we want
    to sample uniformly over the candidate frame list rather than all raw frames.

    Args:
        candidate_indices:
            Candidate original video frame indices.
            Example: [0, 60, 120, 180, ...]
        budget:
            Number of indices to sample.
        include_last:
            If True, include first and last candidate when budget > 1.

    Returns:
        Sorted list of sampled original video frame indices.
    """
    candidates = sorted({int(idx) for idx in candidate_indices})

    if len(candidates) == 0:
        return []

    if budget <= 0:
        return []

    if budget >= len(candidates):
        return candidates

    sampled_positions = uniform_sample_indices(
        total_frames=len(candidates),
        budget=budget,
        include_last=include_last,
    )

    sampled = [candidates[pos] for pos in sampled_positions]
    return sorted(sampled)


def uniform_sample_timestamps(
    duration: float,
    budget: int,
    *,
    include_last: bool = True,
) -> list[float]:
    """
    Uniformly sample timestamps from a video duration.

    Args:
        duration:
            Video duration in seconds.
        budget:
            Number of timestamps to sample.
        include_last:
            If True, include 0 and duration when budget > 1.

    Returns:
        Sorted timestamps in seconds.
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    if budget <= 0:
        return []

    if budget == 1:
        return [duration / 2.0]

    if include_last:
        return [
            i * duration / (budget - 1)
            for i in range(budget)
        ]

    step = duration / budget
    return [
        (i + 0.5) * step
        for i in range(budget)
    ]


def _ensure_unique_budget(
    indices: Sequence[int],
    total_frames: int,
    budget: int,
) -> list[int]:
    """
    Ensure sampled indices are unique and match the requested budget.

    Rounding can create duplicate indices in rare cases. This function fills
    missing positions deterministically.
    """
    unique = sorted({int(idx) for idx in indices})

    if len(unique) >= budget:
        return sorted(unique[:budget])

    used = set(unique)

    for idx in range(total_frames):
        if idx not in used:
            unique.append(idx)
            used.add(idx)

        if len(unique) == budget:
            break

    return sorted(unique[:budget])


@dataclass
class UniformFrameSelector:
    """
    Uniform frame selector for baseline experiments.

    This selector works at the video level.
    It does not use question, features, or event information.
    """

    budget: int
    include_last: bool = True

    def select_by_num_frames(self, total_frames: int) -> list[int]:
        """
        Select uniformly spaced frame indices from a video.

        Args:
            total_frames:
                Total number of frames in the video.

        Returns:
            Selected original video frame indices.
        """
        return uniform_sample_indices(
            total_frames=total_frames,
            budget=self.budget,
            include_last=self.include_last,
        )

    def select_from_candidates(
        self,
        candidate_indices: Sequence[int],
    ) -> list[int]:
        """
        Select uniformly from candidate frame indices.

        Args:
            candidate_indices:
                Candidate original video frame indices.

        Returns:
            Selected original video frame indices.
        """
        return uniform_sample_from_candidates(
            candidate_indices=candidate_indices,
            budget=self.budget,
            include_last=self.include_last,
        )

    def select_by_duration(
        self,
        duration: float,
    ) -> list[float]:
        """
        Select uniformly spaced timestamps from video duration.

        Args:
            duration:
                Video duration in seconds.

        Returns:
            Selected timestamps in seconds.
        """
        return uniform_sample_timestamps(
            duration=duration,
            budget=self.budget,
            include_last=self.include_last,
        )