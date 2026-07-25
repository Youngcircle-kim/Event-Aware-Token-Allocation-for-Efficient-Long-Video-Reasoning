from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from evalloc.segmentation.base import BaseSegmenter, Event


@dataclass
class SemanticSegmenter(BaseSegmenter):
    """
    CLIP feature-based semantic event segmenter.

    For each possible boundary position t, the segmenter compares:

        mean(features before t)
        mean(features from t onward)

    A high cosine distance indicates a semantic transition.
    """

    window_size: int = 4
    threshold_percentile: float = 85.0
    local_max_radius: int = 2
    min_event_sec: float = 15.0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")

        if not 0.0 <= self.threshold_percentile <= 100.0:
            raise ValueError(
                "threshold_percentile must be between 0 and 100."
            )

        if self.local_max_radius < 0:
            raise ValueError(
                "local_max_radius must be non-negative."
            )

        if self.min_event_sec < 0:
            raise ValueError(
                "min_event_sec must be non-negative."
            )

    def segment(
        self,
        frame_indices: list[int],
        timestamps: list[float],
        features: torch.Tensor,
    ) -> list[Event]:
        self._validate_inputs(
            frame_indices=frame_indices,
            timestamps=timestamps,
            features=features,
        )

        num_candidates = len(frame_indices)

        if num_candidates == 0:
            return []

        if num_candidates == 1:
            event = Event(
                event_id=0,
                start_idx=0,
                end_idx=0,
                start_time=timestamps[0],
                end_time=timestamps[0],
                frame_indices=[frame_indices[0]],
                boundary_score=None,
                metadata={
                    "segmenter": "semantic",
                },
            )
            event.validate()
            return [event]

        normalized_features = F.normalize(
            features.float(),
            dim=-1,
        )

        boundary_scores = self._compute_boundary_scores(
            normalized_features
        )

        candidate_boundaries = self._detect_candidate_boundaries(
            boundary_scores
        )

        selected_boundaries = self._enforce_minimum_event_duration(
            candidate_boundaries=candidate_boundaries,
            boundary_scores=boundary_scores,
            timestamps=timestamps,
        )

        events = self._build_events(
            boundaries=selected_boundaries,
            boundary_scores=boundary_scores,
            frame_indices=frame_indices,
            timestamps=timestamps,
        )

        return events

    def _compute_boundary_scores(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute one semantic-change score per candidate position.

        Position t means a possible boundary between:
            t - 1 and t

        scores[0] is always zero because no frame precedes position 0.
        """
        num_candidates = features.shape[0]

        scores = torch.zeros(
            num_candidates,
            dtype=torch.float32,
        )

        for boundary_idx in range(1, num_candidates):
            left_start = max(
                0,
                boundary_idx - self.window_size,
            )
            right_end = min(
                num_candidates,
                boundary_idx + self.window_size,
            )

            left_features = features[
                left_start:boundary_idx
            ]
            right_features = features[
                boundary_idx:right_end
            ]

            if (
                left_features.shape[0] == 0
                or right_features.shape[0] == 0
            ):
                continue

            left_representation = F.normalize(
                left_features.mean(dim=0),
                dim=0,
            )

            right_representation = F.normalize(
                right_features.mean(dim=0),
                dim=0,
            )

            similarity = torch.dot(
                left_representation,
                right_representation,
            )

            scores[boundary_idx] = (
                1.0 - similarity
            ).clamp(min=0.0)

        return scores

    def _detect_candidate_boundaries(
        self,
        boundary_scores: torch.Tensor,
    ) -> list[int]:
        valid_scores = boundary_scores[1:]

        if valid_scores.numel() == 0:
            return []

        threshold = torch.quantile(
            valid_scores,
            self.threshold_percentile / 100.0,
        ).item()

        candidates: list[int] = []

        for boundary_idx in range(
            1,
            len(boundary_scores),
        ):
            score = boundary_scores[
                boundary_idx
            ].item()

            if score < threshold:
                continue

            local_start = max(
                1,
                boundary_idx - self.local_max_radius,
            )

            local_end = min(
                len(boundary_scores),
                boundary_idx
                + self.local_max_radius
                + 1,
            )

            local_scores = boundary_scores[
                local_start:local_end
            ]

            # 주변 지점보다 작은 경우 boundary로 선택하지 않음
            if score < local_scores.max().item():
                continue

            candidates.append(boundary_idx)

        return candidates

    def _enforce_minimum_event_duration(
        self,
        *,
        candidate_boundaries: list[int],
        boundary_scores: torch.Tensor,
        timestamps: list[float],
    ) -> list[int]:
        """
        Select strong boundaries while preventing very short events.

        Boundaries are considered in descending score order.
        A candidate is accepted only when it is sufficiently far from
        already selected boundaries and the video boundaries.
        """
        if not candidate_boundaries:
            return []

        video_start = timestamps[0]
        video_end = timestamps[-1]

        sorted_candidates = sorted(
            candidate_boundaries,
            key=lambda index: boundary_scores[index].item(),
            reverse=True,
        )

        selected: list[int] = []

        for candidate in sorted_candidates:
            candidate_time = timestamps[candidate]

            # 첫 event가 지나치게 짧아지는 것 방지
            if candidate_time - video_start < self.min_event_sec:
                continue

            # 마지막 event가 지나치게 짧아지는 것 방지
            if video_end - candidate_time < self.min_event_sec:
                continue

            if any(
                abs(
                    candidate_time
                    - timestamps[selected_boundary]
                )
                < self.min_event_sec
                for selected_boundary in selected
            ):
                continue

            selected.append(candidate)

        return sorted(selected)

    def _build_events(
        self,
        *,
        boundaries: list[int],
        boundary_scores: torch.Tensor,
        frame_indices: list[int],
        timestamps: list[float],
    ) -> list[Event]:
        """
        Boundary index b means:
            previous event ends at b - 1
            next event starts at b
        """
        start_positions = [0] + boundaries

        end_positions = (
            [boundary - 1 for boundary in boundaries]
            + [len(frame_indices) - 1]
        )

        events: list[Event] = []

        for event_id, (
            start_idx,
            end_idx,
        ) in enumerate(
            zip(start_positions, end_positions)
        ):
            if start_idx > end_idx:
                continue

            boundary_score = (
                boundary_scores[start_idx].item()
                if start_idx > 0
                else None
            )

            event = Event(
                event_id=event_id,
                start_idx=start_idx,
                end_idx=end_idx,
                start_time=timestamps[start_idx],
                end_time=timestamps[end_idx],
                frame_indices=frame_indices[
                    start_idx : end_idx + 1
                ],
                boundary_score=boundary_score,
                metadata={
                    "segmenter": "semantic",
                    "window_size": self.window_size,
                    "threshold_percentile": (
                        self.threshold_percentile
                    ),
                },
            )

            event.validate()
            events.append(event)

        return events

    @staticmethod
    def _validate_inputs(
        *,
        frame_indices: list[int],
        timestamps: list[float],
        features: torch.Tensor,
    ) -> None:
        num_candidates = len(frame_indices)

        if len(timestamps) != num_candidates:
            raise ValueError(
                "frame_indices and timestamps must have "
                "the same length."
            )

        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [N, D], "
                f"got {tuple(features.shape)}."
            )

        if features.shape[0] != num_candidates:
            raise ValueError(
                f"features contains {features.shape[0]} rows, "
                f"but frame_indices contains {num_candidates}."
            )

        if any(
            later < earlier
            for earlier, later in zip(
                timestamps,
                timestamps[1:],
            )
        ):
            raise ValueError(
                "timestamps must be sorted in ascending order."
            )

        if any(
            later <= earlier
            for earlier, later in zip(
                frame_indices,
                frame_indices[1:],
            )
        ):
            raise ValueError(
                "frame_indices must be strictly increasing."
            )