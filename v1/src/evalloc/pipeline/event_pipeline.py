from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from evalloc.allocation.softmax import SoftmaxAllocator
from evalloc.data.base import QASample
from evalloc.data.video_reader import VideoReader
from evalloc.features.clip_extractor import (
    CLIPFeatureExtractor,
    VideoFeatures,
)
from evalloc.inference.answer_parser import (
    MultipleChoiceAnswerParser,
)
from evalloc.inference.prompt_builder import PromptBuilder
from evalloc.inference.qwen2_vl import Qwen2VLInferencer
from evalloc.scoring.base import BaseEventScorer, EventScore
from evalloc.segmentation.base import BaseSegmenter, Event
from evalloc.selection.diverse import (
    RelevanceDiverseSelector,
)


@dataclass(frozen=True)
class EventAwarePipelineResult:
    sample_id: str

    answer: str | None
    prediction: str | None
    correct: bool

    raw_output: str
    parser_pattern: str | None

    requested_frame_budget: int
    actual_frame_budget: int

    candidate_stride_sec: float
    num_candidate_frames: int
    num_events: int

    selected_frame_indices: list[int]
    selected_timestamps: list[float]

    events: list[dict[str, Any]]
    event_scores: list[dict[str, Any]]
    allocation: dict[int, int]

    input_token_count: int
    generated_token_count: int

    video_info_time: float
    candidate_feature_time: float
    segmentation_time: float
    scoring_time: float
    allocation_time: float
    selection_time: float
    final_frame_decode_time: float
    inference_time: float
    total_time: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventAwarePipeline:
    """
    End-to-end Event-aware frame selection pipeline.

    QASample
        -> candidate-frame extraction
        -> CLIP image/text features
        -> semantic event segmentation
        -> event scoring
        -> event-level frame-budget allocation
        -> within-event frame selection
        -> Qwen2-VL inference
        -> multiple-choice answer parsing
    """

    def __init__(
        self,
        *,
        video_reader: VideoReader,
        feature_extractor: CLIPFeatureExtractor,
        segmenter: BaseSegmenter,
        scorer: BaseEventScorer,
        allocator: SoftmaxAllocator,
        selector: RelevanceDiverseSelector,
        prompt_builder: PromptBuilder,
        inferencer: Qwen2VLInferencer,
        answer_parser: MultipleChoiceAnswerParser,
        frame_budget: int,
        candidate_stride_sec: float,
        candidate_decode_batch_size: int,
    ) -> None:
        if frame_budget <= 0:
            raise ValueError(
                f"frame_budget must be positive, got {frame_budget}."
            )

        if candidate_stride_sec <= 0:
            raise ValueError(
                "candidate_stride_sec must be positive, got "
                f"{candidate_stride_sec}."
            )

        if candidate_decode_batch_size <= 0:
            raise ValueError(
                "candidate_decode_batch_size must be positive, got "
                f"{candidate_decode_batch_size}."
            )

        self.video_reader = video_reader
        self.feature_extractor = feature_extractor
        self.segmenter = segmenter
        self.scorer = scorer
        self.allocator = allocator
        self.selector = selector
        self.prompt_builder = prompt_builder
        self.inferencer = inferencer
        self.answer_parser = answer_parser

        self.frame_budget = frame_budget
        self.candidate_stride_sec = candidate_stride_sec
        self.candidate_decode_batch_size = (
            candidate_decode_batch_size
        )

    def run(
        self,
        sample: QASample,
    ) -> EventAwarePipelineResult:
        self._validate_sample(sample)

        assert sample.options is not None

        total_start = time.perf_counter()

        # -------------------------------------------------------------
        # 1. Read video metadata
        # -------------------------------------------------------------
        video_info_start = time.perf_counter()

        video_info = self.video_reader.get_video_info(
            sample.video_path
        )

        video_info_time = (
            time.perf_counter() - video_info_start
        )

        # -------------------------------------------------------------
        # 2. Construct candidate frame positions
        # -------------------------------------------------------------
        candidate_indices = self._make_candidate_indices(
            total_frames=video_info.total_frames,
            fps=video_info.fps,
        )

        candidate_timestamps = [
            frame_index / video_info.fps
            for frame_index in candidate_indices
        ]

        # -------------------------------------------------------------
        # 3. Decode candidate frames and extract CLIP features
        # -------------------------------------------------------------
        feature_start = time.perf_counter()

        visual_features = self._extract_candidate_features(
            video_path=sample.video_path,
            candidate_indices=candidate_indices,
        )

        question_feature = self.feature_extractor.encode_text(
            sample.question
        )

        candidate_feature_time = (
            time.perf_counter() - feature_start
        )

        video_features = VideoFeatures(
            frame_indices=candidate_indices,
            timestamps=candidate_timestamps,
            features=visual_features,
        )

        # -------------------------------------------------------------
        # 4. Semantic event segmentation
        # -------------------------------------------------------------
        segmentation_start = time.perf_counter()

        events = self.segmenter.segment(
            frame_indices=video_features.frame_indices,
            timestamps=video_features.timestamps,
            features=video_features.features,
        )

        segmentation_time = (
            time.perf_counter() - segmentation_start
        )

        if not events:
            raise RuntimeError(
                f"Segmenter produced no events for sample "
                f"{sample.sample_id}."
            )

        self._validate_events(
            events=events,
            num_candidates=len(candidate_indices),
        )

        # -------------------------------------------------------------
        # 5. Event-level scoring
        # -------------------------------------------------------------
        scoring_start = time.perf_counter()

        event_scores = self.scorer.score(
            events=events,
            features=video_features.features,
            question_feature=question_feature,
        )

        scoring_time = (
            time.perf_counter() - scoring_start
        )

        self._validate_scores(
            events=events,
            event_scores=event_scores,
        )

        # -------------------------------------------------------------
        # 6. Event-level budget allocation
        # -------------------------------------------------------------
        allocation_start = time.perf_counter()

        allocation = self.allocator.allocate(
            events=events,
            scores=event_scores,
            total_budget=self.frame_budget,
        )

        allocation_time = (
            time.perf_counter() - allocation_start
        )

        effective_budget = min(
            self.frame_budget,
            len(candidate_indices),
        )

        allocated_budget = sum(allocation.values())

        if allocated_budget != effective_budget:
            raise RuntimeError(
                "Unexpected allocation sum: "
                f"expected={effective_budget}, "
                f"actual={allocated_budget}"
            )

        # -------------------------------------------------------------
        # 7. Within-event relevance-diverse frame selection
        # -------------------------------------------------------------
        selection_start = time.perf_counter()

        selected_indices: list[int] = []

        for event in events:
            event_budget = allocation.get(
                event.event_id,
                0,
            )

            if event_budget == 0:
                continue

            event_selected_indices = self.selector.select(
                event=event,
                video_features=video_features,
                question_feature=question_feature,
                budget=event_budget,
            )

            if len(event_selected_indices) != event_budget:
                raise RuntimeError(
                    f"Selector returned an unexpected number of frames "
                    f"for event {event.event_id}: "
                    f"expected={event_budget}, "
                    f"actual={len(event_selected_indices)}"
                )

            selected_indices.extend(
                event_selected_indices
            )

        selection_time = (
            time.perf_counter() - selection_start
        )

        if len(selected_indices) != len(set(selected_indices)):
            duplicates = self._find_duplicates(
                selected_indices
            )

            raise RuntimeError(
                f"Duplicate frame selections detected: {duplicates}"
            )

        selected_indices = sorted(selected_indices)

        if len(selected_indices) != effective_budget:
            raise RuntimeError(
                "Final selected-frame count mismatch: "
                f"expected={effective_budget}, "
                f"actual={len(selected_indices)}"
            )

        selected_timestamps = [
            frame_index / video_info.fps
            for frame_index in selected_indices
        ]

        # -------------------------------------------------------------
        # 8. Decode final selected frames
        # -------------------------------------------------------------
        final_decode_start = time.perf_counter()

        selected_frames = self.video_reader.read_frames(
            sample.video_path,
            selected_indices,
            strict=True,
        )

        final_frame_decode_time = (
            time.perf_counter() - final_decode_start
        )

        if len(selected_frames) != len(selected_indices):
            raise RuntimeError(
                "Final frame decoding count mismatch: "
                f"requested={len(selected_indices)}, "
                f"decoded={len(selected_frames)}"
            )

        # -------------------------------------------------------------
        # 9. Build prompt and run Qwen2-VL
        # -------------------------------------------------------------
        prompt = self.prompt_builder.build(sample)

        inference_start = time.perf_counter()

        generation_result = self.inferencer.generate(
            frames=selected_frames,
            prompt=prompt,
        )

        inference_time = (
            time.perf_counter() - inference_start
        )

        # -------------------------------------------------------------
        # 10. Parse and evaluate answer
        # -------------------------------------------------------------
        parsed_answer = self.answer_parser.parse(
            generation_result.text,
            num_options=len(sample.options),
        )

        normalized_ground_truth = (
            sample.answer.strip().upper()
            if sample.answer is not None
            else None
        )

        correct = (
            parsed_answer.answer is not None
            and normalized_ground_truth is not None
            and parsed_answer.answer
            == normalized_ground_truth
        )

        total_time = (
            time.perf_counter() - total_start
        )

        event_records = self._build_event_records(
            events=events,
            scores=event_scores,
            allocation=allocation,
        )

        return EventAwarePipelineResult(
            sample_id=sample.sample_id,
            answer=sample.answer,
            prediction=parsed_answer.answer,
            correct=correct,
            raw_output=generation_result.text,
            parser_pattern=parsed_answer.matched_pattern,
            requested_frame_budget=self.frame_budget,
            actual_frame_budget=len(selected_indices),
            candidate_stride_sec=self.candidate_stride_sec,
            num_candidate_frames=len(candidate_indices),
            num_events=len(events),
            selected_frame_indices=selected_indices,
            selected_timestamps=selected_timestamps,
            events=event_records,
            event_scores=[
                score.to_dict()
                for score in event_scores
            ],
            allocation=allocation,
            input_token_count=(
                generation_result.input_token_count
            ),
            generated_token_count=(
                generation_result.generated_token_count
            ),
            video_info_time=video_info_time,
            candidate_feature_time=candidate_feature_time,
            segmentation_time=segmentation_time,
            scoring_time=scoring_time,
            allocation_time=allocation_time,
            selection_time=selection_time,
            final_frame_decode_time=(
                final_frame_decode_time
            ),
            inference_time=inference_time,
            total_time=total_time,
        )

    def _extract_candidate_features(
        self,
        *,
        video_path: str | Path,
        candidate_indices: list[int],
    ) -> torch.Tensor:
        """
        Decode and encode candidate frames in chunks.

        Only CLIP features remain in memory after each batch.
        Raw candidate images are released after encoding.
        """
        feature_batches: list[torch.Tensor] = []

        for start in range(
            0,
            len(candidate_indices),
            self.candidate_decode_batch_size,
        ):
            batch_indices = candidate_indices[
                start :
                start + self.candidate_decode_batch_size
            ]

            batch_frames = self.video_reader.read_frames(
                video_path,
                batch_indices,
                strict=True,
            )

            if len(batch_frames) != len(batch_indices):
                raise RuntimeError(
                    "Candidate decoding count mismatch: "
                    f"requested={len(batch_indices)}, "
                    f"decoded={len(batch_frames)}"
                )

            batch_features = (
                self.feature_extractor.encode_images(
                    batch_frames
                )
            )

            if batch_features.ndim != 2:
                raise RuntimeError(
                    "CLIP features must have shape [N, D], "
                    f"got {tuple(batch_features.shape)}."
                )

            if (
                batch_features.shape[0]
                != len(batch_indices)
            ):
                raise RuntimeError(
                    "CLIP feature count mismatch: "
                    f"expected={len(batch_indices)}, "
                    f"actual={batch_features.shape[0]}"
                )

            feature_batches.append(
                batch_features.detach().cpu()
            )

            del batch_frames
            del batch_features

        if not feature_batches:
            raise RuntimeError(
                "No candidate features were extracted."
            )

        features = torch.cat(
            feature_batches,
            dim=0,
        )

        if features.shape[0] != len(candidate_indices):
            raise RuntimeError(
                "Total candidate-feature count mismatch: "
                f"expected={len(candidate_indices)}, "
                f"actual={features.shape[0]}"
            )

        return features

    def _make_candidate_indices(
        self,
        *,
        total_frames: int,
        fps: float,
    ) -> list[int]:
        if total_frames <= 0:
            raise ValueError(
                f"total_frames must be positive, got {total_frames}."
            )

        if fps <= 0:
            raise ValueError(
                f"fps must be positive, got {fps}."
            )

        stride_frames = max(
            1,
            round(self.candidate_stride_sec * fps),
        )

        indices = list(
            range(
                0,
                total_frames,
                stride_frames,
            )
        )

        last_frame_index = total_frames - 1

        if not indices:
            return [last_frame_index]

        if indices[-1] != last_frame_index:
            indices.append(last_frame_index)

        return indices

    @staticmethod
    def _validate_sample(
        sample: QASample,
    ) -> None:
        if not sample.is_multi_choice():
            raise NotImplementedError(
                "The current Event-aware pipeline supports "
                "multiple-choice QA only."
            )

        if not sample.has_options():
            raise ValueError(
                f"Sample {sample.sample_id} has no answer options."
            )

        if sample.answer is None:
            raise ValueError(
                f"Sample {sample.sample_id} has no ground-truth answer."
            )

        if not sample.video_path.exists():
            raise FileNotFoundError(
                f"Video file not found for {sample.sample_id}: "
                f"{sample.video_path}"
            )

    @staticmethod
    def _validate_events(
        *,
        events: list[Event],
        num_candidates: int,
    ) -> None:
        expected_start = 0

        for expected_id, event in enumerate(events):
            event.validate()

            if event.event_id != expected_id:
                raise RuntimeError(
                    "Event IDs must be consecutive and ordered: "
                    f"expected={expected_id}, "
                    f"actual={event.event_id}"
                )

            if event.start_idx != expected_start:
                raise RuntimeError(
                    "Events must form a contiguous partition: "
                    f"expected start={expected_start}, "
                    f"actual start={event.start_idx}"
                )

            expected_start = event.end_idx + 1

        if expected_start != num_candidates:
            raise RuntimeError(
                "Events do not cover every candidate frame: "
                f"covered={expected_start}, "
                f"candidates={num_candidates}"
            )

    @staticmethod
    def _validate_scores(
        *,
        events: list[Event],
        event_scores: list[EventScore],
    ) -> None:
        if len(events) != len(event_scores):
            raise RuntimeError(
                "Scorer returned an unexpected number of scores: "
                f"events={len(events)}, "
                f"scores={len(event_scores)}"
            )

        for event, score in zip(
            events,
            event_scores,
        ):
            if event.event_id != score.event_id:
                raise RuntimeError(
                    "Event-score ID mismatch: "
                    f"event={event.event_id}, "
                    f"score={score.event_id}"
                )

            is_finite = torch.isfinite(
                torch.tensor(
                    score.score,
                    dtype=torch.float32,
                )
            ).item()

            if not is_finite:
                raise RuntimeError(
                    f"Non-finite score detected for "
                    f"event {score.event_id}: {score.score}"
                )

    @staticmethod
    def _build_event_records(
        *,
        events: list[Event],
        scores: list[EventScore],
        allocation: dict[int, int],
    ) -> list[dict[str, Any]]:
        score_by_event = {
            score.event_id: score
            for score in scores
        }

        records: list[dict[str, Any]] = []

        for event in events:
            score = score_by_event.get(
                event.event_id
            )

            if score is None:
                raise RuntimeError(
                    f"No score found for event "
                    f"{event.event_id}."
                )

            record = event.to_dict()

            record["score"] = score.score
            record["score_components"] = (
                score.components
            )
            record["allocated_frames"] = (
                allocation.get(event.event_id, 0)
            )

            records.append(record)

        return records

    @staticmethod
    def _find_duplicates(
        values: list[int],
    ) -> list[int]:
        seen: set[int] = set()
        duplicates: set[int] = set()

        for value in values:
            if value in seen:
                duplicates.add(value)
            else:
                seen.add(value)

        return sorted(duplicates)