# src/evalloc/pipeline/baseline_pipeline.py

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from evalloc.data.base import QASample
from evalloc.data.video_reader import VideoReader
from evalloc.inference.answer_parser import MultipleChoiceAnswerParser
from evalloc.inference.prompt_builder import PromptBuilder
from evalloc.inference.qwen2_vl import Qwen2VLInferencer
from evalloc.selection.uniform import UniformFrameSelector


@dataclass(frozen=True)
class UniformPipelineResult:
    sample_id: str
    answer: str | None
    prediction: str | None
    correct: bool

    raw_output: str
    parser_pattern: str | None

    frame_budget: int
    selected_frame_indices: list[int]
    selected_timestamps: list[float]

    video_fps: float
    video_total_frames: int
    video_duration: float

    input_token_count: int
    generated_token_count: int

    video_info_time: float
    frame_selection_time: float
    frame_decode_time: float
    inference_time: float
    total_time: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UniformBaselinePipeline:
    """
    End-to-end Uniform Sampling baseline.

    QASample
        -> video metadata
        -> uniformly selected frame indices
        -> decoded PIL frames
        -> Qwen2-VL inference
        -> answer parsing
        -> evaluation result
    """

    def __init__(
        self,
        *,
        video_reader: VideoReader,
        frame_selector: UniformFrameSelector,
        prompt_builder: PromptBuilder,
        inferencer: Qwen2VLInferencer,
        answer_parser: MultipleChoiceAnswerParser,
    ) -> None:
        self.video_reader = video_reader
        self.frame_selector = frame_selector
        self.prompt_builder = prompt_builder
        self.inferencer = inferencer
        self.answer_parser = answer_parser

    def run(self, sample: QASample) -> UniformPipelineResult:
        if not sample.is_multi_choice():
            raise NotImplementedError(
                "The initial Uniform baseline currently supports "
                "multiple-choice QA only."
            )

        if not sample.has_options():
            raise ValueError(
                f"Sample {sample.sample_id} has no answer options."
            )

        assert sample.options is not None

        total_start = time.perf_counter()

        info_start = time.perf_counter()
        video_info = self.video_reader.get_video_info(sample.video_path)
        video_info_time = time.perf_counter() - info_start

        selection_start = time.perf_counter()
        selected_indices = self.frame_selector.select_by_num_frames(
            video_info.total_frames
        )
        selected_timestamps = [
            frame_index / video_info.fps
            for frame_index in selected_indices
        ]
        frame_selection_time = time.perf_counter() - selection_start

        decode_start = time.perf_counter()
        frames = self.video_reader.read_frames(
            sample.video_path,
            selected_indices,
            strict=True,
        )
        frame_decode_time = time.perf_counter() - decode_start

        if len(frames) != len(selected_indices):
            raise RuntimeError(
                f"Decoded frame count mismatch for sample {sample.sample_id}: "
                f"requested={len(selected_indices)}, decoded={len(frames)}"
            )

        prompt = self.prompt_builder.build(sample)

        inference_start = time.perf_counter()
        generation = self.inferencer.generate(
            frames=frames,
            prompt=prompt,
        )
        inference_time = time.perf_counter() - inference_start

        parsed = self.answer_parser.parse(
            generation.text,
            num_options=len(sample.options),
        )

        correct = (
            parsed.answer is not None
            and sample.answer is not None
            and parsed.answer.strip().upper()
            == sample.answer.strip().upper()
        )

        total_time = time.perf_counter() - total_start

        return UniformPipelineResult(
            sample_id=sample.sample_id,
            answer=sample.answer,
            prediction=parsed.answer,
            correct=correct,
            raw_output=generation.text,
            parser_pattern=parsed.matched_pattern,
            frame_budget=self.frame_selector.budget,
            selected_frame_indices=selected_indices,
            selected_timestamps=selected_timestamps,
            video_fps=video_info.fps,
            video_total_frames=video_info.total_frames,
            video_duration=video_info.duration,
            input_token_count=generation.input_token_count,
            generated_token_count=generation.generated_token_count,
            video_info_time=video_info_time,
            frame_selection_time=frame_selection_time,
            frame_decode_time=frame_decode_time,
            inference_time=inference_time,
            total_time=total_time,
        )