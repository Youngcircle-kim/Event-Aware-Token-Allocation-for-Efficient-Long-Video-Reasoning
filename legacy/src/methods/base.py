"""
Abstract base class for QA methods.

A Method takes a (MockVideoSpec, VideoQAExample) and returns a Prediction.
This unified interface lets us plug in:
  - UniformBaseline
  - OracleAllocation
  - EventAwareMethod (ours)
and compare them with the same evaluation harness.
"""
from abc import ABC, abstractmethod
import time
import numpy as np

from src.utils.types import VideoQAExample, Prediction
from src.data.mock_generator import MockVideoSpec
from src.models.mock_mllm import MockMLLM


class BaseMethod(ABC):
    """Base class for all QA methods."""

    def __init__(self, mllm: MockMLLM, name: str = "base"):
        self.mllm = mllm
        self.name = name

    @abstractmethod
    def select_frames(
        self,
        spec: MockVideoSpec,
        example: VideoQAExample,
    ) -> np.ndarray:
        """
        Choose which frames (timestamps) to feed to the MLLM.
        Returns an array of timestamps in seconds.
        """
        raise NotImplementedError

    def __call__(
        self,
        spec: MockVideoSpec,
        example: VideoQAExample,
    ) -> Prediction:
        start = time.perf_counter()
        frame_timestamps = self.select_frames(spec, example)
        mllm_out = self.mllm.answer(
            question=example.question,
            options=example.options,
            frame_timestamps=frame_timestamps,
            spec=spec,
            gt_answer=example.answer,
        )
        elapsed = time.perf_counter() - start

        return Prediction(
            example_id=example.video_id,
            predicted_answer=mllm_out.predicted_letter,
            gt_answer=example.answer,
            is_correct=(mllm_out.predicted_letter == example.answer),
            num_frames_used=len(frame_timestamps),
            num_events=getattr(self, "_last_num_events", 0),
            allocation=getattr(self, "_last_allocation", None),
            latency_seconds=elapsed,
            raw_output=mllm_out.raw_text,
        )
