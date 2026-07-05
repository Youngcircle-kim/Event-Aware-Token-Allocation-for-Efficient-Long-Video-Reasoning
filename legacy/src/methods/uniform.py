"""
Uniform sampling baseline.

Samples T frames at equal intervals across the whole video, ignoring
content and question. This is the primary baseline we must beat.
"""
import numpy as np

from src.methods.base import BaseMethod
from src.utils.types import VideoQAExample
from src.data.mock_generator import MockVideoSpec
from src.models.mock_mllm import MockMLLM


class UniformBaseline(BaseMethod):

    def __init__(self, mllm: MockMLLM, total_budget: int = 32):
        super().__init__(mllm, name=f"uniform_T{total_budget}")
        self.total_budget = total_budget

    def select_frames(
        self,
        spec: MockVideoSpec,
        example: VideoQAExample,
    ) -> np.ndarray:
        T = self.total_budget
        timestamps = np.linspace(0, spec.duration, T, endpoint=False)
        self._last_num_events = 0
        self._last_allocation = None
        return timestamps
