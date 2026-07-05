"""
Mock Multimodal LLM.

Simulates Qwen2-VL's interface:
  answer(question, options, frames, frame_latents) -> str (letter)

The mock MLLM uses a simple "does the frame set contain the answer event?"
heuristic to decide correctness. This lets us validate end-to-end pipelines:
if our allocation method puts frames in the correct event, the mock MLLM
will return the correct answer.

# TODO: REAL
Replace MockMLLM with a Qwen2VLWrapper around transformers'
Qwen2VLForConditionalGeneration. Keep the `answer(...)` signature identical.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from src.data.mock_generator import MockVideoSpec


@dataclass
class MLLMOutput:
    predicted_letter: str
    raw_text: str
    # Instrumentation
    num_input_frames: int
    latency_seconds: float = 0.0
    gpu_memory_mb: float = 0.0


class MockMLLM:
    """
    Mock MLLM that pretends to answer a video QA question.

    Correctness rule:
      If at least `min_frames_in_gt_event` frames of the input fall within
      the ground-truth answer event's time range, the model answers correctly.
      Otherwise, it picks a random wrong option.

    This simulates the intuition that models need enough visual evidence
    about the relevant part of the video to answer correctly.
    """

    def __init__(self, min_frames_in_gt_event: int = 2, seed: int = 0):
        self.min_frames_in_gt_event = min_frames_in_gt_event
        self.rng = np.random.default_rng(seed)

    def answer(
        self,
        question: str,
        options: List[str],
        frame_timestamps: np.ndarray,
        spec: MockVideoSpec,
        gt_answer: str,
    ) -> MLLMOutput:
        """
        Args:
            question: question text (unused by mock, kept for interface parity)
            options: answer options
            frame_timestamps: (N,) timestamps in seconds of provided frames
            spec: ground-truth video spec (needed only in mock mode)
            gt_answer: ground-truth letter, e.g. "A"

        Returns:
            MLLMOutput with predicted letter and instrumentation.

        # TODO: REAL
        The `spec` and `gt_answer` params are mock-only cheats. The real
        wrapper will not need them — it'll just call the MLLM with frames
        and question.
        """
        gt_event_idx = spec.answer_event_idx
        gt_start = spec.event_boundaries[gt_event_idx]
        gt_end = spec.event_boundaries[gt_event_idx + 1]

        frames_in_gt = np.sum(
            (frame_timestamps >= gt_start) & (frame_timestamps < gt_end)
        )

        if frames_in_gt >= self.min_frames_in_gt_event:
            predicted = gt_answer
            raw_text = f"Answer: {gt_answer}"
        else:
            # Guess a wrong option
            letters = [chr(65 + i) for i in range(len(options))]
            wrong_letters = [l for l in letters if l != gt_answer]
            predicted = self.rng.choice(wrong_letters) if wrong_letters else gt_answer
            raw_text = f"Answer: {predicted}"

        return MLLMOutput(
            predicted_letter=predicted,
            raw_text=raw_text,
            num_input_frames=len(frame_timestamps),
        )
