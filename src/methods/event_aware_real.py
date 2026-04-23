import time
from typing import Any, Dict

import numpy as np

from src.methods.real_utils import (
    get_video_meta,
    simple_event_boundaries,
    allocate_budget_by_segment_lengths,
    sample_indices_within_segments,
    load_frames_as_pil,
)
from src.models.qwen_vl_mcq import QwenVLMCQ


class EventAwareMethodReal:
    def __init__(self, qa_model: QwenVLMCQ, stage1_stride_sec: float = 2.0):
        self.name = "event_aware_real"
        self.stage1_stride_sec = stage1_stride_sec
        self.qa_model = qa_model

    def run(self, example, token_budget: int) -> Dict[str, Any]:
        _, num_frames, fps, _ = get_video_meta(example.video_path)

        stage1_start = time.perf_counter()

        boundaries = np.asarray(
            simple_event_boundaries(
                num_frames=num_frames,
                fps=fps,
                stage1_stride_sec=self.stage1_stride_sec,
            ),
            dtype=int,
        )

        allocations = allocate_budget_by_segment_lengths(boundaries, token_budget)
        indices = sample_indices_within_segments(boundaries, allocations)

        stage1_latency = time.perf_counter() - stage1_start

        stage2_start = time.perf_counter()
        frames = load_frames_as_pil(example.video_path, indices)
        qa_result = self.qa_model.answer_mcq(
            frames=frames,
            question=example.question,
            options=example.options,
        )
        stage2_latency = time.perf_counter() - stage2_start

        return {
            "predicted_answer": qa_result["predicted_answer"],
            "raw_output": qa_result["raw_output"],
            "num_visual_tokens": int(len(indices)),
            "num_frames_used": int(len(indices)),
            "stage1_latency_s": float(stage1_latency),
            "stage2_latency_s": float(stage2_latency),
            "num_events_detected": int(len(allocations)),
            "allocation": allocations,
        }