import time
from typing import Dict, Any

import numpy as np

from src.methods.real_utils import (
    get_video_meta,
    simple_event_boundaries,
    allocate_budget_by_segment_lengths,
    sample_indices_within_segments,
)


class EventAwareMethodReal:
    def __init__(self, stage1_stride_sec: float = 2.0, diff_threshold: float = 20.0):
        self.name = "event_aware_real"
        self.stage1_stride_sec = stage1_stride_sec
        self.diff_threshold = diff_threshold

    def run(self, example, token_budget: int) -> Dict[str, Any]:
        """
        Lightweight real-video version:
        1) load video metadata
        2) sparse segmentation
        3) allocate budget across segments
        4) sample final frames
        5) dummy answer for now
        """
        t0 = time.perf_counter()

        vr, num_frames, fps, duration = get_video_meta(example.video_path)

        stage1_start = time.perf_counter()

        boundaries = simple_event_boundaries(
            num_frames=num_frames,
            fps=fps,
            stage1_stride_sec=self.stage1_stride_sec,
            diff_threshold=self.diff_threshold,
        )

        if len(boundaries) < 2:
            boundaries = np.array([0, num_frames], dtype=int)
        else:
            boundaries = np.asarray(boundaries, dtype=int)
            if boundaries[0] != 0:
                boundaries = np.insert(boundaries, 0, 0)
            if boundaries[-1] != num_frames:
                boundaries = np.append(boundaries, num_frames)

        allocations = allocate_budget_by_segment_lengths(boundaries, token_budget)
        selected_indices = sample_indices_within_segments(boundaries, allocations)

        stage1_latency = time.perf_counter() - stage1_start

        stage2_start = time.perf_counter()

        # TODO: real MLLM inference 연결 전까지 dummy output
        predicted_answer = "A"
        raw_output = (
            f"[DUMMY] Event-aware selected {len(selected_indices)} frames "
            f"across {len(allocations)} events for {example.video_id}. "
            f"Replace with Qwen2-VL inference."
        )

        stage2_latency = time.perf_counter() - stage2_start
        _ = vr

        return {
            "predicted_answer": predicted_answer,
            "raw_output": raw_output,
            "num_visual_tokens": int(len(selected_indices)),
            "num_frames_used": int(len(selected_indices)),
            "stage1_latency_s": float(stage1_latency),
            "stage2_latency_s": float(stage2_latency),
            "num_events_detected": int(len(allocations)),
            "allocation": allocations,
        }