import time
from typing import Dict, Any

from src.methods.real_utils import (
    get_video_meta,
    uniform_frame_indices,
)


class UniformBaselineReal:
    def __init__(self):
        self.name = "uniform_real"

    def run(self, example, token_budget: int) -> Dict[str, Any]:
        """
        example: VideoMMELongExample
        return: dict expected by run_full_eval.py
        """
        t0 = time.perf_counter()

        vr, num_frames, fps, duration = get_video_meta(example.video_path)

        stage1_latency = 0.0

        indices = uniform_frame_indices(num_frames, token_budget)

        # TODO: real MLLM inference 연결 전까지는 dummy output
        # 현재는 파이프라인 검증용
        stage2_start = time.perf_counter()

        predicted_answer = "A"   # 임시 baseline
        raw_output = (
            f"[DUMMY] Uniform baseline selected {len(indices)} frames "
            f"from {example.video_id}. Replace with Qwen2-VL inference."
        )

        stage2_latency = time.perf_counter() - stage2_start
        _ = vr  # keep explicit for clarity

        return {
            "predicted_answer": predicted_answer,
            "raw_output": raw_output,
            "num_visual_tokens": int(len(indices)),
            "num_frames_used": int(len(indices)),
            "stage1_latency_s": float(stage1_latency),
            "stage2_latency_s": float(stage2_latency),
            "num_events_detected": 0,
            "allocation": None,
        }