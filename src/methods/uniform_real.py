import time
from typing import Any, Dict

from src.methods.real_utils import (
    get_video_meta,
    uniform_frame_indices,
    load_frames_as_pil,
)
from src.models.qwen_vl_mcq import QwenVLMCQ


class UniformBaselineReal:
    def __init__(self, qa_model: QwenVLMCQ):
        self.name = "uniform_real"
        self.qa_model = qa_model

    def run(self, example, token_budget: int) -> Dict[str, Any]:
        _, num_frames, _, _ = get_video_meta(example.video_path)

        stage1_start = time.perf_counter()
        indices = uniform_frame_indices(num_frames, token_budget)
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
            "num_events_detected": 0,
            "allocation": None,
        }