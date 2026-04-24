import time
from typing import Any, Dict

import numpy as np

from src.methods.real_utils import (
    get_video_meta,
    visual_change_event_boundaries,
    compute_segment_complexity_scores,
    normalize_scores,
    allocate_budget_by_importance,
    sample_indices_within_segments,
    summarize_event_allocation,
    load_frames_as_pil,
)
from src.models.clip_relevance import CLIPRelevanceScorer
from src.models.qwen_vl_mcq import QwenVLMCQ


class EventAwareMethodReal:
    def __init__(
        self,
        qa_model: QwenVLMCQ,
        clip_scorer: CLIPRelevanceScorer,
        stage1_stride_sec: float = 2.0,
        min_event_sec: float = 8.0,
        max_segments: int = 80,
        allocation_temperature: float = 1.0,
        relevance_temperature: float = 0.07,
        complexity_weight: float = 0.5,
        relevance_weight: float = 0.5,
    ):
        self.name = "event_aware_clip_relevance"
        self.stage1_stride_sec = stage1_stride_sec
        self.min_event_sec = min_event_sec
        self.max_segments = max_segments
        self.allocation_temperature = allocation_temperature
        self.relevance_temperature = relevance_temperature
        self.complexity_weight = complexity_weight
        self.relevance_weight = relevance_weight
        self.qa_model = qa_model
        self.clip_scorer = clip_scorer

    def run(self, example, token_budget: int) -> Dict[str, Any]:
        _, num_frames, fps, duration = get_video_meta(example.video_path)

        stage1_start = time.perf_counter()

        boundaries = visual_change_event_boundaries(
            video_path=example.video_path,
            num_frames=num_frames,
            fps=fps,
            sample_stride_sec=self.stage1_stride_sec,
            threshold_percentile=85.0,
            min_event_sec=self.min_event_sec,
            max_segments=self.max_segments,
        )

        complexity_scores = compute_segment_complexity_scores(
            video_path=example.video_path,
            boundaries=boundaries,
            fps=fps,
            samples_per_segment=4,
        )
        event_embeddings = self.clip_scorer.compute_event_embeddings(
            video_path=example.video_path,
            boundaries=boundaries,
        )

        relevance_scores = self.clip_scorer.compute_query_relevance(
            question=example.question,
            options=example.options,
            event_embeddings=event_embeddings,
            temperature=self.relevance_temperature,
        )

        relevance_scores = normalize_scores(relevance_scores)
        complexity_scores = normalize_scores(complexity_scores)

        importance_scores = (
            self.complexity_weight * complexity_scores
            + self.relevance_weight * relevance_scores
        )

        importance_scores = normalize_scores(importance_scores)

        num_events = len(boundaries) - 1
        min_per_event = 1 if num_events <= token_budget else 0

        allocations = allocate_budget_by_importance(
            importance=importance_scores,
            total_budget=token_budget,
            min_per_event=min_per_event,
            temperature=self.allocation_temperature,
        )

        indices = sample_indices_within_segments(boundaries, allocations)

        top_events = summarize_event_allocation(
            boundaries=boundaries,
            fps=fps,
            allocations=allocations,
            importance_scores=importance_scores,
            complexity_scores=complexity_scores,
            top_k=10,
        )

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
            "video_duration_s": float(duration),
            "num_events_detected": int(num_events),
            "boundaries": boundaries.tolist(),
            "allocation": allocations,
            "complexity_scores": complexity_scores.tolist(),
            "relevance_scores": relevance_scores.tolist(),
            "importance_scores": importance_scores.tolist(),
            "sampled_indices": indices.tolist(),
            "top_events": top_events,
        }