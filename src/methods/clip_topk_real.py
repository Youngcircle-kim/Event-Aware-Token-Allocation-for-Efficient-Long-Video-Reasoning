"""
CLIP Top-K baseline (real-data version).

The second baseline specified in the paper's Experimental Plan, alongside
Uniform sampling. Picks the T frames whose CLIP image embeddings have the
highest cosine similarity to the question embedding — a content-aware but
event-blind selection.

If our event-aware method beats Uniform, a reviewer will immediately ask
"why not just take CLIP's top-K frames?". This baseline answers that
directly: it uses the same CLIP signal but without event structure, so
the gap between CLIP-Top-K and Event-Aware isolates the contribution of
event-level adaptive allocation.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import decord
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.methods.real_utils import get_video_meta, load_frames_as_pil
from src.models.clip_relevance import CLIPRelevanceScorer
from src.models.qwen_vl_mcq import QwenVLMCQ


class CLIPTopKBaseline:
    """
    Stage 1: score every candidate frame by CLIP(frame) · CLIP(question);
             pick the top `token_budget` frames.
    Stage 2: feed those frames to the QA MLLM.

    Args:
        qa_model: the answering MLLM (Qwen2-VL etc.).
        clip_scorer: CLIPRelevanceScorer providing encode_text / encode_images.
        candidate_stride_sec: how densely to sample candidate frames before
            ranking. Smaller = more candidates (better but slower).
        candidate_batch_size: how many frames to push through CLIP at once.
    """

    def __init__(
        self,
        qa_model: QwenVLMCQ,
        clip_scorer: CLIPRelevanceScorer,
        candidate_stride_sec: float = 2.0,
        candidate_batch_size: int = 64,
    ):
        self.name = "clip_topk"
        self.qa_model = qa_model
        self.clip_scorer = clip_scorer
        self.candidate_stride_sec = candidate_stride_sec
        self.candidate_batch_size = candidate_batch_size

    @torch.no_grad()
    def _score_candidates(
        self, video_path: str, candidate_indices: np.ndarray, question: str
    ) -> np.ndarray:
        """Cosine similarities (one per candidate frame) to the question embedding."""
        if candidate_indices.size == 0:
            return np.array([], dtype=np.float32)

        # Encode question once.
        text_feat = self.clip_scorer.encode_text(question).squeeze(0)  # [D]

        # Decode all candidate frames once.
        vr = decord.VideoReader(str(Path(video_path)))
        batch = vr.get_batch(candidate_indices.tolist()).asnumpy()
        all_pil = [Image.fromarray(arr) for arr in batch]

        # Encode in chunks so we don't blow up memory for very long videos.
        sims: List[float] = []
        for i in range(0, len(all_pil), self.candidate_batch_size):
            chunk = all_pil[i:i + self.candidate_batch_size]
            feats = self.clip_scorer.encode_images(chunk)  # [n, D], already L2-normalized
            chunk_sims = (feats @ text_feat).detach().cpu().numpy().tolist()
            sims.extend(chunk_sims)

        return np.asarray(sims, dtype=np.float32)

    def run(self, example, token_budget: int) -> Dict[str, Any]:
        _, num_frames, fps, duration = get_video_meta(example.video_path)

        stage1_start = time.perf_counter()

        # Build the candidate pool (dense uniform sampling).
        if num_frames <= 1 or fps <= 0:
            candidates = np.arange(num_frames, dtype=int)
        else:
            stride = max(1, int(round(self.candidate_stride_sec * fps)))
            candidates = np.arange(0, num_frames, stride, dtype=int)
            if candidates.size == 0 or candidates[-1] != num_frames - 1:
                candidates = np.append(candidates, num_frames - 1)

        # Score and pick top-K.
        if candidates.size > token_budget:
            sims = self._score_candidates(
                example.video_path, candidates, example.question
            )
            # argpartition then sort temporally so the MLLM sees frames in order.
            top_local = np.argpartition(-sims, token_budget)[:token_budget]
            top_local = top_local[np.argsort(-sims[top_local])][:token_budget]
            indices = np.sort(candidates[top_local])
        else:
            # Fewer candidates than budget: just use all of them.
            indices = candidates

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
            "num_events_detected": 0,
            "allocation": None,
            "sampled_indices": indices.tolist(),
        }