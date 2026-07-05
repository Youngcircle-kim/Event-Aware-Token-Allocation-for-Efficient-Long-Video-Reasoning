"""
The proposed Event-Aware Token Allocation method.

Pipeline:
  1. Stage-1 sparse frame encoding (via MockCLIPEncoder)
  2. Event segmentation (CLIP-based boundary detection)
  3. Importance estimation (complexity x relevance)
  4. Adaptive allocation (capacity-aware, n_min-guaranteed)
  5. Frame selection within events
  6. Stage-2 reasoning (MLLM)

This entire pipeline runs with mock data today. When real models are
plugged in (TODO: REAL markers in mock_encoder.py and mock_mllm.py),
nothing in this file has to change.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from src.methods.base import BaseMethod
from src.models.mock_encoder import MockCLIPEncoder
from src.models.mock_mllm import MockMLLM
from src.modules.segmentation import segment_events
from src.modules.importance import (
    estimate_importance,
    ComplexityWeights,
)
from src.modules.allocation import allocate_frames
from src.modules.selection import select_frames
from src.data.mock_generator import MockVideoSpec, MockVideoGenerator
from src.utils.types import VideoQAExample


@dataclass
class EventAwareConfig:
    """All hyperparameters in one place for easy sweeping."""
    total_budget: int = 32
    n_min: int = 2
    # Stage-1 sampling
    stage1_fps: float = 1.0
    # Segmentation
    segmentation_window: int = 5
    segmentation_strategy: str = "percentile"
    segmentation_percentile: float = 15.0
    segmentation_threshold: Optional[float] = None
    min_event_frames: int = 3
    max_num_events: int = 64
    # Importance
    complexity_weights: ComplexityWeights = field(default_factory=ComplexityWeights)
    tau_relevance: float = 1.0
    use_question: bool = True
    # Allocation
    tau_allocation: float = 1.0
    # Frame selection within event
    selection_strategy: str = "uniform"  # "uniform" | "topk" | "hybrid"


class EventAwareMethod(BaseMethod):
    """Our proposed method."""

    def __init__(
        self,
        mllm: MockMLLM,
        encoder: MockCLIPEncoder,
        video_generator: MockVideoGenerator,
        config: Optional[EventAwareConfig] = None,
    ):
        self.config = config or EventAwareConfig()
        super().__init__(mllm, name=f"event_aware_T{self.config.total_budget}")
        self.encoder = encoder
        self.video_generator = video_generator

    def select_frames(
        self,
        spec: MockVideoSpec,
        example: VideoQAExample,
    ) -> np.ndarray:
        cfg = self.config

        # -------- Stage 1: Sparse frame latents + encoding --------
        raw_latents, timestamps = self.video_generator.render_frame_latents(
            spec, sample_fps=cfg.stage1_fps
        )
        embeddings = self.encoder.encode_frames(raw_latents)

        # -------- Mock auxiliary signals (motion, density) --------
        # In real implementation, these come from RAFT / GroundingDINO.
        # For mock, we derive them from the ground-truth spec.
        motion_per_frame, density_per_frame = self._mock_auxiliary_signals(
            spec, timestamps
        )

        # -------- Module 1: Event segmentation --------
        events = segment_events(
            embeddings=embeddings,
            timestamps=timestamps,
            video_id=spec.video_id,
            window=cfg.segmentation_window,
            strategy=cfg.segmentation_strategy,
            percentile=cfg.segmentation_percentile,
            threshold=cfg.segmentation_threshold,
            min_event_frames=cfg.min_event_frames,
            max_num_events=cfg.max_num_events,
        )

        # Guard: if segmentation failed (1 event covering whole video),
        # fall back gracefully — allocation still works.
        if len(events) == 0:
            return np.linspace(0, spec.duration, cfg.total_budget, endpoint=False)

        # -------- Module 2: Importance estimation --------
        q_emb = None
        if cfg.use_question:
            q_emb = self.encoder.encode_text(example.question)
        events = estimate_importance(
            events=events,
            frame_embeddings=embeddings,
            question_embedding=q_emb,
            per_frame_motion=motion_per_frame,
            per_frame_density=density_per_frame,
            weights=cfg.complexity_weights,
            tau_relevance=cfg.tau_relevance,
        )

        # -------- Module 3: Adaptive allocation --------
        events, alloc = allocate_frames(
            events=events,
            total_budget=cfg.total_budget,
            n_min=cfg.n_min,
            tau_allocation=cfg.tau_allocation,
        )

        # -------- Module 4: Frame selection within events --------
        # For topk / hybrid we need per-frame scores; use relevance*complexity
        # heuristic reusing event-level relevance as the per-frame prior.
        frame_scores = self._per_frame_scores(embeddings, q_emb, motion_per_frame)

        selected_indices = []
        for i, ev in enumerate(events):
            n_k = int(alloc[i])
            if n_k <= 0:
                continue
            idxs = select_frames(
                event=ev,
                n_k=n_k,
                strategy=cfg.selection_strategy,
                frame_scores=frame_scores,
            )
            selected_indices.append(idxs)

        if not selected_indices:
            return np.linspace(0, spec.duration, cfg.total_budget, endpoint=False)

        all_indices = np.concatenate(selected_indices)
        selected_timestamps = timestamps[all_indices]

        # Instrumentation
        self._last_num_events = len(events)
        self._last_allocation = alloc.tolist()

        return selected_timestamps

    # ---------------------------------------------------------------
    # Mock-mode helpers
    # ---------------------------------------------------------------
    def _mock_auxiliary_signals(self, spec: MockVideoSpec, timestamps: np.ndarray):
        """
        In mock mode, we derive motion/density from the ground-truth spec.
        # TODO: REAL — replace with RAFT and GroundingDINO calls.
        """
        N = len(timestamps)
        motion = np.zeros(N)
        density = np.zeros(N)
        for i, t in enumerate(timestamps):
            k = self.video_generator._find_event(t, spec.event_boundaries)
            motion[i] = spec.event_motion[k]
            density[i] = spec.event_object_density[k]
        # Add realistic noise
        rng = np.random.default_rng(42 + N)
        motion += rng.normal(0, 0.05, size=N)
        density += rng.normal(0, 0.3, size=N)
        return motion, density

    def _per_frame_scores(
        self,
        embeddings: np.ndarray,
        q_emb: Optional[np.ndarray],
        motion_per_frame: np.ndarray,
    ) -> np.ndarray:
        """Simple per-frame score combining relevance and motion."""
        if q_emb is None:
            return motion_per_frame.copy()
        relevance = embeddings @ q_emb  # (N,)
        # Normalize both and combine
        relevance = (relevance - relevance.min()) / (
            relevance.max() - relevance.min() + 1e-8
        )
        motion_norm = (motion_per_frame - motion_per_frame.min()) / (
            motion_per_frame.max() - motion_per_frame.min() + 1e-8
        )
        return 0.7 * relevance + 0.3 * motion_norm
