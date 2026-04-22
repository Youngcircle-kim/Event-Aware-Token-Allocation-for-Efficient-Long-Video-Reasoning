"""
Module 2: Event Importance Estimation.

Importance(E_k) = Complexity(E_k) * Relevance(E_k, Q)

Complexity = alpha * Motion + beta * ObjectDensity + gamma * FeatureVariance
Relevance  = softmax(Q . E_k / tau_r) over all events k

All components are min-max normalized within a single video before combining.
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from src.utils.types import Event, EventSet, FrameEmbeddings


@dataclass
class ComplexityWeights:
    """Weights for the three complexity components. Should sum to ~1."""
    alpha: float = 0.4   # motion
    beta: float = 0.3    # object density
    gamma: float = 0.3   # feature variance

    def normalized(self) -> "ComplexityWeights":
        s = self.alpha + self.beta + self.gamma
        return ComplexityWeights(self.alpha / s, self.beta / s, self.gamma / s)


# =============================================================================
# Component scores (per event)
# =============================================================================

def compute_feature_variance(
    embeddings: np.ndarray,
    event: Event,
) -> float:
    """
    Mean pairwise cosine distance within the event.

    Since embeddings are L2-normalized, cos_dist(a, b) = 1 - a.b.
    We compute average pairwise 1 - cos(f_i, f_j) for i != j.

    Fast formulation: mean = 1 - (||sum||^2 - N) / (N * (N-1))
    (derivation: expand sum over pairs using ||Σ f_i||^2 = Σ f_i.f_j)

    Returns:
        scalar in [0, 2]; higher = more diverse
    """
    event_emb = embeddings[event.start_idx:event.end_idx]
    n = len(event_emb)
    if n < 2:
        return 0.0
    s = event_emb.sum(axis=0)
    sum_sq_norm = float(np.dot(s, s))
    # Σ_{i!=j} f_i.f_j = ||Σ||^2 - Σ ||f_i||^2 = ||Σ||^2 - n  (since normalized)
    avg_dot = (sum_sq_norm - n) / (n * (n - 1))
    avg_cos_dist = 1.0 - avg_dot
    return float(avg_cos_dist)


def aggregate_motion_scores(
    per_frame_motion: Optional[np.ndarray],
    event: Event,
) -> float:
    """Average motion score within the event.

    Args:
        per_frame_motion: (N,) or None. If None, returns 0.
    """
    if per_frame_motion is None:
        return 0.0
    segment = per_frame_motion[event.start_idx:event.end_idx]
    return float(segment.mean()) if len(segment) > 0 else 0.0


def aggregate_object_density(
    per_frame_density: Optional[np.ndarray],
    event: Event,
) -> float:
    """Average object density within the event."""
    if per_frame_density is None:
        return 0.0
    segment = per_frame_density[event.start_idx:event.end_idx]
    return float(segment.mean()) if len(segment) > 0 else 0.0


# =============================================================================
# Normalization
# =============================================================================

def minmax_normalize(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize to [0, 1] within an array. Constant arrays map to 0.5."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < eps:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - s_min) / (s_max - s_min)


def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    scaled = x / tau
    scaled = scaled - scaled.max()
    ex = np.exp(scaled)
    return ex / ex.sum()


# =============================================================================
# Main entry
# =============================================================================

def estimate_importance(
    events: EventSet,
    frame_embeddings: np.ndarray,
    question_embedding: Optional[np.ndarray] = None,
    per_frame_motion: Optional[np.ndarray] = None,
    per_frame_density: Optional[np.ndarray] = None,
    weights: Optional[ComplexityWeights] = None,
    tau_relevance: float = 1.0,
    epsilon: float = 0.01,
) -> EventSet:
    """
    Compute complexity, relevance, and importance for each event.
    Mutates events in place with the computed values.

    Args:
        events: EventSet with events to score
        frame_embeddings: (N, D) frame embeddings (L2-normalized)
        question_embedding: (D,) question embedding, or None to disable relevance
        per_frame_motion: (N,) optional motion magnitudes
        per_frame_density: (N,) optional object counts
        weights: ComplexityWeights (alpha, beta, gamma)
        tau_relevance: softmax temperature for relevance
        epsilon: small constant to avoid zero importance

    Returns:
        The same EventSet (mutated).
    """
    if weights is None:
        weights = ComplexityWeights()
    weights = weights.normalized()

    K = len(events)
    if K == 0:
        return events

    # --- Complexity ---
    motion = np.array([aggregate_motion_scores(per_frame_motion, e) for e in events])
    density = np.array([aggregate_object_density(per_frame_density, e) for e in events])
    variance = np.array([compute_feature_variance(frame_embeddings, e) for e in events])

    # Normalize each component within video
    motion_n = minmax_normalize(motion)
    density_n = minmax_normalize(density)
    variance_n = minmax_normalize(variance)

    complexity = (
        weights.alpha * motion_n
        + weights.beta * density_n
        + weights.gamma * variance_n
    )

    # --- Relevance ---
    if question_embedding is None:
        # Question-free fallback: uniform relevance
        relevance = np.full(K, 1.0 / K)
    else:
        # Event representative = mean of its frame embeddings
        event_reps = np.stack([
            frame_embeddings[e.start_idx:e.end_idx].mean(axis=0)
            for e in events
        ])
        # Renormalize (mean of normalized vectors is not normalized)
        norms = np.linalg.norm(event_reps, axis=-1, keepdims=True)
        event_reps = event_reps / (norms + 1e-8)

        raw_scores = event_reps @ question_embedding  # (K,)
        relevance = softmax(raw_scores, tau=tau_relevance)

    # --- Importance = Complexity * Relevance (with floor) ---
    importance = (complexity + epsilon) * (relevance + epsilon)

    # Write back into events
    for i, ev in enumerate(events):
        ev.complexity = float(complexity[i])
        ev.relevance = float(relevance[i])
        ev.importance = float(importance[i])

    return events
