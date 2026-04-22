"""Unit tests for importance estimation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.utils.types import Event, EventSet
from src.modules.importance import (
    compute_feature_variance,
    minmax_normalize,
    softmax,
    estimate_importance,
    ComplexityWeights,
)


def _make_events(num_events: int, frames_per_event: int = 10):
    events = []
    for i in range(num_events):
        e = Event(
            event_id=i,
            start_idx=i * frames_per_event,
            end_idx=(i + 1) * frames_per_event,
            start_time=i * 10.0,
            end_time=(i + 1) * 10.0,
        )
        events.append(e)
    return EventSet(events=events, video_id="test")


def test_feature_variance_identical_frames():
    """All identical frames -> variance = 0."""
    embeddings = np.ones((20, 8)) / np.sqrt(8)  # normalized
    event = Event(0, 0, 20, 0.0, 20.0)
    var = compute_feature_variance(embeddings, event)
    assert var < 1e-6


def test_feature_variance_diverse_frames():
    """Orthogonal frames -> variance = 1."""
    D = 4
    embeddings = np.eye(D)  # each row is a different basis vector
    event = Event(0, 0, D, 0.0, float(D))
    var = compute_feature_variance(embeddings, event)
    # All pairwise dot products = 0, so avg cos distance = 1
    assert np.isclose(var, 1.0)


def test_feature_variance_single_frame():
    """Single-frame event -> variance 0 (undefined, treated as 0)."""
    embeddings = np.ones((1, 8)) / np.sqrt(8)
    event = Event(0, 0, 1, 0.0, 1.0)
    var = compute_feature_variance(embeddings, event)
    assert var == 0.0


def test_minmax_constant():
    """Constant array normalizes to 0.5."""
    x = np.array([3.0, 3.0, 3.0])
    out = minmax_normalize(x)
    assert np.allclose(out, 0.5)


def test_minmax_range():
    """Output lies in [0, 1]."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = minmax_normalize(x)
    assert np.isclose(out.min(), 0.0)
    assert np.isclose(out.max(), 1.0)


def test_softmax_sums_to_one():
    """Softmax output must sum to 1."""
    x = np.random.randn(10)
    out = softmax(x, tau=1.0)
    assert np.isclose(out.sum(), 1.0)
    assert (out >= 0).all()


def test_softmax_temperature_effect():
    """Smaller tau -> sharper distribution."""
    x = np.array([1.0, 2.0, 3.0])
    sharp = softmax(x, tau=0.1)
    smooth = softmax(x, tau=10.0)
    # Sharp should be more peaked
    assert sharp.max() > smooth.max()


def test_estimate_importance_populates_all_fields():
    """All events should have complexity, relevance, importance filled."""
    events = _make_events(num_events=3)
    embeddings = np.random.RandomState(0).normal(size=(30, 8))
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
    question_embedding = np.random.RandomState(1).normal(size=8)
    question_embedding /= np.linalg.norm(question_embedding)

    events_out = estimate_importance(
        events=events,
        frame_embeddings=embeddings,
        question_embedding=question_embedding,
        per_frame_motion=np.ones(30),
        per_frame_density=np.ones(30),
    )

    for e in events_out:
        assert e.complexity is not None
        assert e.relevance is not None
        assert e.importance is not None
        assert 0 <= e.complexity <= 1
        assert 0 <= e.relevance <= 1
        assert e.importance > 0


def test_estimate_importance_no_question_fallback():
    """With no question, relevance is uniform."""
    events = _make_events(num_events=4)
    embeddings = np.random.RandomState(0).normal(size=(40, 8))
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)

    events_out = estimate_importance(
        events=events,
        frame_embeddings=embeddings,
        question_embedding=None,  # no question
    )

    relevances = [e.relevance for e in events_out]
    # All should be ~0.25 (uniform over 4 events)
    assert all(np.isclose(r, 0.25) for r in relevances)


def test_complexity_weights_normalization():
    """Weights are auto-normalized."""
    w = ComplexityWeights(alpha=2.0, beta=3.0, gamma=5.0)
    wn = w.normalized()
    assert np.isclose(wn.alpha + wn.beta + wn.gamma, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
