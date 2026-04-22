"""Integration test: end-to-end pipeline on a mock example."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.data.mock_generator import MockVideoGenerator, MockQAGenerator, generate_mock_dataset
from src.models.mock_encoder import MockCLIPEncoder
from src.models.mock_mllm import MockMLLM
from src.methods.uniform import UniformBaseline
from src.methods.oracle import OracleAllocation
from src.methods.event_aware import EventAwareMethod, EventAwareConfig


def test_generate_mock_dataset():
    """Mock dataset generator produces valid examples."""
    dataset = generate_mock_dataset(n_examples=5, seed=0)
    assert len(dataset) == 5
    for spec, example in dataset:
        assert spec.video_id == example.video_id
        assert example.answer in ["A", "B", "C", "D"]
        assert len(example.options) == 4
        assert example.gt_timestamp is not None


def test_uniform_baseline_pipeline():
    """Uniform baseline runs without errors on a mock example."""
    dataset = generate_mock_dataset(n_examples=3, seed=0)
    mllm = MockMLLM(min_frames_in_gt_event=2, seed=0)
    method = UniformBaseline(mllm, total_budget=16)

    for spec, example in dataset:
        pred = method(spec, example)
        assert pred.predicted_answer in ["A", "B", "C", "D"]
        assert pred.num_frames_used == 16


def test_oracle_pipeline():
    """Oracle allocation runs and concentrates on GT event."""
    dataset = generate_mock_dataset(n_examples=3, seed=0)
    mllm = MockMLLM(min_frames_in_gt_event=2, seed=0)
    method = OracleAllocation(mllm, total_budget=32, gt_event_share=0.7)

    for spec, example in dataset:
        pred = method(spec, example)
        # Oracle should virtually always be correct (frames targeted at GT event)
        assert pred.is_correct


def test_event_aware_pipeline():
    """End-to-end event-aware method runs without errors."""
    dataset = generate_mock_dataset(n_examples=3, seed=0)
    encoder = MockCLIPEncoder(input_dim=64, embed_dim=128, seed=0)
    mllm = MockMLLM(min_frames_in_gt_event=2, seed=0)
    vid_gen = MockVideoGenerator(embedding_dim=64, seed=42)

    method = EventAwareMethod(
        mllm=mllm, encoder=encoder, video_generator=vid_gen,
        config=EventAwareConfig(total_budget=32, n_min=2),
    )

    for spec, example in dataset:
        pred = method(spec, example)
        assert pred.predicted_answer in ["A", "B", "C", "D"]
        assert pred.num_frames_used <= 32
        assert pred.num_events > 0


def test_oracle_outperforms_uniform():
    """Sanity: oracle should beat or match uniform on average."""
    dataset = generate_mock_dataset(n_examples=30, seed=0)
    mllm = MockMLLM(min_frames_in_gt_event=2, seed=0)
    uniform = UniformBaseline(mllm, total_budget=16)
    oracle = OracleAllocation(mllm, total_budget=16, gt_event_share=0.7)

    uni_correct = sum(uniform(s, e).is_correct for s, e in dataset)
    ora_correct = sum(oracle(s, e).is_correct for s, e in dataset)

    # Oracle should be at least as good
    assert ora_correct >= uni_correct


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
