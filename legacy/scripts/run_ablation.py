"""
Ablation study: measure the contribution of each importance component.

Tests:
  - Full (C x R)
  - Complexity-only (no relevance)
  - Relevance-only (no complexity)
  - No motion, no density, no variance (each ablation)

Run: python scripts/run_ablation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mock_generator import generate_mock_dataset, MockVideoGenerator
from src.models.mock_encoder import MockCLIPEncoder
from src.models.mock_mllm import MockMLLM
from src.methods.uniform import UniformBaseline
from src.methods.event_aware import EventAwareMethod, EventAwareConfig
from src.modules.importance import ComplexityWeights
from src.eval.runner import run_evaluation


def make_method(mllm, encoder, vid_gen, config):
    return EventAwareMethod(
        mllm=mllm, encoder=encoder, video_generator=vid_gen, config=config
    )


def main():
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    dataset = generate_mock_dataset(n_examples=50, seed=42)
    encoder = MockCLIPEncoder(input_dim=64, embed_dim=128, seed=0)
    mllm = MockMLLM(min_frames_in_gt_event=2, seed=0)
    vid_gen = MockVideoGenerator(embedding_dim=64, seed=42)
    T = 32

    configs = {
        "uniform_baseline":       None,  # special: uniform
        "full":                   EventAwareConfig(total_budget=T),
        "complexity_only":        EventAwareConfig(total_budget=T, use_question=False),
        "relevance_only":         EventAwareConfig(
            total_budget=T,
            complexity_weights=ComplexityWeights(0, 0, 1),  # just variance (degenerate)
        ),
        "no_motion":              EventAwareConfig(
            total_budget=T,
            complexity_weights=ComplexityWeights(alpha=0, beta=0.5, gamma=0.5),
        ),
        "no_density":             EventAwareConfig(
            total_budget=T,
            complexity_weights=ComplexityWeights(alpha=0.5, beta=0, gamma=0.5),
        ),
        "no_variance":            EventAwareConfig(
            total_budget=T,
            complexity_weights=ComplexityWeights(alpha=0.5, beta=0.5, gamma=0),
        ),
        "selection_topk":         EventAwareConfig(total_budget=T, selection_strategy="topk"),
        "selection_hybrid":       EventAwareConfig(total_budget=T, selection_strategy="hybrid"),
    }

    print(f"\n{'Variant':<25s} | {'Acc':>5s} | {'Frames':>7s} | {'Events':>7s}")
    print("-" * 55)

    for name, cfg in configs.items():
        if cfg is None:
            method = UniformBaseline(mllm, total_budget=T)
        else:
            method = make_method(mllm, encoder, vid_gen, cfg)

        res = run_evaluation(method, dataset)
        print(f"{name:<25s} | {res.accuracy:>5.3f} | "
              f"{res.avg_num_frames:>7.1f} | {res.avg_num_events:>7.1f}")

    print()


if __name__ == "__main__":
    main()
