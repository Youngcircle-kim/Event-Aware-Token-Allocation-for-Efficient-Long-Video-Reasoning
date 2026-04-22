"""
Budget sweep: measure accuracy across different T values.

Produces the accuracy-budget tradeoff curve that will be the main
figure in the paper.

Run: python scripts/budget_sweep.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mock_generator import generate_mock_dataset, MockVideoGenerator
from src.models.mock_encoder import MockCLIPEncoder
from src.models.mock_mllm import MockMLLM
from src.methods.uniform import UniformBaseline
from src.methods.event_aware import EventAwareMethod, EventAwareConfig
from src.eval.runner import run_evaluation


def main():
    print("=" * 70)
    print("BUDGET SWEEP — Accuracy vs Token Budget")
    print("=" * 70)

    dataset = generate_mock_dataset(n_examples=50, seed=42)
    encoder = MockCLIPEncoder(input_dim=64, embed_dim=128, seed=0)
    mllm = MockMLLM(min_frames_in_gt_event=2, seed=0)
    vid_gen = MockVideoGenerator(embedding_dim=64, seed=42)

    budgets = [8, 16, 32, 64, 128]

    print(f"\n{'Budget T':>10s} | {'Uniform':>10s} | {'Event-Aware':>12s} | {'Delta':>8s}")
    print("-" * 55)

    for T in budgets:
        uniform = UniformBaseline(mllm, total_budget=T)
        ours = EventAwareMethod(
            mllm=mllm, encoder=encoder, video_generator=vid_gen,
            config=EventAwareConfig(total_budget=T, n_min=2),
        )

        r_uniform = run_evaluation(uniform, dataset)
        r_ours = run_evaluation(ours, dataset)

        delta = r_ours.accuracy - r_uniform.accuracy
        print(f"{T:>10d} | {r_uniform.accuracy:>10.3f} | "
              f"{r_ours.accuracy:>12.3f} | {delta:>+8.3f}")

    print()


if __name__ == "__main__":
    main()
