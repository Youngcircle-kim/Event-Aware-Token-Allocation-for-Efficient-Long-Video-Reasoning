"""
Pilot study: measure the upper bound of event-aware allocation.

Compares:
  - Uniform baseline (no content awareness)
  - Oracle allocation (cheats with GT event)
  - Event-Aware method (our proposed, no cheating)

If (Oracle - Uniform) gap is large, there's headroom for our method.
If (Event-Aware - Uniform) gap is a significant fraction of the oracle
gap, our method is capturing the available signal.

Run: python scripts/pilot_study.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mock_generator import generate_mock_dataset, MockVideoGenerator
from src.models.mock_encoder import MockCLIPEncoder
from src.models.mock_mllm import MockMLLM
from src.methods.uniform import UniformBaseline
from src.methods.oracle import OracleAllocation
from src.methods.event_aware import EventAwareMethod, EventAwareConfig
from src.eval.runner import compare_methods


def main():
    print("=" * 70)
    print("PILOT STUDY — Event-Aware Token Allocation")
    print("=" * 70)

    # ---- Generate mock dataset ----
    print("\n[1/3] Generating mock dataset...")
    dataset = generate_mock_dataset(n_examples=50, embedding_dim=64, seed=42)
    print(f"      Generated {len(dataset)} examples")

    # ---- Set up shared components ----
    encoder = MockCLIPEncoder(input_dim=64, embed_dim=128, seed=0)
    mllm = MockMLLM(min_frames_in_gt_event=2, seed=0)
    vid_gen = MockVideoGenerator(embedding_dim=64, seed=42)

    # ---- Define methods to compare ----
    print("\n[2/3] Initializing methods...")
    T = 32

    methods = [
        UniformBaseline(mllm, total_budget=T),
        OracleAllocation(mllm, total_budget=T, gt_event_share=0.7),
        EventAwareMethod(
            mllm=mllm,
            encoder=encoder,
            video_generator=vid_gen,
            config=EventAwareConfig(total_budget=T, n_min=2),
        ),
    ]

    # ---- Run and compare ----
    print(f"\n[3/3] Evaluating at budget T={T}...")
    results = compare_methods(methods, dataset, verbose=False)

    # ---- Gap analysis ----
    uniform_acc = results["uniform_T32"].accuracy
    oracle_acc = results["oracle_T32"].accuracy
    ours_acc = results["event_aware_T32"].accuracy

    oracle_gap = oracle_acc - uniform_acc
    ours_gap = ours_acc - uniform_acc
    capture_rate = ours_gap / oracle_gap if oracle_gap > 1e-8 else 0.0

    print("\nGAP ANALYSIS")
    print(f"  Oracle gap  (upper bound):     {oracle_gap:+.3f}")
    print(f"  Our method gap:                {ours_gap:+.3f}")
    print(f"  Capture rate (ours/oracle):    {capture_rate:.1%}")
    print()
    if oracle_gap < 0.03:
        print("  WARNING: Oracle gap is small. Limited headroom for event-aware methods.")
    elif capture_rate > 0.5:
        print("  STRONG: Our method captures >50% of the oracle gap on mock data.")
    elif capture_rate > 0.2:
        print("  OK: Our method captures 20-50% of the oracle gap. Tune hyperparameters.")
    else:
        print("  WEAK: Method captures <20% of oracle gap. Debug segmentation / importance.")
    print()


if __name__ == "__main__":
    main()
