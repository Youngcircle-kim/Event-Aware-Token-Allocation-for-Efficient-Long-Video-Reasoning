"""
Main evaluation script for VideoMME long subset.

Runs multiple methods at multiple token budgets and produces:
  - Overall accuracy per (method, budget)
  - Task-type breakdowns
  - Visual token counts
  - Latency / memory stats
  - Accuracy-budget curve (main figure)

Usage:
    python scripts/run_full_eval.py --dataset videomme_long/long_dataset.json \
                                     --budgets 8 16 32 64 \
                                     --max_samples 50 \
                                     --output_dir ./results
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict
from src.models.clip_relevance import CLIPRelevanceScorer

import torch
from src.eval.metrics import (
    SampleResult, full_report, save_report, print_report,
    method_comparison_at_budgets,
)
from src.data.videomme_loader import load_videomme_long
from src.methods.uniform_real import UniformBaselineReal
from src.methods.event_aware_real import EventAwareMethodReal
from src.models.qwen_vl_mcq import QwenVLMCQ
from src.data.videomme_loader import load_videomme_long

def run_method_at_budget(
    method, examples: List, token_budget: int, method_name: str,
) -> List[SampleResult]:
    """Run a method on all examples at a specific token budget."""
    results = []
    for i, ex in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] {ex.video_id}...", end=" ", flush=True)

        torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()

        try:
            output = method.run(ex, token_budget=token_budget)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        total_latency = time.perf_counter() - t_start
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

        predicted = output.get("predicted_answer", "?")
        correct = predicted == ex.answer

        result = SampleResult(
            video_id=ex.video_id,
            question_id=ex.question_id,
            duration_category=ex.duration,
            task_type=ex.task_type,
            domain=ex.domain,
            video_duration_sec=ex.video_duration_sec,
            predicted_answer=predicted,
            gt_answer=ex.answer,
            is_correct=correct,
            raw_output=output.get("raw_output", ""),
            num_visual_tokens=output.get("num_visual_tokens", 0),
            num_frames_used=output.get("num_frames_used", 0),
            stage1_latency_s=output.get("stage1_latency_s", 0.0),
            stage2_latency_s=output.get("stage2_latency_s", 0.0),
            total_latency_s=total_latency,
            peak_gpu_memory_mb=peak_mem_mb,
            method_name=method_name,
            token_budget=token_budget,
            num_events_detected=output.get("num_events_detected", 0),
            allocation=output.get("allocation", None),
        )
        results.append(result)
        print(f"{'✓' if correct else '✗'} "
              f"({output.get('num_visual_tokens', 0)} tok, "
              f"{total_latency:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Path to long_dataset.json from download_videomme_long.py")
    parser.add_argument("--budgets", type=int, nargs='+', default=[8, 16, 32, 64],
                        help="Token budgets to evaluate")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of samples for quick testing")
    parser.add_argument("--methods", nargs='+',
                        default=["uniform", "event_aware"],
                        help="Methods to evaluate")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--qwen_path", type=str, required=True)
    parser.add_argument(
    "--clip_path",
    type=str,
    default="openai/clip-vit-base-patch32",
    help="CLIP model name or local path",
    )
    parser.add_argument(
        "--clip_frames_per_event",
        type=int,
        default=4,
        help="Number of frames per event for CLIP event embedding",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("=" * 70)
    print("Loading VideoMME long subset...")
    print("=" * 70)
    examples = load_videomme_long(args.dataset)
    if args.max_samples:
        examples = examples[:args.max_samples]
    print(f"Loaded {len(examples)} examples.\n")

    # Initialize methods
    # Initialize methods
    print("Initializing methods (this loads Qwen2-VL, ~1 minute)...")
    qa_model = QwenVLMCQ(model_name_or_path=args.qwen_path)
    methods = {}

    if "uniform" in args.methods:
        methods["uniform"] = UniformBaselineReal(qa_model=qa_model)

    if "event_aware" in args.methods:
        print("Initializing CLIP relevance scorer...")
        clip_scorer = CLIPRelevanceScorer(
            model_name=args.clip_path,
            frames_per_event=args.clip_frames_per_event,
        )

        methods["event_aware"] = EventAwareMethodReal(
            qa_model=qa_model,
            clip_scorer=clip_scorer,
            stage1_stride_sec=2.0,
            min_event_sec=8.0,
            max_segments=80,
            allocation_temperature=1.0,
            relevance_temperature=0.07,
            complexity_weight=0.4,
            relevance_weight=0.6,
        )

    # Run all (method, budget) combinations
    all_results = {}  # {method_name: {budget: [SampleResult]}}
    for method_name, method in methods.items():
        all_results[method_name] = {}
        for budget in args.budgets:
            print(f"\n{'=' * 70}")
            print(f"Method: {method_name}  |  Budget: T={budget}")
            print("=" * 70)
            results = run_method_at_budget(method, examples, budget, method_name)
            all_results[method_name][budget] = results

            # Per-config report
            report = full_report(results)
            report_path = output_dir / f"{method_name}_T{budget}_report.json"
            save_report(report, str(report_path))

            # Save raw per-sample results too
            raw_path = output_dir / f"{method_name}_T{budget}_raw.json"
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump([r.to_dict() for r in results], f, indent=2,
                          ensure_ascii=False)

            print_report(report, f"{method_name} @ T={budget}")

    # Summary: accuracy-budget comparison
    print("\n\n" + "=" * 70)
    print("ACCURACY-BUDGET CURVE")
    print("=" * 70)
    comp = method_comparison_at_budgets(all_results)
    print(f"\n{'Method':<20s}" + "".join(f"T={b:<8d}" for b in args.budgets))
    print("-" * 70)
    for m, curve in comp.items():
        row = f"{m:<20s}"
        for b in args.budgets:
            row += f"{curve.get(b, 0.0):<10.3f}"
        print(row)

    summary_path = output_dir / "accuracy_budget_curve.json"
    with open(summary_path, 'w') as f:
        json.dump(comp, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")


if __name__ == "__main__":
    main()
