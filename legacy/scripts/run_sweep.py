"""
Hyperparameter sweep for Event-Aware method.
Loads Qwen2-VL ONCE, then runs all hyperparameter combinations.

Usage:
    python scripts/run_sweep.py \
        --dataset videomme_long/long_dataset.json \
        --max_samples 30 \
        --budgets 16 \
        --qwen_path Qwen/Qwen2-VL-2B-Instruct \
        --output_dir ./results/sweep
"""
import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import torch

from src.data.videomme_loader import load_videomme_long
from src.eval.metrics import (
    SampleResult, full_report, save_report, print_report,
)
from src.methods.uniform_real import UniformBaselineReal
from src.methods.event_aware_real import EventAwareMethodReal
from src.models.clip_relevance import CLIPRelevanceScorer
from src.models.qwen_vl_mcq import QwenVLMCQ


# =============================================================================
# Hyperparameter grid (이 부분만 바꿔서 다른 sweep 실행)
# =============================================================================

SWEEP_CONFIG = {
    "allocation_temperature": [0.1, 0.3, 0.5, 1.0],
    "importance_mode":        ["multiply"],
    # 추가하고 싶은 hyperparameter 더 넣으면 됨
    # "threshold_percentile":   [85.0, 90.0, 95.0],
    # "min_event_sec":          [10.0, 15.0, 20.0],
}


# =============================================================================
# Helper functions
# =============================================================================

def run_single_config(
    method,
    examples,
    token_budget: int,
    method_name: str,
) -> List[SampleResult]:
    """기존 run_method_at_budget과 동일."""
    results = []
    for i, ex in enumerate(examples):
        torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        
        try:
            output = method.run(ex, token_budget=token_budget)
        except Exception as e:
            print(f"  [{i+1}/{len(examples)}] ERROR: {e}")
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
        symbol = "✓" if correct else "✗"
        print(f"  [{i+1}/{len(examples)}] {symbol}", end=" ", flush=True)
    
    print()  # newline
    return results


def make_config_id(config: Dict[str, Any]) -> str:
    """Hyperparameter dict → 문자열 ID (폴더 이름용)."""
    parts = []
    for k, v in sorted(config.items()):
        # tau0.3, mode-multiply 같은 형태로
        clean_v = str(v).replace(".", "").replace(" ", "")
        short_k = k.split("_")[0]   # allocation_temperature → allocation
        parts.append(f"{short_k}{clean_v}")
    return "_".join(parts)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[16])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--qwen_path", type=str, required=True)
    parser.add_argument("--clip_path", type=str,
                        default="openai/clip-vit-base-patch32")
    parser.add_argument("--output_dir", default="./results/sweep")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("=" * 70)
    print("Loading dataset...")
    print("=" * 70)
    examples = load_videomme_long(args.dataset)
    if args.max_samples:
        examples = examples[:args.max_samples]
    print(f"  {len(examples)} examples\n")
    
    # Load models ONCE
    print("Loading Qwen2-VL (this is the expensive step, only ONCE)...")
    qa_model = QwenVLMCQ(model_name_or_path=args.qwen_path)
    
    print("Loading CLIP scorer...")
    clip_scorer = CLIPRelevanceScorer(model_name=args.clip_path)
    
    # Generate all hyperparameter combinations
    keys = list(SWEEP_CONFIG.keys())
    values = [SWEEP_CONFIG[k] for k in keys]
    all_configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        all_configs.append(config)
    
    print(f"\n{'=' * 70}")
    print(f"SWEEP PLAN")
    print(f"{'=' * 70}")
    print(f"  Total configs: {len(all_configs)}")
    print(f"  Budgets:       {args.budgets}")
    print(f"  Total runs:    {len(all_configs) * len(args.budgets)}")
    estimated_min = len(all_configs) * len(args.budgets) * len(examples) * 2  # ~2 min/sample
    print(f"  Estimated:     ~{estimated_min} min")
    print()
    
    # Run all configs
    sweep_results = {}   # {config_id: {budget: report}}
    
    for config_idx, config in enumerate(all_configs):
        config_id = make_config_id(config)
        sweep_results[config_id] = {"config": config, "budgets": {}}
        
        print(f"\n{'=' * 70}")
        print(f"Config {config_idx+1}/{len(all_configs)}: {config_id}")
        print(f"  Hyperparams: {config}")
        print(f"{'=' * 70}")
        
        # Build event_aware method with this config
        method = EventAwareMethodReal(
            qa_model=qa_model,
            clip_scorer=clip_scorer,
            stage1_stride_sec=2.0,
            min_event_sec=15.0,
            max_segments=None,
            relevance_temperature=0.07,
            complexity_weight=0.5,
            relevance_weight=0.5,
            **config,    # hyperparameter override
        )
        
        for budget in args.budgets:
            print(f"\n  Budget T={budget}")
            results = run_single_config(method, examples, budget, "event_aware")
            
            report = full_report(results)
            sweep_results[config_id]["budgets"][budget] = {
                "accuracy": report["overall_accuracy"],
                "avg_visual_tokens": report["avg_visual_tokens"],
                "avg_total_latency_s": report["avg_total_latency_s"],
                "allocation_entropy": report["allocation_entropy"],
                "avg_events_detected": report["avg_events_detected"],
            }
            
            # Save per-config raw + report
            run_dir = output_dir / config_id
            run_dir.mkdir(exist_ok=True)
            save_report(report, str(run_dir / f"T{budget}_report.json"))
            
            with open(run_dir / f"T{budget}_raw.json", "w") as f:
                json.dump([r.to_dict() for r in results], f, indent=2,
                          ensure_ascii=False)
            
            print(f"    Accuracy: {report['overall_accuracy']:.3f}, "
                  f"Entropy: {report['allocation_entropy']:.2f}")
    
    # Save sweep summary
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(sweep_results, f, indent=2, ensure_ascii=False)
    
    # Print final comparison table
    print("\n\n" + "=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)
    print()
    
    # Header
    header = f"{'Config':<40s}"
    for b in args.budgets:
        header += f"T={b:<8d}"
    header += "Entropy"
    print(header)
    print("-" * 70)
    
    # Rows sorted by best budget accuracy
    best_budget = args.budgets[len(args.budgets)//2]   # median budget
    sorted_configs = sorted(
        sweep_results.items(),
        key=lambda x: -x[1]["budgets"].get(best_budget, {}).get("accuracy", 0)
    )
    
    for config_id, data in sorted_configs:
        row = f"{config_id:<40s}"
        last_entropy = 0
        for b in args.budgets:
            v = data["budgets"].get(b, {})
            acc = v.get("accuracy", None)
            row += f"{acc:<10.3f}" if acc is not None else f"{'--':<10s}"
            last_entropy = v.get("allocation_entropy", 0)
        row += f"{last_entropy:.2f}"
        print(row)
    
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    main()