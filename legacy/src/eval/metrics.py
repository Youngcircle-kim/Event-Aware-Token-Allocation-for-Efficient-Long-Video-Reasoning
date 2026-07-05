"""
Metrics module for VideoMME long subset evaluation.

Covers:
  Primary:
    - QA Accuracy (overall)
    - QA Accuracy by task type
    - Visual Tokens
    - Accuracy-Budget Curve

  Secondary:
    - Inference Latency (wall-clock)
    - GPU Memory Peak
    - Token Efficiency Ratio
    - Accuracy by video duration bucket

  Optional:
    - Number of events detected
    - Allocation entropy
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from collections import defaultdict
import json
import numpy as np


@dataclass
class SampleResult:
    """Per-example record. Everything a metric might need."""
    # Identity
    video_id: str
    question_id: str
    duration_category: str       # "short" | "medium" | "long"
    task_type: str
    domain: str
    # Optional: video length in seconds
    video_duration_sec: Optional[float] = None

    # Prediction
    predicted_answer: str = "?"
    gt_answer: str = ""
    is_correct: bool = False
    raw_output: str = ""

    # Efficiency metrics
    num_visual_tokens: int = 0
    num_frames_used: int = 0
    stage1_latency_s: float = 0.0
    stage2_latency_s: float = 0.0
    total_latency_s: float = 0.0
    peak_gpu_memory_mb: float = 0.0

    # Method-specific
    method_name: str = ""
    token_budget: int = 0
    num_events_detected: int = 0
    allocation: Optional[List[int]] = None

    def to_dict(self):
        return asdict(self)


# =============================================================================
# Primary Metrics
# =============================================================================

def qa_accuracy(results: List[SampleResult]) -> float:
    """Overall QA accuracy."""
    if not results:
        return 0.0
    return sum(r.is_correct for r in results) / len(results)


def accuracy_by_task_type(results: List[SampleResult]) -> Dict[str, float]:
    """Per task-type accuracy."""
    buckets = defaultdict(list)
    for r in results:
        buckets[r.task_type].append(r.is_correct)
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def accuracy_by_domain(results: List[SampleResult]) -> Dict[str, float]:
    """Per domain accuracy (knowledge, film, sports, etc.)"""
    buckets = defaultdict(list)
    for r in results:
        buckets[r.domain].append(r.is_correct)
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def avg_visual_tokens(results: List[SampleResult]) -> float:
    """Average visual tokens used."""
    if not results:
        return 0.0
    return float(np.mean([r.num_visual_tokens for r in results]))


# =============================================================================
# Accuracy-Budget Curve (main figure of the paper)
# =============================================================================

def accuracy_budget_curve(
    results_by_budget: Dict[int, List[SampleResult]],
) -> Dict[int, float]:
    """
    Main paper figure.

    Args:
        results_by_budget: {token_budget: [results at that budget]}

    Returns:
        {token_budget: accuracy}
    """
    return {T: qa_accuracy(results) for T, results in results_by_budget.items()}


def method_comparison_at_budgets(
    results_per_method: Dict[str, Dict[int, List[SampleResult]]],
) -> Dict[str, Dict[int, float]]:
    """
    For the multi-line comparison figure.

    Args:
        results_per_method: {method_name: {budget: [results]}}

    Returns:
        {method_name: {budget: accuracy}}
    """
    return {
        method: accuracy_budget_curve(by_budget)
        for method, by_budget in results_per_method.items()
    }


# =============================================================================
# Secondary Metrics
# =============================================================================

def avg_latency(results: List[SampleResult], stage: str = "total") -> float:
    """
    Average latency in seconds.

    stage: "stage1" | "stage2" | "total"
    """
    if not results:
        return 0.0
    attr = {
        "stage1": "stage1_latency_s",
        "stage2": "stage2_latency_s",
        "total": "total_latency_s",
    }[stage]
    return float(np.mean([getattr(r, attr) for r in results]))


def avg_peak_memory(results: List[SampleResult]) -> float:
    """Average peak GPU memory in MB."""
    if not results:
        return 0.0
    return float(np.mean([r.peak_gpu_memory_mb for r in results]))


def token_efficiency_ratio(results: List[SampleResult]) -> float:
    """
    (Accuracy / Visual Tokens) × 1000.
    Higher = more efficient.
    """
    acc = qa_accuracy(results)
    tok = avg_visual_tokens(results)
    if tok == 0:
        return 0.0
    return acc / tok * 1000


def accuracy_by_duration_bucket(
    results: List[SampleResult],
    buckets_sec: List[tuple] = [(0, 1800), (1800, 2700), (2700, 3600), (3600, 7200)],
) -> Dict[str, float]:
    """
    Accuracy by video duration bucket (in seconds).

    Default buckets for VideoMME long subset (~30-60 min):
      - 0-30 min
      - 30-45 min
      - 45-60 min
      - 60+ min
    """
    bucket_results = defaultdict(list)
    for r in results:
        if r.video_duration_sec is None:
            continue
        for low, high in buckets_sec:
            if low <= r.video_duration_sec < high:
                key = f"{low/60:.0f}-{high/60:.0f}min"
                bucket_results[key].append(r.is_correct)
                break
    return {k: float(np.mean(v)) for k, v in bucket_results.items()}


# =============================================================================
# Optional / Diagnostic Metrics
# =============================================================================

def avg_events_detected(results: List[SampleResult]) -> float:
    """For methods that detect events. 0 for uniform baseline."""
    events = [r.num_events_detected for r in results if r.num_events_detected > 0]
    if not events:
        return 0.0
    return float(np.mean(events))


def allocation_entropy(results: List[SampleResult]) -> float:
    """
    Average allocation entropy across samples.

    High entropy = spread out (close to uniform)
    Low entropy  = concentrated on few events

    Useful for showing that our method produces meaningfully non-uniform
    allocations (i.e., actually using the signal).
    """
    entropies = []
    for r in results:
        if r.allocation is None or len(r.allocation) == 0:
            continue
        p = np.array(r.allocation, dtype=float)
        p = p / p.sum()
        p = p[p > 0]
        H = -np.sum(p * np.log(p))
        entropies.append(H)
    if not entropies:
        return 0.0
    return float(np.mean(entropies))


# =============================================================================
# Aggregated Report
# =============================================================================

def full_report(results: List[SampleResult]) -> Dict[str, Any]:
    """Single function that returns everything for a given method-budget pair."""
    return {
        # Primary
        "overall_accuracy": qa_accuracy(results),
        "num_samples": len(results),
        "accuracy_by_task_type": accuracy_by_task_type(results),
        "accuracy_by_domain": accuracy_by_domain(results),
        "avg_visual_tokens": avg_visual_tokens(results),

        # Secondary
        "avg_stage1_latency_s": avg_latency(results, "stage1"),
        "avg_stage2_latency_s": avg_latency(results, "stage2"),
        "avg_total_latency_s": avg_latency(results, "total"),
        "avg_peak_gpu_memory_mb": avg_peak_memory(results),
        "token_efficiency_ratio": token_efficiency_ratio(results),
        "accuracy_by_duration": accuracy_by_duration_bucket(results),

        # Optional
        "avg_events_detected": avg_events_detected(results),
        "allocation_entropy": allocation_entropy(results),
    }


def save_report(report: Dict, filepath: str):
    """Save a report dict to JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def print_report(report: Dict, title: str = "EVALUATION REPORT"):
    """Pretty-print a report."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"\nOverall accuracy: {report['overall_accuracy']:.3f} ({report['num_samples']} samples)")
    print(f"Avg visual tokens: {report['avg_visual_tokens']:.1f}")
    print(f"Token efficiency:  {report['token_efficiency_ratio']:.2f} (acc/token × 1000)")
    print(f"Avg total latency: {report['avg_total_latency_s']:.2f}s")
    print(f"Peak GPU memory:   {report['avg_peak_gpu_memory_mb']:.0f} MB")

    if report['avg_events_detected'] > 0:
        print(f"Avg events:        {report['avg_events_detected']:.1f}")
        print(f"Allocation entropy: {report['allocation_entropy']:.3f}")

    if report['accuracy_by_task_type']:
        print(f"\nAccuracy by task type:")
        for task, acc in sorted(report['accuracy_by_task_type'].items(),
                                key=lambda x: -x[1]):
            print(f"  {task:30s}: {acc:.3f}")

    if report['accuracy_by_duration']:
        print(f"\nAccuracy by duration:")
        for dur, acc in report['accuracy_by_duration'].items():
            print(f"  {dur:15s}: {acc:.3f}")
