"""
Evaluation runner. Runs a method over a dataset and aggregates metrics.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

from src.utils.types import Prediction
from src.methods.base import BaseMethod
from src.data.mock_generator import MockVideoSpec
from src.utils.types import VideoQAExample


@dataclass
class EvalResult:
    method_name: str
    accuracy: float
    avg_num_frames: float
    avg_num_events: float
    avg_latency_s: float
    predictions: List[Prediction]

    def summary(self) -> str:
        return (
            f"{self.method_name:25s}  "
            f"acc={self.accuracy:.3f}  "
            f"frames={self.avg_num_frames:.1f}  "
            f"events={self.avg_num_events:.1f}  "
            f"latency={self.avg_latency_s*1000:.1f}ms"
        )


def run_evaluation(
    method: BaseMethod,
    dataset: List[Tuple[MockVideoSpec, VideoQAExample]],
    verbose: bool = False,
) -> EvalResult:
    """Run `method` over every example in `dataset` and aggregate metrics."""
    preds = []
    for spec, example in dataset:
        pred = method(spec, example)
        preds.append(pred)
        if verbose:
            print(f"  {example.video_id}: "
                  f"pred={pred.predicted_answer} gt={pred.gt_answer} "
                  f"frames={pred.num_frames_used}")

    if not preds:
        return EvalResult(method.name, 0.0, 0.0, 0.0, 0.0, [])

    acc = np.mean([p.is_correct for p in preds])
    avg_frames = np.mean([p.num_frames_used for p in preds])
    avg_events = np.mean([p.num_events for p in preds])
    avg_latency = np.mean([p.latency_seconds for p in preds])

    return EvalResult(
        method_name=method.name,
        accuracy=float(acc),
        avg_num_frames=float(avg_frames),
        avg_num_events=float(avg_events),
        avg_latency_s=float(avg_latency),
        predictions=preds,
    )


def compare_methods(
    methods: List[BaseMethod],
    dataset: List[Tuple[MockVideoSpec, VideoQAExample]],
    verbose: bool = False,
) -> Dict[str, EvalResult]:
    """Evaluate each method and print a comparison table."""
    results = {}
    for m in methods:
        if verbose:
            print(f"\nEvaluating: {m.name}")
        results[m.name] = run_evaluation(m, dataset, verbose=verbose)

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Method':25s}  {'Acc':>5s}  {'Frames':>7s}  {'Events':>7s}  {'Latency':>10s}")
    print("-" * 70)
    for name, res in results.items():
        print(res.summary())
    print("=" * 70)

    return results
