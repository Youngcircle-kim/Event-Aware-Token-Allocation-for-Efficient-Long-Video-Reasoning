from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from evalloc.allocation.softmax import SoftmaxAllocator
from evalloc.data.video_reader import VideoReader
from evalloc.data.videomme import VideoMMEDataset
from evalloc.evaluation.qa_metrics import AccuracyMeter
from evalloc.features.clip_extractor import CLIPFeatureExtractor
from evalloc.inference.answer_parser import MultipleChoiceAnswerParser
from evalloc.inference.prompt_builder import PromptBuilder
from evalloc.inference.qwen2_vl import Qwen2VLInferencer
from evalloc.pipeline.event_pipeline import EventAwarePipeline
from evalloc.scoring.base import BaseEventScorer
from evalloc.scoring.combined import CombinedScorer
from evalloc.scoring.complexity import ComplexityScorer
from evalloc.scoring.duration import DurationScorer
from evalloc.scoring.random import RandomScorer
from evalloc.scoring.relevance import RelevanceScorer
from evalloc.segmentation.semantic_segmenter import SemanticSegmenter
from evalloc.selection.diverse import RelevanceDiverseSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Event-aware frame selection on Video-MME."
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    # ---------------------------------------------------------
    # Final frame budget
    # ---------------------------------------------------------
    parser.add_argument(
        "--frame-budget",
        type=int,
        required=True,
        help="Number of frames finally passed to Qwen2-VL.",
    )

    # ---------------------------------------------------------
    # Candidate-frame extraction
    # ---------------------------------------------------------
    parser.add_argument(
        "--candidate-stride-sec",
        type=float,
        required=True,
        help="Temporal interval between CLIP candidate frames.",
    )

    parser.add_argument(
        "--candidate-decode-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--video-backend",
        type=str,
        default="decord",
        choices=["decord", "opencv"],
    )

    # ---------------------------------------------------------
    # CLIP
    # ---------------------------------------------------------
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-large-patch14",
    )

    parser.add_argument(
        "--clip-device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--clip-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
    )

    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=16,
    )

    # ---------------------------------------------------------
    # Semantic segmentation
    # ---------------------------------------------------------
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=85.0,
    )

    parser.add_argument(
        "--local-max-radius",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--min-event-sec",
        type=float,
        default=15.0,
    )

    # ---------------------------------------------------------
    # Event scoring
    # ---------------------------------------------------------
    parser.add_argument(
        "--scorer",
        type=str,
        default="combined",
        choices=[
            "relevance",
            "complexity",
            "combined",
            "duration",
            "random",
        ],
    )

    parser.add_argument(
        "--relevance-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--complexity-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--motion-weight",
        type=float,
        default=1.0 / 3.0,
    )

    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=1.0 / 3.0,
    )

    parser.add_argument(
        "--variance-weight",
        type=float,
        default=1.0 / 3.0,
    )

    # ---------------------------------------------------------
    # Allocation
    # ---------------------------------------------------------
    parser.add_argument(
        "--allocation-temperature",
        type=float,
        default=0.3,
    )

    # ---------------------------------------------------------
    # Within-event selection
    # ---------------------------------------------------------
    parser.add_argument(
        "--selection-relevance-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--selection-diversity-weight",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--selection-temporal-weight",
        type=float,
        default=0.1,
    )

    # ---------------------------------------------------------
    # Qwen2-VL
    # ---------------------------------------------------------
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
    )

    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        choices=[
            "auto",
            "float16",
            "bfloat16",
            "float32",
        ],
    )

    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
    )

    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Minimum pixels per visual input.",
    )

    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Maximum pixels per visual input.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
    )

    # ---------------------------------------------------------
    # Experiment control
    # ---------------------------------------------------------
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    return parser.parse_args()


def resolve_torch_dtype(
    dtype_name: str,
) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported CLIP dtype: {dtype_name}"
        )

    return mapping[dtype_name]


def build_scorer(
    args: argparse.Namespace,
) -> BaseEventScorer:
    if args.scorer == "relevance":
        return RelevanceScorer(
            normalize=True,
        )

    if args.scorer == "complexity":
        return ComplexityScorer(
            motion_weight=args.motion_weight,
            diversity_weight=args.diversity_weight,
            variance_weight=args.variance_weight,
            normalize_components=True,
        )

    if args.scorer == "combined":
        return CombinedScorer(
            relevance_weight=args.relevance_weight,
            complexity_weight=args.complexity_weight,
            relevance_scorer=RelevanceScorer(
                normalize=True,
            ),
            complexity_scorer=ComplexityScorer(
                motion_weight=args.motion_weight,
                diversity_weight=args.diversity_weight,
                variance_weight=args.variance_weight,
                normalize_components=True,
            ),
        )

    if args.scorer == "duration":
        return DurationScorer(
            normalize=True,
        )

    if args.scorer == "random":
        return RandomScorer(
            seed=args.seed,
        )

    raise ValueError(
        f"Unsupported scorer: {args.scorer}"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def read_completed_ids(
    predictions_path: Path,
) -> set[str]:
    if not predictions_path.exists():
        return set()

    completed_ids: set[str] = set()

    with predictions_path.open(
        encoding="utf-8"
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if "sample_id" in record:
                completed_ids.add(
                    str(record["sample_id"])
                )

    return completed_ids


def build_config(
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "videomme",
            "annotation": str(args.annotation),
            "video_root": (
                str(args.video_root)
                if args.video_root is not None
                else None
            ),
            "max_samples": args.max_samples,
        },
        "selection": {
            "method": "event_aware",
            "frame_budget": args.frame_budget,
            "candidate_stride_sec": (
                args.candidate_stride_sec
            ),
            "candidate_decode_batch_size": (
                args.candidate_decode_batch_size
            ),
        },
        "clip": {
            "model": args.clip_model,
            "device": args.clip_device,
            "dtype": args.clip_dtype,
            "batch_size": args.clip_batch_size,
        },
        "segmentation": {
            "window_size": args.window_size,
            "threshold_percentile": (
                args.threshold_percentile
            ),
            "local_max_radius": (
                args.local_max_radius
            ),
            "min_event_sec": args.min_event_sec,
        },
        "scoring": {
            "type": args.scorer,
            "relevance_weight": (
                args.relevance_weight
            ),
            "complexity_weight": (
                args.complexity_weight
            ),
            "motion_weight": args.motion_weight,
            "diversity_weight": (
                args.diversity_weight
            ),
            "variance_weight": (
                args.variance_weight
            ),
        },
        "allocation": {
            "temperature": (
                args.allocation_temperature
            ),
        },
        "within_event_selection": {
            "relevance_weight": (
                args.selection_relevance_weight
            ),
            "diversity_weight": (
                args.selection_diversity_weight
            ),
            "temporal_weight": (
                args.selection_temporal_weight
            ),
        },
        "model": {
            "name": args.model_name,
            "torch_dtype": args.torch_dtype,
            "attention": (
                args.attn_implementation
            ),
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_new_tokens": (
                args.max_new_tokens
            ),
        },
        "experiment": {
            "seed": args.seed,
        },
    }


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = (
        args.output_dir / "config.json"
    )
    predictions_path = (
        args.output_dir / "predictions.jsonl"
    )
    metrics_path = (
        args.output_dir / "metrics.json"
    )
    errors_path = (
        args.output_dir / "errors.jsonl"
    )

    write_json(
        config_path,
        build_config(args),
    )

    if not args.resume:
        predictions_path.unlink(
            missing_ok=True
        )
        metrics_path.unlink(
            missing_ok=True
        )
        errors_path.unlink(
            missing_ok=True
        )

    completed_ids = (
        read_completed_ids(predictions_path)
        if args.resume
        else set()
    )

    dataset = VideoMMEDataset(
        annotation_path=args.annotation,
        video_root=args.video_root,
        require_video_exists=True,
        require_answer=True,
        max_samples=args.max_samples,
    )

    scorer = build_scorer(args)

    pipeline = EventAwarePipeline(
        video_reader=VideoReader(
            backend=args.video_backend,
        ),
        feature_extractor=CLIPFeatureExtractor(
            model_name=args.clip_model,
            device=args.clip_device,
            dtype=resolve_torch_dtype(
                args.clip_dtype
            ),
            batch_size=args.clip_batch_size,
        ),
        segmenter=SemanticSegmenter(
            window_size=args.window_size,
            threshold_percentile=(
                args.threshold_percentile
            ),
            local_max_radius=(
                args.local_max_radius
            ),
            min_event_sec=args.min_event_sec,
        ),
        scorer=scorer,
        allocator=SoftmaxAllocator(
            temperature=(
                args.allocation_temperature
            ),
        ),
        selector=RelevanceDiverseSelector(
            relevance_weight=(
                args.selection_relevance_weight
            ),
            diversity_weight=(
                args.selection_diversity_weight
            ),
            temporal_weight=(
                args.selection_temporal_weight
            ),
        ),
        prompt_builder=PromptBuilder(),
        inferencer=Qwen2VLInferencer(
            model_name=args.model_name,
            torch_dtype=args.torch_dtype,
            attn_implementation=(
                args.attn_implementation
            ),
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_new_tokens=(
                args.max_new_tokens
            ),
            do_sample=False,
        ),
        answer_parser=(
            MultipleChoiceAnswerParser()
        ),
        frame_budget=args.frame_budget,
        candidate_stride_sec=(
            args.candidate_stride_sec
        ),
        candidate_decode_batch_size=(
            args.candidate_decode_batch_size
        ),
    )

    meter = AccuracyMeter()

    progress = tqdm(
        dataset,
        total=len(dataset),
        desc=(
            f"Event-aware "
            f"{args.scorer} "
            f"B={args.frame_budget}"
        ),
    )

    for sample in progress:
        if sample.sample_id in completed_ids:
            continue

        try:
            result = pipeline.run(sample)
            record = result.to_dict()

            append_jsonl(
                predictions_path,
                record,
            )

            meter.update(
                prediction=result.prediction,
                answer=result.answer,
                sample_id=result.sample_id,
            )

            write_json(
                metrics_path,
                meter.to_dict(),
            )

            progress.set_postfix(
                accuracy=f"{meter.accuracy:.4f}",
                parse_failures=(
                    meter.parse_failures
                ),
            )

        except Exception as exc:
            error_record = {
                "sample_id": sample.sample_id,
                "error_type": (
                    type(exc).__name__
                ),
                "error_message": str(exc),
            }

            append_jsonl(
                errors_path,
                error_record,
            )

            if not args.continue_on_error:
                raise

            print(
                f"\n[ERROR] {sample.sample_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    write_json(
        metrics_path,
        meter.to_dict(),
    )

    print("\n===== Experiment Complete =====")
    print(f"Scorer       : {args.scorer}")
    print(
        f"Frame budget : {args.frame_budget}"
    )
    print(
        f"Stride       : "
        f"{args.candidate_stride_sec}s"
    )
    print(f"Samples      : {meter.total}")
    print(f"Correct      : {meter.correct}")
    print(
        f"Accuracy     : "
        f"{meter.accuracy:.4f}"
    )
    print(
        f"Predictions  : {predictions_path}"
    )
    print(f"Metrics      : {metrics_path}")


if __name__ == "__main__":
    main()