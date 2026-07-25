# scripts/run_experiment.py

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

from evalloc.data.video_reader import VideoReader
from evalloc.data.videomme import VideoMMEDataset
from evalloc.evaluation.qa_metrics import AccuracyMeter
from evalloc.inference.answer_parser import MultipleChoiceAnswerParser
from evalloc.inference.prompt_builder import PromptBuilder
from evalloc.inference.qwen2_vl import Qwen2VLInferencer
from evalloc.pipeline.baseline_pipeline import UniformBaselinePipeline
from evalloc.selection.uniform import UniformFrameSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Uniform Sampling baseline on Video-MME."
    )

    parser.add_argument(
        "--annotation",
        type=str,
        required=True,
        help="Processed Video-MME JSONL annotation path.",
    )

    parser.add_argument(
        "--video-root",
        type=str,
        default=None,
        help=(
            "Optional root prepended to relative video paths in the "
            "processed annotation."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for predictions, metrics, and configuration.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=32,
        help="Number of uniformly selected frames.",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="decord",
        choices=["decord", "opencv"],
        help="Video decoding backend.",
    )

    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )

    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help=(
            "Optional Transformers attention implementation, "
            "for example flash_attention_2 or sdpa."
        ),
    )

    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample limit for debugging.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sample IDs already stored in predictions.jsonl.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failed samples and continue the experiment.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def append_jsonl(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(data, ensure_ascii=False) + "\n"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

    return records


def build_config(
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "experiment": {
            "method": "uniform",
            "seed": args.seed,
            "frame_budget": args.budget,
        },
        "dataset": {
            "name": "videomme",
            "annotation": args.annotation,
            "video_root": args.video_root,
            "max_samples": args.max_samples,
        },
        "video_reader": {
            "backend": args.backend,
        },
        "model": {
            "model_name": args.model_name,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
    }


def main() -> None:
    args = parse_args()

    if args.budget <= 0:
        raise ValueError(
            f"--budget must be positive, got {args.budget}."
        )

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "config.json"
    errors_path = output_dir / "errors.jsonl"

    config = build_config(args)
    write_json(config_path, config)

    dataset = VideoMMEDataset(
        annotation_path=args.annotation,
        video_root=args.video_root,
        require_video_exists=True,
        require_answer=True,
        max_samples=args.max_samples,
    )

    completed_records = (
        read_jsonl(predictions_path)
        if args.resume
        else []
    )

    completed_ids = {
        str(record["sample_id"])
        for record in completed_records
        if "sample_id" in record
    }

    if not args.resume:
        predictions_path.unlink(missing_ok=True)
        errors_path.unlink(missing_ok=True)

    meter = AccuracyMeter()

    # Restore metric state from previous results when resuming.
    for record in completed_records:
        meter.update(
            prediction=record.get("prediction"),
            answer=record.get("answer"),
            sample_id=record.get("sample_id"),
        )

    video_reader = VideoReader(backend=args.backend)

    frame_selector = UniformFrameSelector(
        budget=args.budget,
        include_last=True,
    )

    prompt_builder = PromptBuilder()

    inferencer = Qwen2VLInferencer(
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    answer_parser = MultipleChoiceAnswerParser()

    pipeline = UniformBaselinePipeline(
        video_reader=video_reader,
        frame_selector=frame_selector,
        prompt_builder=prompt_builder,
        inferencer=inferencer,
        answer_parser=answer_parser,
    )

    progress = tqdm(
        dataset,
        desc=f"Uniform B={args.budget}",
        total=len(dataset),
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
                parse_failures=meter.parse_failures,
            )

        except Exception as exc:
            error_record = {
                "sample_id": sample.sample_id,
                "error_type": type(exc).__name__,
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
    print(f"Method          : Uniform")
    print(f"Frame budget    : {args.budget}")
    print(f"Samples         : {meter.total}")
    print(f"Correct         : {meter.correct}")
    print(f"Accuracy        : {meter.accuracy:.4f}")
    print(f"Parse failures  : {meter.parse_failures}")
    print(f"Predictions     : {predictions_path}")
    print(f"Metrics         : {metrics_path}")


if __name__ == "__main__":
    main()