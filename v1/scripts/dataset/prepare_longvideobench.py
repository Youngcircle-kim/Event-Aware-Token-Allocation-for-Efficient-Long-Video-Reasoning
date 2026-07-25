from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from prepare_utils import (
    find_annotation_files,
    find_video_file,
    get_first_existing,
    load_table,
    normalize_answer,
    normalize_options,
    parse_float_or_none,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        type=str,
        default="data/raw/longvideobench",
        help="Raw LongVideoBench dataset root.",
    )

    parser.add_argument(
        "--video-root",
        type=str,
        default=None,
        help="Video directory. If omitted, raw-root is used.",
    )

    parser.add_argument(
        "--annotation",
        type=str,
        default=None,
        help="Specific annotation file. If omitted, script searches automatically.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/longvideobench/all.jsonl",
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--allow-missing-video",
        action="store_true",
        help="Keep samples even if video file is not found.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max samples for debugging.",
    )

    return parser.parse_args()


def load_raw_rows(raw_root: Path, annotation: Path | None) -> list[dict[str, Any]]:
    if annotation is not None:
        print(f"[INFO] Loading annotation: {annotation}")
        return load_table(annotation)

    candidates = find_annotation_files(raw_root)

    if not candidates:
        raise FileNotFoundError(f"No annotation files found under: {raw_root}")

    print("[INFO] Found annotation candidates:")
    for path in candidates:
        print(f"  - {path}")

    preferred_keywords = [
        "longvideobench",
        "lvb",
        "test",
        "validation",
        "val",
        "dev",
        "annotation",
        "qa",
    ]

    scored: list[tuple[int, Path]] = []

    for path in candidates:
        name = path.name.lower()
        score = sum(1 for kw in preferred_keywords if kw in name)
        scored.append((score, path))

    scored.sort(key=lambda x: x[0], reverse=True)

    annotation_path = scored[0][1]
    print(f"[INFO] Auto-selected annotation: {annotation_path}")

    return load_table(annotation_path)


def extract_question(row: dict[str, Any]) -> str | None:
    # LongVideoBench may contain query/referred context fields depending on release.
    question = get_first_existing(
        row,
        ["question", "query", "problem", "qa_question"],
    )

    if question is None:
        return None

    question = str(question).strip()

    referred_context = get_first_existing(
        row,
        ["referred_context", "referring_query", "context", "subtitle_context"],
    )

    if referred_context is not None and str(referred_context).strip():
        question = f"{str(referred_context).strip()}\n\nQuestion: {question}"

    return question


def convert_row(
    row: dict[str, Any],
    video_root: Path,
    *,
    allow_missing_video: bool,
) -> dict[str, Any] | None:
    sample_id = get_first_existing(
        row,
        ["sample_id", "id", "qid", "question_id", "uid"],
    )

    video_id = get_first_existing(
        row,
        ["video_id", "videoID", "video", "video_name", "youtube_id"],
    )

    question = extract_question(row)

    raw_options = get_first_existing(
        row,
        ["options", "candidates", "choices", "answer_options"],
    )

    raw_answer = get_first_existing(
        row,
        [
            "answer",
            "correct_answer",
            "correct_choice",
            "label",
            "gt",
            "ground_truth",
            "answer_idx",
        ],
    )

    if question is None:
        print(f"[WARN] Skipping row without question: {row}")
        return None

    options = normalize_options(raw_options)
    answer = normalize_answer(raw_answer, options)

    if sample_id is None:
        sample_id_parts = []
        if video_id is not None:
            sample_id_parts.append(str(video_id))
        sample_id_parts.append(str(abs(hash(str(row)))))
        sample_id = "_".join(sample_id_parts)
    else:
        sample_id = str(sample_id)

    video_candidates = [
        get_first_existing(row, ["video_path", "video_file", "filename", "file"]),
        video_id,
        get_first_existing(row, ["video"]),
    ]

    video_path = find_video_file(video_root, [x for x in video_candidates if x is not None])

    if video_path is None:
        if not allow_missing_video:
            print(f"[WARN] Skipping sample because video file was not found: {sample_id}")
            return None
        fallback_name = str(video_id) if video_id is not None else sample_id
        video_path = str(video_root / f"{fallback_name}.mp4")

    duration_value = get_first_existing(
        row,
        ["duration", "duration_sec", "duration_seconds", "video_duration"],
    )

    duration = parse_float_or_none(duration_value)

    converted = {
        "sample_id": sample_id,
        "video_path": video_path,
        "question": question,
        "task_type": "multi_choice",
        "options": options,
        "answer": answer,
        "duration": duration,
        "metadata": {
            "dataset": "longvideobench",
            "video_id": str(video_id) if video_id is not None else None,
            "category": get_first_existing(row, ["category", "question_type", "type"]),
            "referred_context": get_first_existing(
                row,
                ["referred_context", "referring_query", "context", "subtitle_context"],
            ),
            "subtitle_path": get_first_existing(row, ["subtitle_path", "subtitle", "subtitles"]),
            "raw": row,
        },
    }

    return converted


def main() -> None:
    args = parse_args()

    raw_root = Path(args.raw_root)
    video_root = Path(args.video_root) if args.video_root else raw_root
    annotation = Path(args.annotation) if args.annotation else None
    output = Path(args.output)

    rows = load_raw_rows(raw_root, annotation)

    processed: list[dict[str, Any]] = []

    for row in rows:
        sample = convert_row(
            row,
            video_root,
            allow_missing_video=args.allow_missing_video,
        )

        if sample is None:
            continue

        processed.append(sample)

        if args.max_samples is not None and len(processed) >= args.max_samples:
            break

    write_jsonl(processed, output)

    print("\n===== Prepare LongVideoBench Done =====")
    print(f"raw_root       : {raw_root}")
    print(f"video_root     : {video_root}")
    print(f"num_raw_rows   : {len(rows)}")
    print(f"num_processed  : {len(processed)}")
    print(f"output         : {output}")


if __name__ == "__main__":
    main()