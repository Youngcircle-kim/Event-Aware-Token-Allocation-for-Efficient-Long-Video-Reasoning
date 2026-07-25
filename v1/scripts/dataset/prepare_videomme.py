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
        default="data/raw/videomme",
        help="Raw Video-MME dataset root.",
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
        default="data/processed/videomme/all.jsonl",
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--duration-filter",
        type=str,
        default=None,
        choices=["short", "medium", "long"],
        help="Optional Video-MME duration filter.",
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

    # Prefer files that look like main annotation files.
    preferred_keywords = ["videomme", "test", "validation", "val", "annotation", "qa"]
    scored: list[tuple[int, Path]] = []

    for path in candidates:
        name = path.name.lower()
        score = sum(1 for kw in preferred_keywords if kw in name)
        scored.append((score, path))

    scored.sort(key=lambda x: x[0], reverse=True)

    annotation_path = scored[0][1]
    print(f"[INFO] Auto-selected annotation: {annotation_path}")

    return load_table(annotation_path)


def convert_row(
    row: dict[str, Any],
    video_root: Path,
    *,
    allow_missing_video: bool,
) -> dict[str, Any] | None:
    video_id = get_first_existing(
        row,
        ["video_id", "videoID", "video", "video_name", "youtube_id", "url"],
    )

    question_id = get_first_existing(
        row,
        ["question_id", "qid", "id", "sample_id"],
    )

    question = get_first_existing(
        row,
        ["question", "query", "problem"],
    )

    raw_options = get_first_existing(
        row,
        ["options", "candidates", "choices", "answer_options"],
    )

    raw_answer = get_first_existing(
        row,
        ["answer", "correct_answer", "label", "gt", "ground_truth"],
    )

    if question is None:
        print(f"[WARN] Skipping row without question: {row}")
        return None

    options = normalize_options(raw_options)
    answer = normalize_answer(raw_answer, options)

    sample_id_parts = []

    if video_id is not None:
        sample_id_parts.append(str(video_id))

    if question_id is not None:
        sample_id_parts.append(str(question_id))

    if not sample_id_parts:
        sample_id_parts.append(str(abs(hash(str(row)))))

    sample_id = "_".join(sample_id_parts)

    video_candidates = [
        get_first_existing(row, ["video_path", "video_file", "filename", "file"]),
        video_id,
        get_first_existing(row, ["videoID"]),
    ]

    video_path = find_video_file(video_root, [x for x in video_candidates if x is not None])

    if video_path is None:
        if not allow_missing_video:
            print(f"[WARN] Skipping sample because video file was not found: {sample_id}")
            return None
        video_path = str(video_root / f"{video_id}.mp4")

    duration_value = get_first_existing(row, ["duration_sec", "duration_seconds", "video_duration"])
    duration_numeric = parse_float_or_none(duration_value)

    duration_group = get_first_existing(row, ["duration", "duration_group", "length"])
    if duration_numeric is None and isinstance(duration_group, (int, float)):
        duration_numeric = float(duration_group)

    converted = {
        "sample_id": sample_id,
        "video_path": video_path,
        "question": str(question).strip(),
        "task_type": "multi_choice",
        "options": options,
        "answer": answer,
        "duration": duration_numeric,
        "metadata": {
            "dataset": "videomme",
            "video_id": str(video_id) if video_id is not None else None,
            "question_id": str(question_id) if question_id is not None else None,
            "duration_group": str(duration_group) if duration_group is not None else None,
            "domain": get_first_existing(row, ["domain"]),
            "sub_category": get_first_existing(row, ["sub_category", "subcategory"]),
            "task_type_original": get_first_existing(row, ["task_type"]),
            "url": get_first_existing(row, ["url"]),
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
        if args.duration_filter is not None:
            duration_group = get_first_existing(row, ["duration", "duration_group", "length"])
            if str(duration_group).lower() != args.duration_filter.lower():
                continue

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

    print("\n===== Prepare Video-MME Done =====")
    print(f"raw_root       : {raw_root}")
    print(f"video_root     : {video_root}")
    print(f"num_raw_rows   : {len(rows)}")
    print(f"num_processed  : {len(processed)}")
    print(f"output         : {output}")


if __name__ == "__main__":
    main()