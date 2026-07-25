from __future__ import annotations

import csv
import json
import numpy as np
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas and pyarrow are required to read parquet files. "
            "Install with: pip install pandas pyarrow"
        ) from exc
    df = pd.read_parquet(path)
    df.columns = [str(col) for col in df.columns]
    records = df.to_dict(orient="records")
    return [
        {str(key): value for key, value in row.items()}
        for row in records
    ]


def load_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = read_json(path)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Common patterns: {"data": [...]}, {"annotations": [...]}
            for key in ["data", "annotations", "items", "questions", "examples"]:
                if key in data and isinstance(data[key], list):
                    return data[key]

            # Fallback: one dict as one row
            return [data]

        raise ValueError(f"Unsupported JSON structure: {path}")

    if suffix == ".jsonl":
        return read_jsonl(path)

    if suffix == ".csv":
        return read_csv(path)

    if suffix == ".parquet":
        return read_parquet(path)

    raise ValueError(f"Unsupported annotation file type: {path}")


def write_jsonl(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            serializable_row = to_json_serializable(row)

            f.write(
                json.dumps(
                    serializable_row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def find_annotation_files(root: Path) -> list[Path]:
    exts = {".json", ".jsonl", ".csv", ".parquet"}
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in exts
    )


def find_video_file(video_root: Path, candidates: list[str]) -> str | None:
    video_exts = [".mp4", ".mkv", ".webm", ".avi", ".mov"]

    normalized_candidates: list[str] = []

    for candidate in candidates:
        if candidate is None:
            continue

        candidate = str(candidate).strip()

        if not candidate:
            continue

        normalized_candidates.append(candidate)

        stem = Path(candidate).stem
        if stem:
            normalized_candidates.append(stem)

    for candidate in normalized_candidates:
        candidate_path = Path(candidate)

        # If candidate already includes extension/path
        direct_path = video_root / candidate_path
        if direct_path.exists() and direct_path.is_file():
            return str(direct_path)

        # Try candidate as stem
        for ext in video_exts:
            maybe = video_root / f"{candidate}{ext}"
            if maybe.exists() and maybe.is_file():
                return str(maybe)

            maybe_recursive = list(video_root.rglob(f"{candidate}{ext}"))
            if maybe_recursive:
                return str(maybe_recursive[0])

    return None


def normalize_options(raw_options: Any) -> list[str] | None:
    if raw_options is None:
        return None

    try:
        import numpy as np

        if isinstance(raw_options, np.ndarray):
            raw_options = raw_options.tolist()
    except ImportError:
        pass

    if isinstance(raw_options, (list, tuple)):
        options = [
            str(option).strip()
            for option in raw_options
            if str(option).strip()
        ]
        return strip_option_labels(options)

    if isinstance(raw_options, str):
        text = raw_options.strip()

        if not text:
            return None

        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                options = [
                    str(option).strip()
                    for option in parsed
                    if str(option).strip()
                ]
                return strip_option_labels(options)

        except json.JSONDecodeError:
            pass

        if "\n" in text:
            options = [
                option.strip()
                for option in text.splitlines()
                if option.strip()
            ]
            return strip_option_labels(options)

        if "|" in text:
            options = [
                option.strip()
                for option in text.split("|")
                if option.strip()
            ]
            return strip_option_labels(options)

        return [text]

    return None


def strip_option_labels(options: list[str]) -> list[str]:
    cleaned: list[str] = []

    for option in options:
        option = option.strip()

        # Remove common prefixes: "A. text", "A) text", "(A) text"
        if len(option) >= 3:
            if option[0].upper() in "ABCDE" and option[1] in [".", ")", ":"]:
                option = option[2:].strip()
            elif (
                option[0] == "("
                and len(option) >= 4
                and option[1].upper() in "ABCDE"
                and option[2] == ")"
            ):
                option = option[3:].strip()

        cleaned.append(option)

    return cleaned


def normalize_answer(raw_answer: Any, options: list[str] | None = None) -> str | None:
    if raw_answer is None:
        return None

    answer = str(raw_answer).strip()

    if not answer:
        return None

    # Already A/B/C/D/E
    if len(answer) == 1 and answer.upper() in "ABCDE":
        return answer.upper()

    # Numeric index
    if answer.isdigit():
        idx = int(answer)

        # Some datasets use 0-based index
        if options is not None and 0 <= idx < len(options):
            return chr(ord("A") + idx)

        # Some datasets use 1-based index
        if options is not None and 1 <= idx <= len(options):
            return chr(ord("A") + idx - 1)

    # "C. something"
    if answer[0].upper() in "ABCDE" and len(answer) >= 2 and answer[1] in [".", ")", ":"]:
        return answer[0].upper()

    # If answer text matches one of options
    if options is not None:
        answer_lower = answer.lower()
        for i, option in enumerate(options):
            if answer_lower == option.lower():
                return chr(ord("A") + i)

    return answer


def get_first_existing(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def parse_float_or_none(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def to_json_serializable(value: Any) -> Any:
    """
    Recursively convert NumPy/Pandas values into JSON-serializable
    Python objects.
    """
    if isinstance(value, np.ndarray):
        return [
            to_json_serializable(item)
            for item in value.tolist()
        ]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): to_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            to_json_serializable(item)
            for item in value
        ]

    return value