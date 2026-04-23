"""
Prepare a VideoMME long-video subset from the annotation table.

This version reflects the actual VideoMME table structure:
- each row is a QA example
- video metadata is stored in parquet
- actual videos are referenced by YouTube URLs (not bundled mp4 chunk zips)

What this script does:
1. Download/read the VideoMME annotation parquet
2. Filter to the long subset
3. Save the filtered annotations
4. Optionally download a small number of videos with yt-dlp
5. Build long_dataset.json for downstream evaluation

Examples:
    # Just build metadata for all long examples
    python scripts/download_videomme_long.py

    # Keep only 8 long videos in the output JSON
    python scripts/download_videomme_long.py --max_videos 8

    # Download 5 real videos via yt-dlp
    python scripts/download_videomme_long.py --max_videos 5 --download_videos

    # Use a local parquet you already downloaded
    python scripts/download_videomme_long.py --parquet_path /path/to/train.parquet
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd
from huggingface_hub import hf_hub_download


LOCAL_DIR = Path("./videomme_long")
LOCAL_DIR.mkdir(exist_ok=True)
VIDEOS_DIR = LOCAL_DIR / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def parse_duration_seconds(value) -> int | None:
    """Convert duration metadata to seconds.

    Supports:
    - categorical values: short / medium / long
    - clock strings: HH:MM:SS or MM:SS
    - numeric values already in seconds
    """
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered == "short":
        return 0
    if lowered == "medium":
        return 1
    if lowered == "long":
        return 2

    if ":" in text:
        parts = text.split(":")
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            return None
        if len(nums) == 3:
            h, m, s = nums
            return h * 3600 + m * 60 + s
        if len(nums) == 2:
            m, s = nums
            return m * 60 + s

    return None


def is_long_duration(value, threshold_seconds: int) -> bool:
    parsed = parse_duration_seconds(value)
    if parsed is None:
        return False
    if parsed == 2 and str(value).strip().lower() == "long":
        return True
    if parsed in (0, 1) and str(value).strip().lower() in {"short", "medium"}:
        return False
    return parsed >= threshold_seconds


def safe_list(value):
    import numpy as np

    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, (list, tuple)):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    return [value]


def find_existing_video_file(video_stem: str) -> Path | None:
    exts = (".mp4", ".mkv", ".webm", ".mov", ".avi")
    for ext in exts:
        candidate = VIDEOS_DIR / f"{video_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# =============================================================================
# Step 1: Download or read annotations
# =============================================================================

def download_annotations(parquet_path: str | None = None) -> pd.DataFrame:
    print("=" * 70)
    print("[Step 1] Loading VideoMME annotations...")
    print("=" * 70)

    if parquet_path:
        source_path = Path(parquet_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {source_path}")
    else:
        candidates = [
            "videomme/train-00000-of-00001.parquet",
            "videomme/test-00000-of-00001.parquet",
            "train-00000-of-00001.parquet",
            "test-00000-of-00001.parquet",
        ]
        source_path = None
        errors = []
        for candidate in candidates:
            try:
                downloaded = hf_hub_download(
                    repo_id="lmms-lab/Video-MME",
                    filename=candidate,
                    repo_type="dataset",
                    local_dir=str(LOCAL_DIR),
                )
                source_path = Path(downloaded)
                print(f"  Loaded annotation file: {candidate}")
                break
            except Exception as exc:  # pragma: no cover - fallback behavior
                errors.append(f"{candidate}: {exc}")
        if source_path is None:
            joined = "\n".join(errors)
            raise RuntimeError(
                "Could not locate a VideoMME parquet file. Tried:\n" + joined
            )

    df = pd.read_parquet(source_path)
    print(f"  Source: {source_path}")
    print(f"  Total QA examples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    if "duration" in df.columns:
        print("  Duration distribution:")
        for dur, count in df["duration"].value_counts(dropna=False).items():
            print(f"    {dur}: {count}")

    return df


# =============================================================================
# Step 2: Filter long-only
# =============================================================================

def filter_long_subset(
    df: pd.DataFrame,
    max_videos: int | None = None,
    long_threshold_minutes: int = 30,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("[Step 2] Filtering to long subset...")
    print("=" * 70)

    threshold_seconds = long_threshold_minutes * 60
    long_mask = df["duration"].apply(lambda x: is_long_duration(x, threshold_seconds))
    long_df = df[long_mask].copy()

    if max_videos is not None:
        keep_ids = list(dict.fromkeys(long_df["video_id"].tolist()))[:max_videos]
        long_df = long_df[long_df["video_id"].isin(keep_ids)].copy()

    unique_videos = long_df["video_id"].nunique()
    print(f"  Long QA examples: {len(long_df)}")
    print(f"  Unique long videos: {unique_videos}")

    if "task_type" in long_df.columns:
        print("\n  Task type distribution (long subset):")
        for task, count in long_df["task_type"].value_counts().items():
            print(f"    {task}: {count}")

    if "domain" in long_df.columns:
        print("\n  Domain distribution (long subset):")
        for domain, count in long_df["domain"].value_counts().items():
            print(f"    {domain}: {count}")

    long_df.to_parquet(LOCAL_DIR / "long_annotations.parquet", index=False)
    long_df.to_json(
        LOCAL_DIR / "long_annotations.json",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    print(f"\n  Saved filtered annotations to {LOCAL_DIR}/long_annotations.*")

    return long_df


# =============================================================================
# Step 3: Optional YouTube download
# =============================================================================

def ensure_yt_dlp_installed() -> None:
    if shutil.which("yt-dlp") is None:
        raise RuntimeError(
            "yt-dlp is not installed or not on PATH. Install it with: pip install yt-dlp"
        )


def iter_unique_videos(long_df: pd.DataFrame) -> Iterable[dict]:
    seen = set()
    for _, row in long_df.iterrows():
        video_id = str(row["video_id"])
        if video_id in seen:
            continue
        seen.add(video_id)
        yield row.to_dict()


def download_videos(long_df: pd.DataFrame, max_videos: int | None = None) -> set[str]:
    print("\n" + "=" * 70)
    print("[Step 3] Downloading videos from YouTube with yt-dlp...")
    print("=" * 70)

    ensure_yt_dlp_installed()

    downloaded_ids: set[str] = set()
    failed_ids: list[str] = []

    unique_rows = list(iter_unique_videos(long_df))
    if max_videos is not None:
        unique_rows = unique_rows[:max_videos]

    for idx, row in enumerate(unique_rows, start=1):
        video_id = str(row["video_id"])
        url = str(row.get("url", "")).strip()
        youtube_id = str(row.get("videoID", "")).strip()

        if not url:
            if youtube_id:
                url = f"https://www.youtube.com/watch?v={youtube_id}"
            else:
                print(f"  [{idx}/{len(unique_rows)}] Skipping {video_id}: missing url/videoID")
                failed_ids.append(video_id)
                continue

        existing = find_existing_video_file(video_id)
        if existing is not None:
            print(f"  [{idx}/{len(unique_rows)}] Already downloaded: {existing.name}")
            downloaded_ids.add(video_id)
            continue

        print(f"  [{idx}/{len(unique_rows)}] Downloading {video_id} from {url}")
        output_template = str(VIDEOS_DIR / f"{video_id}.%(ext)s")
        cmd = [
            "yt-dlp",
            "-f",
            "mp4/bestvideo+bestaudio/best",
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
            url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            downloaded_ids.add(video_id)
        else:
            failed_ids.append(video_id)
            print(f"    Failed: {video_id}")
            stderr_preview = result.stderr.strip().splitlines()[-5:]
            for line in stderr_preview:
                print(f"      {line}")

    print(f"\n  Downloaded videos: {len(downloaded_ids)}")
    if failed_ids:
        print(f"  Failed downloads: {len(failed_ids)}")
        failed_path = LOCAL_DIR / "failed_video_ids.txt"
        failed_path.write_text("\n".join(failed_ids), encoding="utf-8")
        print(f"  Saved failed ids to: {failed_path}")

    return downloaded_ids


# =============================================================================
# Step 4: Produce final matched dataset
# =============================================================================

def build_final_dataset(long_df: pd.DataFrame, downloaded_video_ids: set[str] | None = None):
    print("\n" + "=" * 70)
    print("[Step 4] Building final dataset...")
    print("=" * 70)

    out = []
    downloaded_count = 0

    for _, row in long_df.iterrows():
        video_id = str(row["video_id"])
        local_video = find_existing_video_file(video_id)
        has_local_video = local_video is not None

        if downloaded_video_ids is not None and video_id in downloaded_video_ids:
            downloaded_count += int(has_local_video)
        elif downloaded_video_ids is None:
            downloaded_count += int(has_local_video)

        out.append({
            "video_id": video_id,
            "videoID": row.get("videoID"),
            "url": row.get("url"),
            "video_path": str(local_video) if local_video else None,
            "has_local_video": has_local_video,
            "duration": row.get("duration"),
            "domain": row.get("domain"),
            "sub_category": row.get("sub_category"),
            "question_id": row.get("question_id"),
            "task_type": row.get("task_type"),
            "question": row.get("question"),
            "options": safe_list(row.get("options")),
            "answer": row.get("answer"),
        })

    json_path = LOCAL_DIR / "long_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    total_size_bytes = sum(
        f.stat().st_size for f in VIDEOS_DIR.glob("*") if f.is_file()
    )

    print(f"  QA examples saved: {len(out)}")
    print(f"  Unique videos in subset: {long_df['video_id'].nunique()}")
    print(f"  Local videos present: {len({item['video_id'] for item in out if item['has_local_video']})}")
    print(f"\n  ✓ Saved full dataset to: {json_path}")
    print(f"  ✓ Video directory: {VIDEOS_DIR}")
    print(f"  ✓ Total video size on disk: {total_size_bytes / 1e9:.2f} GB")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_path",
        type=str,
        default=None,
        help="Path to a local VideoMME parquet file. If omitted, download from Hugging Face.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Limit the subset to this many unique long videos.",
    )
    parser.add_argument(
        "--download_videos",
        action="store_true",
        help="Actually download YouTube videos with yt-dlp.",
    )
    parser.add_argument(
        "--long_threshold_minutes",
        type=int,
        default=30,
        help="Used only when duration is clock-format like HH:MM:SS. Default: 30 minutes.",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Deprecated. Kept only for backward compatibility and ignored.",
    )
    args = parser.parse_args()

    if args.max_chunks is not None:
        print("[Info] --max_chunks is deprecated for VideoMME and will be ignored.")

    df = download_annotations(parquet_path=args.parquet_path)
    long_df = filter_long_subset(
        df,
        max_videos=args.max_videos,
        long_threshold_minutes=args.long_threshold_minutes,
    )

    downloaded_video_ids = None
    if args.download_videos:
        downloaded_video_ids = download_videos(long_df, max_videos=args.max_videos)

    build_final_dataset(long_df, downloaded_video_ids=downloaded_video_ids)


if __name__ == "__main__":
    main()
