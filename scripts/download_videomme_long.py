"""
Download VideoMME Long subset only.

VideoMME has three duration categories: short, medium, long.
Our research targets long videos (30-60min), so we download only
the chunks that contain long-duration videos.

Strategy:
1. Download annotations (~1MB) to map video_id -> duration
2. Identify which chunks contain long videos
3. Download only those chunks
4. Extract only long videos

This saves bandwidth: instead of 101GB, you download ~30-40GB.

Usage:
    python scripts/download_videomme_long.py --max_chunks 2
    
    # Minimal test (just a few long videos):
    python scripts/download_videomme_long.py --max_chunks 1 --max_videos 2
"""
import argparse
import json
import zipfile
from pathlib import Path
from typing import Set

import pandas as pd
from huggingface_hub import hf_hub_download


LOCAL_DIR = Path("./videomme_long")
LOCAL_DIR.mkdir(exist_ok=True)


# =============================================================================
# Step 1: Download annotations
# =============================================================================

def download_annotations() -> pd.DataFrame:
    print("=" * 70)
    print("[Step 1] Downloading annotations (small, <1MB)...")
    print("=" * 70)

    parquet_path = hf_hub_download(
        repo_id="lmms-lab/Video-MME",
        filename="videomme/test-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir=str(LOCAL_DIR),
    )
    df = pd.read_parquet(parquet_path)

    print(f"  Total QA examples: {len(df)}")
    print(f"  Duration distribution:")
    for dur, count in df['duration'].value_counts().items():
        print(f"    {dur}: {count}")

    return df


# =============================================================================
# Step 2: Filter long-only
# =============================================================================

def filter_long_subset(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("[Step 2] Filtering to long subset...")
    print("=" * 70)

    long_df = df[df['duration'] == 'long'].copy()

    unique_videos = long_df['video_id'].nunique()
    print(f"  Long QA examples: {len(long_df)}")
    print(f"  Unique long videos: {unique_videos}")

    # Task type distribution
    print(f"\n  Task type distribution (long subset):")
    for task, count in long_df['task_type'].value_counts().items():
        print(f"    {task}: {count}")

    # Domain distribution
    print(f"\n  Domain distribution (long subset):")
    for domain, count in long_df['domain'].value_counts().items():
        print(f"    {domain}: {count}")

    # Save the filtered annotations
    long_df.to_parquet(LOCAL_DIR / "long_annotations.parquet", index=False)
    long_df.to_json(LOCAL_DIR / "long_annotations.json",
                    orient="records", lines=True, force_ascii=False)
    print(f"\n  Saved filtered annotations to {LOCAL_DIR}/long_annotations.*")

    return long_df


# =============================================================================
# Step 3: Locate which chunks contain long videos
# =============================================================================

def download_chunks_for_long(
    long_df: pd.DataFrame,
    max_chunks: int = None,
    max_videos: int = None,
) -> Set[str]:
    """
    Download chunks incrementally until we've covered enough long videos.

    We don't know a priori which chunks contain long videos, so we download
    one at a time, extract, then see how many long videos we got.
    """
    print("\n" + "=" * 70)
    print("[Step 3] Downloading chunks incrementally...")
    print("=" * 70)

    long_video_ids = set(long_df['video_id'].unique())
    found_long = set()

    # VideoMME has 20 chunks (videos_chunked_01..20)
    for chunk_idx in range(1, 21):
        if max_chunks is not None and (chunk_idx - 1) >= max_chunks:
            print(f"\n  Reached max_chunks={max_chunks}, stopping.")
            break

        if max_videos is not None and len(found_long) >= max_videos:
            print(f"\n  Reached max_videos={max_videos}, stopping.")
            break

        chunk_filename = f"videos_chunked_{chunk_idx:02d}.zip"
        print(f"\n  Downloading {chunk_filename}...")
        print("  (This is 5-26 GB; progress shown below.)")

        try:
            zip_path = hf_hub_download(
                repo_id="lmms-lab/Video-MME",
                filename=chunk_filename,
                repo_type="dataset",
                local_dir=str(LOCAL_DIR / "chunks"),
            )
        except Exception as e:
            print(f"    Error: {e}")
            continue

        # Extract only long videos
        extracted_this_chunk = extract_long_videos_from_chunk(
            zip_path, long_video_ids, found_long, max_videos=max_videos
        )
        print(f"    Extracted {len(extracted_this_chunk)} long videos from this chunk")

        # Delete the zip to save disk
        Path(zip_path).unlink()
        print(f"    Deleted zip file ({chunk_filename})")

        print(f"    Running total of long videos: {len(found_long)}")

    return found_long


def extract_long_videos_from_chunk(
    zip_path: str,
    long_video_ids: Set[str],
    already_found: Set[str],
    max_videos: int = None,
) -> Set[str]:
    """Extract only MP4s corresponding to long video_ids."""
    extract_dir = LOCAL_DIR / "long_videos"
    extract_dir.mkdir(exist_ok=True)

    extracted = set()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            if not member.endswith('.mp4'):
                continue

            # Extract filename without path
            filename = Path(member).name
            # Strip .mp4 -> this is the video_id
            video_id = filename[:-4]

            if video_id not in long_video_ids:
                continue
            if video_id in already_found:
                continue
            if max_videos is not None and len(already_found) >= max_videos:
                break

            # Extract just this file to extract_dir
            zf.extract(member, path=LOCAL_DIR / "tmp_extract")
            src = LOCAL_DIR / "tmp_extract" / member
            dst = extract_dir / filename
            src.rename(dst)

            extracted.add(video_id)
            already_found.add(video_id)

    # Clean up the temp extraction dir
    import shutil
    tmp = LOCAL_DIR / "tmp_extract"
    if tmp.exists():
        shutil.rmtree(tmp)

    return extracted


# =============================================================================
# Step 4: Produce final matched dataset
# =============================================================================

def build_final_dataset(long_df: pd.DataFrame, found_long: Set[str]):
    print("\n" + "=" * 70)
    print("[Step 4] Building final matched dataset...")
    print("=" * 70)

    matched = long_df[long_df['video_id'].isin(found_long)].copy()
    print(f"  Videos downloaded:  {len(found_long)}")
    print(f"  QA examples matched: {len(matched)}")

    # Save with video paths
    out = []
    for _, row in matched.iterrows():
        video_path = LOCAL_DIR / "long_videos" / f"{row['video_id']}.mp4"
        out.append({
            "video_id": row['video_id'],
            "video_path": str(video_path),
            "duration_category": row['duration'],
            "domain": row['domain'],
            "sub_category": row['sub_category'],
            "question_id": row['question_id'],
            "task_type": row['task_type'],
            "question": row['question'],
            "options": list(row['options']),
            "answer": row['answer'],
        })

    json_path = LOCAL_DIR / "long_dataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Saved full dataset to: {json_path}")
    print(f"  ✓ Videos in: {LOCAL_DIR}/long_videos/")
    print(f"  ✓ Total size on disk: {sum(f.stat().st_size for f in (LOCAL_DIR/'long_videos').glob('*.mp4')) / 1e9:.2f} GB")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="Max number of chunks to download (None = all 20)")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Stop after this many long videos")
    args = parser.parse_args()

    # Step 1
    df = download_annotations()

    # Step 2
    long_df = filter_long_subset(df)

    # Step 3
    found_long = download_chunks_for_long(
        long_df,
        max_chunks=args.max_chunks,
        max_videos=args.max_videos,
    )

    # Step 4
    build_final_dataset(long_df, found_long)


if __name__ == "__main__":
    main()
