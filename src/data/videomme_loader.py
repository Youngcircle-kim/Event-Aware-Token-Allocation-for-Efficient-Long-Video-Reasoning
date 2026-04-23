"""
Load VideoMME long subset from the JSON produced by download_videomme_long.py

Returns list of VideoQAExample-compatible objects.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import json

import decord


@dataclass
class VideoMMELongExample:
    """One long-subset VideoMME example, extended with efficiency info."""
    video_id: str
    video_path: str
    duration_category: str      # always "long" here
    domain: str
    sub_category: str
    question_id: str
    task_type: str
    question: str
    options: List[str]
    answer: str                 # letter "A"/"B"/"C"/"D"
    video_duration_sec: Optional[float] = None
    gt_timestamp: Optional[Tuple[float, float]] = None  # if ever available

    @property
    def video_exists(self):
        return Path(self.video_path).exists()


def _probe_duration(video_path: str) -> Optional[float]:
    """Get actual video length using decord."""
    try:
        vr = decord.VideoReader(video_path)
        return len(vr) / vr.get_avg_fps()
    except Exception:
        return None


def load_videomme_long(
    json_path: str,
    probe_duration: bool = True,
    skip_missing_videos: bool = True,
) -> List[VideoMMELongExample]:
    """
    Load the long-subset dataset produced by download_videomme_long.py.

    Args:
        json_path: path to long_dataset.json
        probe_duration: if True, measure actual video duration using decord.
                        Adds a second or two to load time per video.
        skip_missing_videos: skip rows whose video file is missing.

    Returns:
        List of VideoMMELongExample.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    out = []
    for row in raw:
        example = VideoMMELongExample(
            video_id=row['video_id'],
            video_path=row['video_path'],
            duration_category=row['duration_category'],
            domain=row['domain'],
            sub_category=row['sub_category'],
            question_id=row['question_id'],
            task_type=row['task_type'],
            question=row['question'],
            options=row['options'],
            answer=row['answer'],
        )

        if skip_missing_videos and not example.video_exists:
            continue

        if probe_duration:
            example.video_duration_sec = _probe_duration(example.video_path)

        out.append(example)

    return out
