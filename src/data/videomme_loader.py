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
    duration: str
    domain: str
    sub_category: str
    url: str
    videoID: str
    question_id: str
    task_type: str
    question: str
    options: List[str]
    answer: str
    video_path: Optional[str]
    video_duration_sec: Optional[float] = None
    gt_timestamp: Optional[Tuple[float, float]] = None

    @property
    def video_exists(self):
        return Path(self.url).exists()


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
    json_path_obj = Path(json_path).resolve()
    with json_path_obj.open('r', encoding='utf-8') as f:
        data = json.load(f)
    examples = []
    project_root = json_path_obj.parent.parent
    for row in data:
        options = row.get("options", [])
        if options is None:
            options = []
        elif not isinstance(options, list):
            options = list(options)
        raw_video_path = row.get("video_path")

        video_path = None
        if raw_video_path:
            p = Path(raw_video_path)
            if p.is_absolute():
                video_path = str(p)
            else:
                video_path = str((project_root / p).resolve())
        ex = VideoMMELongExample(
            video_id=str(row.get("video_id", "")),
            duration=str(row.get("duration", "")),
            domain=str(row.get("domain", "")),
            sub_category=str(row.get("sub_category", "")),
            url=str(row.get("url", "")),
            videoID=str(row.get("videoID", "")),
            question_id=str(row.get("question_id", "")),
            task_type=str(row.get("task_type", "")),
            question=str(row.get("question", "")),
            options=options,
            answer=str(row.get("answer", "")),
            video_path=video_path,
            gt_timestamp=row.get("gt_timestamp"),
        )
        if skip_missing_videos and not ex.video_exists:
            continue
        if probe_duration and ex.video_exists and ex.video_path:
            ex.video_duration_sec = _probe_duration(ex.video_path)
        examples.append(ex)
    return examples