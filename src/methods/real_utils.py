from pathlib import Path
from typing import List, Tuple
import numpy as np
import decord


VIDEO_EXTS = [".mp4", ".webm", ".mkv", ".avi", ".mov"]


def load_video_reader(video_path: str) -> decord.VideoReader:
    return decord.VideoReader(video_path)


def get_video_meta(video_path: str) -> Tuple[decord.VideoReader, int, float, float]:
    vr = load_video_reader(video_path)
    num_frames = len(vr)
    fps = float(vr.get_avg_fps())
    duration = num_frames / fps if fps > 0 else 0.0
    return vr, num_frames, fps, duration


def uniform_frame_indices(num_frames: int, budget: int) -> np.ndarray:
    if num_frames <= 0 or budget <= 0:
        return np.array([], dtype=int)
    budget = min(budget, num_frames)
    return np.linspace(0, num_frames - 1, budget, dtype=int)


def timestamps_from_indices(indices: np.ndarray, fps: float) -> np.ndarray:
    if fps <= 0 or len(indices) == 0:
        return np.array([], dtype=float)
    return indices.astype(float) / fps


def simple_event_boundaries(
    num_frames: int,
    fps: float,
    stage1_stride_sec: float = 2.0,
    diff_threshold: float = 20.0,
):
    """
    Very lightweight event segmentation:
    1) sample sparse frames
    2) compute mean absolute RGB difference
    3) cut boundary if difference exceeds threshold
    """
    if num_frames <= 1:
        return [0, num_frames]

    stride = max(1, int(round(stage1_stride_sec * fps)))
    sparse_indices = np.arange(0, num_frames, stride, dtype=int)
    if sparse_indices[-1] != num_frames - 1:
        sparse_indices = np.append(sparse_indices, num_frames - 1)

    return sparse_indices


def allocate_budget_by_segment_lengths(boundaries: np.ndarray, total_budget: int) -> List[int]:
    """
    Allocate frame budget roughly proportional to segment length, with at least 1 per segment if possible.
    """
    if len(boundaries) < 2:
        return [total_budget]

    lengths = np.diff(boundaries).astype(float)
    lengths = np.maximum(lengths, 1.0)

    if total_budget <= 0:
        return [0] * len(lengths)

    base = np.zeros(len(lengths), dtype=int)

    if total_budget >= len(lengths):
        base += 1
        remaining = total_budget - len(lengths)
    else:
        # not enough budget for every event
        order = np.argsort(-lengths)
        for i in order[:total_budget]:
            base[i] = 1
        return base.tolist()

    weights = lengths / lengths.sum()
    extra = np.floor(weights * remaining).astype(int)
    base += extra

    shortfall = total_budget - base.sum()
    if shortfall > 0:
        order = np.argsort(-lengths)
        for i in order[:shortfall]:
            base[i] += 1

    return base.tolist()


def sample_indices_within_segments(boundaries: np.ndarray, allocations: List[int]) -> np.ndarray:
    out = []
    for i, n_k in enumerate(allocations):
        if n_k <= 0:
            continue
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        if end <= start:
            continue
        if end - start <= n_k:
            idxs = np.arange(start, end, dtype=int)
        else:
            idxs = np.linspace(start, end - 1, n_k, dtype=int)
        out.append(idxs)

    if not out:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(out))