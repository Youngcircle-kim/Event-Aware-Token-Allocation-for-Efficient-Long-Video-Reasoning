from pathlib import Path
from typing import List, Sequence, Tuple

import decord
import numpy as np
from numpy.typing import NDArray
from PIL import Image

IntArray = NDArray[np.int_]


def get_video_meta(video_path: str) -> Tuple[decord.VideoReader, int, float, float]:
    vr = decord.VideoReader(str(Path(video_path)))
    num_frames = len(vr)
    fps = float(vr.get_avg_fps())
    duration = num_frames / fps if fps > 0 else 0.0
    return vr, num_frames, fps, duration


def uniform_frame_indices(num_frames: int, budget: int) -> IntArray:
    if num_frames <= 0 or budget <= 0:
        return np.array([], dtype=int)
    budget = min(num_frames, budget)
    return np.linspace(0, num_frames - 1, budget, dtype=int)


def load_frames_as_pil(video_path: str, indices: Sequence[int] | IntArray) -> List[Image.Image]:
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return []

    vr = decord.VideoReader(str(Path(video_path)))
    batch = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(arr) for arr in batch]


def fixed_interval_boundaries(num_frames: int, fps: float, stage1_stride_sec: float = 10.0) -> IntArray:
    if num_frames <= 1:
        return np.array([0, num_frames], dtype=int)

    stride = max(1, int(round(stage1_stride_sec * fps)))
    boundaries = np.arange(0, num_frames, stride, dtype=int)

    if boundaries.size == 0 or boundaries[0] != 0:
        boundaries = np.insert(boundaries, 0, 0)
    if boundaries[-1] != num_frames:
        boundaries = np.append(boundaries, num_frames)

    return boundaries.astype(int)

def merge_short_segments(
        boundaries: Sequence[int] | IntArray,
        fps: float,
        min_event_sec: float = 8.0,
) -> IntArray:
    b = np.asarray(boundaries, dtype=int)
    if b.size <= 2:
        return b

    min_len = int(round(min_event_sec * fps))
    merged = [int(b[0])]
    last = int(b[0])
    for x in b[1:]:
        x = int(x)
        if x - last >= min_len:
            merged.append(x)
            last = x
    if merged[-1] != int(b[-1]):
        merged.append(int(b[-1]))
    return np.asarray(merged, dtype=int)

def allocate_budget_by_segment_lengths(
    boundaries: Sequence[int] | IntArray,
    total_budget: int,
) -> List[int]:
    b = np.asarray(boundaries, dtype=int)

    if b.size < 2 or total_budget <= 0:
        return []

    lengths = np.diff(b)
    lengths = np.maximum(lengths, 1)
    total_len = int(lengths.sum())

    alloc = np.floor(lengths / total_len * total_budget).astype(int)

    while int(alloc.sum()) < total_budget:
        idx = int(np.argmax(lengths - alloc))
        alloc[idx] += 1

    return alloc.tolist()


def sample_indices_within_segments(
    boundaries: Sequence[int] | IntArray,
    allocations: Sequence[int],
) -> IntArray:
    b = np.asarray(boundaries, dtype=int)

    if b.size < 2:
        return np.array([], dtype=int)

    out = []
    for i, k in enumerate(allocations):
        if k <= 0:
            continue

        start = int(b[i])
        end = int(b[i + 1])
        if end <= start:
            continue

        n = min(int(k), end - start)
        idxs = np.linspace(start, end - 1, n, dtype=int)
        out.append(idxs)

    if not out:
        return np.array([], dtype=int)

    return np.unique(np.concatenate(out)).astype(int)

def limit_num_segments(
        boundaries: Sequence[int] | IntArray,
        max_segments: int = 80,
) -> IntArray:
    b = np.asarray(boundaries, dtype=int)
    if b.size <= max_segments + 1:
        return b
    selected = np.linspace(0, b.size - 1, max_segments + 1, dtype=int)
    return b[selected].astype(int)

def load_resized_frames_for_scoring(
    video_path: str,
    indices: Sequence[int] | IntArray,
    size: tuple[int, int] = (64, 64),
) -> NDArray[np.float32]:
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return np.empty((0, size[1], size[0], 3), dtype=np.float32)

    vr = decord.VideoReader(str(Path(video_path)))
    batch = vr.get_batch(idx).asnumpy()

    frames = []
    for arr in batch:
        img = Image.fromarray(arr).resize(size)
        frames.append(np.asarray(img, dtype=np.float32) / 255.0)

    return np.stack(frames, axis=0)

def compute_frame_change_scores(frames: NDArray[np.float32]) -> NDArray[np.float32]:
    if len(frames) <= 1:
        return np.array([], dtype=np.float32)

    diffs = np.mean(np.abs(frames[1:] - frames[:-1]), axis=(1, 2, 3))
    return diffs.astype(np.float32)

def moving_average(x: NDArray[np.float32], window: int = 5) -> NDArray[np.float32]:
    if x.size == 0 or window <= 1:
        return x

    window = min(window, x.size)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same").astype(np.float32)

def visual_change_event_boundaries(
    video_path: str,
    num_frames: int,
    fps: float,
    sample_stride_sec: float = 2.0,
    threshold_percentile: float = 85.0,
    min_event_sec: float = 8.0,
    max_segments: int = 80,
) -> IntArray:
    if num_frames <= 1:
        return np.array([0, num_frames], dtype=int)

    sample_stride = max(1, int(round(sample_stride_sec * fps)))
    sampled_indices = np.arange(0, num_frames, sample_stride, dtype=int)

    if sampled_indices.size == 0:
        return np.array([0, num_frames], dtype=int)

    if sampled_indices[-1] != num_frames - 1:
        sampled_indices = np.append(sampled_indices, num_frames - 1)

    frames = load_resized_frames_for_scoring(video_path, sampled_indices)

    change_scores = compute_frame_change_scores(frames)
    change_scores = moving_average(change_scores, window=5)

    if change_scores.size == 0:
        return np.array([0, num_frames], dtype=int)

    threshold = np.percentile(change_scores, threshold_percentile)

    candidate_positions = np.where(change_scores >= threshold)[0] + 1
    candidate_boundaries = sampled_indices[candidate_positions]

    boundaries = np.concatenate([
        np.array([0], dtype=int),
        candidate_boundaries.astype(int),
        np.array([num_frames], dtype=int),
    ])

    boundaries = np.unique(boundaries)
    boundaries = merge_short_segments(
        boundaries=boundaries,
        fps=fps,
        min_event_sec=min_event_sec,
    )
    boundaries = limit_num_segments(
        boundaries=boundaries,
        max_segments=max_segments,
    )

    return boundaries.astype(int)

def normalize_scores(x: NDArray[np.float32], eps: float = 1e-6) -> NDArray[np.float32]:
    x = np.asarray(x, dtype=np.float32)

    if x.size == 0:
        return x

    min_v = float(np.min(x))
    max_v = float(np.max(x))

    if max_v - min_v < eps:
        return np.ones_like(x, dtype=np.float32)

    return ((x - min_v) / (max_v - min_v + eps)).astype(np.float32)

def compute_segment_complexity_scores(
    video_path: str,
    boundaries: Sequence[int] | IntArray,
    fps: float,
    samples_per_segment: int = 4,
) -> NDArray[np.float32]:
    b = np.asarray(boundaries, dtype=int)
    scores = []

    for i in range(len(b) - 1):
        start = int(b[i])
        end = int(b[i + 1])

        if end <= start + 1:
            scores.append(1e-6)
            continue

        n = min(samples_per_segment, end - start)
        indices = np.linspace(start, end - 1, n, dtype=int)

        frames = load_resized_frames_for_scoring(video_path, indices)

        if len(frames) <= 1:
            scores.append(1e-6)
            continue

        change = compute_frame_change_scores(frames)
        motion_score = float(np.mean(change))
        variance_score = float(np.var(frames))

        score = motion_score + variance_score
        scores.append(score)

    scores = np.asarray(scores, dtype=np.float32)
    scores = normalize_scores(scores)

    return scores

def allocate_budget_by_importance(
    importance: Sequence[float] | NDArray[np.float32],
    total_budget: int,
    min_per_event: int = 0,
    temperature: float = 1.0,
) -> List[int]:
    scores = np.asarray(importance, dtype=np.float32)

    if scores.size == 0 or total_budget <= 0:
        return []

    scores = np.maximum(scores, 1e-6)

    if temperature <= 0:
        temperature = 1.0

    logits = scores / temperature
    logits = logits - np.max(logits)

    probs = np.exp(logits)
    probs = probs / np.sum(probs)

    num_events = scores.size

    if min_per_event > 0 and total_budget >= num_events * min_per_event:
        alloc = np.ones(num_events, dtype=int) * min_per_event
        remaining = total_budget - int(alloc.sum())
    else:
        alloc = np.zeros(num_events, dtype=int)
        remaining = total_budget

    raw = probs * remaining
    extra = np.floor(raw).astype(int)
    alloc += extra

    while int(alloc.sum()) < total_budget:
        residual = raw - np.floor(raw)
        idx = int(np.argmax(residual))
        alloc[idx] += 1
        raw[idx] = 0

    while int(alloc.sum()) > total_budget:
        idx = int(np.argmax(alloc))
        alloc[idx] -= 1

    return alloc.tolist()

def summarize_event_allocation(
    boundaries: Sequence[int] | IntArray,
    fps: float,
    allocations: Sequence[int],
    importance_scores: Sequence[float] | NDArray[np.float32],
    complexity_scores: Sequence[float] | NDArray[np.float32],
    top_k: int = 10,
) -> List[dict]:
    b = np.asarray(boundaries, dtype=int)
    importance = np.asarray(importance_scores, dtype=np.float32)
    complexity = np.asarray(complexity_scores, dtype=np.float32)

    rows = []

    for i in range(len(b) - 1):
        start_f = int(b[i])
        end_f = int(b[i + 1])

        rows.append({
            "event_id": i,
            "start_sec": round(start_f / fps, 2),
            "end_sec": round(end_f / fps, 2),
            "duration_sec": round((end_f - start_f) / fps, 2),
            "allocated_frames": int(allocations[i]) if i < len(allocations) else 0,
            "importance": float(importance[i]) if i < len(importance) else 0.0,
            "complexity": float(complexity[i]) if i < len(complexity) else 0.0,
        })

    rows = sorted(
        rows,
        key=lambda x: (x["allocated_frames"], x["importance"]),
        reverse=True,
    )

    return rows[:top_k]