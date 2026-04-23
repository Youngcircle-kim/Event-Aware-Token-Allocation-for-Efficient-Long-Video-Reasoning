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


def simple_event_boundaries(num_frames: int, fps: float, stage1_stride_sec: float = 2.0) -> IntArray:
    if num_frames <= 1:
        return np.array([0, num_frames], dtype=int)

    stride = max(1, int(round(stage1_stride_sec * fps)))
    boundaries = np.arange(0, num_frames, stride, dtype=int)

    if boundaries.size == 0 or boundaries[0] != 0:
        boundaries = np.insert(boundaries, 0, 0)
    if boundaries[-1] != num_frames:
        boundaries = np.append(boundaries, num_frames)

    return boundaries.astype(int)


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