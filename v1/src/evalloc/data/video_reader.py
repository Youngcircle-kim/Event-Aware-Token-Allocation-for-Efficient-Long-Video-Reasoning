from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Literal

import cv2
from PIL import Image


Backend = Literal["decord", "opencv"]


@dataclass(frozen=True)
class VideoInfo:
    video_path: Path
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int

    def to_dict(self) -> dict:
        return {
            "video_path": str(self.video_path),
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
        }


class VideoReader:
    def __init__(self, backend: Backend = "decord"):
        self.backend = backend

        if backend == "decord":
            try:
                import decord  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Decord is not installed. Install it with: pip install decord "
                    "or use backend='opencv'."
                ) from exc

    def get_video_info(self, video_path: str | Path) -> VideoInfo:
        if self.backend == "decord":
            return self._get_video_info_decord(video_path)
        if self.backend == "opencv":
            return self._get_video_info_opencv(video_path)

        raise ValueError(f"Unsupported video backend: {self.backend}")

    def read_frames(
        self,
        video_path: str | Path,
        frame_indices: Sequence[int],
        *,
        strict: bool = True,
    ) -> list[Image.Image]:
        if self.backend == "decord":
            return self._read_frames_decord(video_path, frame_indices, strict=strict)
        if self.backend == "opencv":
            return self._read_frames_opencv(video_path, frame_indices, strict=strict)

        raise ValueError(f"Unsupported video backend: {self.backend}")

    def _get_video_info_decord(self, video_path: str | Path) -> VideoInfo:
        from decord import VideoReader as DecordReader
        from decord import cpu

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        vr = DecordReader(str(video_path), ctx=cpu(0))

        total_frames = len(vr)
        fps = float(vr.get_avg_fps())

        if total_frames <= 0:
            raise RuntimeError(f"Invalid total frame count for video: {video_path}")

        if fps <= 0:
            raise RuntimeError(f"Invalid FPS value for video: {video_path}")

        first_frame = vr[0].asnumpy()
        height, width = first_frame.shape[:2]

        duration = total_frames / fps

        return VideoInfo(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            width=width,
            height=height,
        )

    def _read_frames_decord(
        self,
        video_path: str | Path,
        frame_indices: Sequence[int],
        *,
        strict: bool = True,
    ) -> list[Image.Image]:
        from decord import VideoReader as DecordReader
        from decord import cpu

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        if len(frame_indices) == 0:
            return []

        vr = DecordReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)

        safe_indices: list[int] = []

        for idx in frame_indices:
            frame_idx = int(idx)

            if frame_idx < 0 or frame_idx >= total_frames:
                message = (
                    f"Frame index {frame_idx} out of range for video {video_path}. "
                    f"Valid range: [0, {total_frames - 1}]"
                )

                if strict:
                    raise IndexError(message)

                continue

            safe_indices.append(frame_idx)

        if len(safe_indices) == 0:
            return []

        # Decord batch decoding
        batch = vr.get_batch(safe_indices).asnumpy()

        frames = [
            Image.fromarray(frame)
            for frame in batch
        ]

        return frames

    def _get_video_info_opencv(self, video_path: str | Path) -> VideoInfo:
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        if fps <= 0:
            raise RuntimeError(f"Invalid FPS value {fps} for video: {video_path}")

        if total_frames <= 0:
            raise RuntimeError(
                f"Invalid total frame count {total_frames} for video: {video_path}"
            )

        duration = total_frames / fps

        return VideoInfo(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            width=width,
            height=height,
        )

    def _read_frames_opencv(
        self,
        video_path: str | Path,
        frame_indices: Sequence[int],
        *,
        strict: bool = True,
    ) -> list[Image.Image]:
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        if len(frame_indices) == 0:
            return []

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames: list[Image.Image] = []

        for idx in frame_indices:
            frame_idx = int(idx)

            if frame_idx < 0 or frame_idx >= total_frames:
                message = (
                    f"Frame index {frame_idx} out of range for video {video_path}. "
                    f"Valid range: [0, {total_frames - 1}]"
                )

                if strict:
                    cap.release()
                    raise IndexError(message)

                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame_bgr = cap.read()

            if not success or frame_bgr is None:
                message = f"Failed to read frame {frame_idx} from video: {video_path}"

                if strict:
                    cap.release()
                    raise RuntimeError(message)

                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        cap.release()

        return frames


def uniform_sample_indices(
    total_frames: int,
    budget: int,
    *,
    include_last: bool = True,
) -> list[int]:
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")

    if budget <= 0:
        return []

    if budget >= total_frames:
        return list(range(total_frames))

    if budget == 1:
        return [total_frames // 2]

    if include_last:
        indices = [
            round(i * (total_frames - 1) / (budget - 1))
            for i in range(budget)
        ]
    else:
        step = total_frames / budget
        indices = [
            round((i + 0.5) * step)
            for i in range(budget)
        ]

    indices = [min(max(int(idx), 0), total_frames - 1) for idx in indices]

    unique_indices = sorted(set(indices))

    if len(unique_indices) < budget:
        used = set(unique_indices)
        for idx in range(total_frames):
            if idx not in used:
                unique_indices.append(idx)
                used.add(idx)
            if len(unique_indices) == budget:
                break

    return sorted(unique_indices[:budget])


def timestamps_from_indices(
    frame_indices: Sequence[int],
    fps: float,
) -> list[float]:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    return [int(idx) / fps for idx in frame_indices]