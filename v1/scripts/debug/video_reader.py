from __future__ import annotations

import argparse
from pathlib import Path

from evalloc.data.video_reader import (
    VideoReader,
    uniform_sample_indices,
    timestamps_from_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default=None)

    parser.add_argument(
        "--backend",
        type=str,
        default="decord",
        choices=["decord", "opencv"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    reader = VideoReader(backend=args.backend)

    info = reader.get_video_info(video_path)

    print("===== Video Info =====")
    for key, value in info.to_dict().items():
        print(f"{key}: {value}")

    frame_indices = uniform_sample_indices(
        total_frames=info.total_frames,
        budget=args.budget,
    )

    timestamps = timestamps_from_indices(
        frame_indices=frame_indices,
        fps=info.fps,
    )

    print("\n===== Sampled Frames =====")
    for idx, ts in zip(frame_indices, timestamps):
        print(f"frame={idx}, time={ts:.2f}s")

    frames = reader.read_frames(
        video_path=video_path,
        frame_indices=frame_indices,
    )

    print(f"\nLoaded frames: {len(frames)}")

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(frames):
            frame_idx = frame_indices[i]
            save_path = save_dir / f"frame_{i:03d}_idx_{frame_idx}.jpg"
            image.save(save_path)

        print(f"Saved sampled frames to: {save_dir}")


if __name__ == "__main__":
    main()