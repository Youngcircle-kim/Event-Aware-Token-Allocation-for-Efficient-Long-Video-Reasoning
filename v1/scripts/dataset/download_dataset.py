from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SUPPORTED_DATASETS = {
    "videomme": {
        "repo_id": "lmms-lab/Video-MME",
        "repo_type": "dataset",
    },
    "longvideobench": {
        "repo_id": "longvideobench/LongVideoBench",
        "repo_type": "dataset",
    },
    "longvideobench-meta": {
        "repo_id": "longvideobench/LongVideoBench-Meta",
        "repo_type": "dataset",
    },
}


def run_command(command: list[str]) -> None:
    print(f"[CMD] {' '.join(command)}")
    subprocess.run(command, check=True)


def download_hf_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    resume: bool = True,
) -> None:
    if dataset_name not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS.keys())
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: {supported}"
        )

    info = SUPPORTED_DATASETS[dataset_name]

    local_dir = output_root / dataset_name
    local_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "huggingface-cli",
        "download",
        info["repo_id"],
        "--repo-type",
        info["repo_type"],
        "--local-dir",
        str(local_dir),
    ]

    # Recent huggingface_hub versions no longer recommend symlink mode.
    command.extend(["--local-dir-use-symlinks", "False"])

    if resume:
        command.append("--resume-download")

    run_command(command)

    print(f"\nDownloaded {dataset_name} to: {local_dir}")


def extract_longvideobench(dataset_dir: Path) -> None:
    """
    LongVideoBench official instruction:
    cat videos.tar.part.* > videos.tar
    tar -xvf videos.tar
    tar -xvf subtitles.tar
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    video_parts = sorted(dataset_dir.glob("videos.tar.part.*"))
    videos_tar = dataset_dir / "videos.tar"
    subtitles_tar = dataset_dir / "subtitles.tar"

    if video_parts and not videos_tar.exists():
        print(f"[INFO] Merging {len(video_parts)} video tar parts...")
        with videos_tar.open("wb") as output_file:
            for part in video_parts:
                print(f"  - {part.name}")
                with part.open("rb") as input_file:
                    output_file.write(input_file.read())

    if videos_tar.exists():
        print("[INFO] Extracting videos.tar...")
        run_command(["tar", "-xvf", str(videos_tar), "-C", str(dataset_dir)])
    else:
        print("[WARN] videos.tar not found. Skipping video extraction.")

    if subtitles_tar.exists():
        print("[INFO] Extracting subtitles.tar...")
        run_command(["tar", "-xvf", str(subtitles_tar), "-C", str(dataset_dir)])
    else:
        print("[WARN] subtitles.tar not found. Skipping subtitle extraction.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(SUPPORTED_DATASETS.keys()),
        help="Dataset name to download.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="data/raw",
        help="Root directory to store raw datasets.",
    )

    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract dataset after download if supported.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume download.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)

    download_hf_dataset(
        dataset_name=args.dataset,
        output_root=output_root,
        resume=not args.no_resume,
    )

    dataset_dir = output_root / args.dataset

    if args.extract:
        if args.dataset == "longvideobench":
            extract_longvideobench(dataset_dir)
        else:
            print(f"[INFO] No extraction step defined for dataset: {args.dataset}")


if __name__ == "__main__":
    main()