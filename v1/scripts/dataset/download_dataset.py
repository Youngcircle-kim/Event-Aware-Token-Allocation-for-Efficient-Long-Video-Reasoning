from __future__ import annotations

import argparse
import shutil
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


def resolve_hf_cli() -> str:
    """
    Resolve the available Hugging Face CLI command.

    New huggingface_hub versions provide `hf`.
    Older versions may provide `huggingface-cli`.
    """
    if shutil.which("hf") is not None:
        return "hf"

    if shutil.which("huggingface-cli") is not None:
        return "huggingface-cli"

    raise RuntimeError(
        "Hugging Face CLI was not found. "
        "Install it with: python -m pip install -U huggingface_hub"
    )


def download_hf_dataset(
    dataset_name: str,
    output_root: Path,
) -> Path:
    if dataset_name not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: {supported}"
        )

    info = SUPPORTED_DATASETS[dataset_name]

    output_root = output_root.expanduser().resolve()
    local_dir = output_root / dataset_name
    local_dir.mkdir(parents=True, exist_ok=True)

    hf_cli = resolve_hf_cli()

    command = [
        hf_cli,
        "download",
        info["repo_id"],
        "--repo-type",
        info["repo_type"],
        "--local-dir",
        str(local_dir),
    ]

    run_command(command)

    print(f"\n[INFO] Downloaded {dataset_name}")
    print(f"[INFO] Repository: {info['repo_id']}")
    print(f"[INFO] Local directory: {local_dir}")

    return local_dir


def extract_longvideobench(dataset_dir: Path) -> None:
    dataset_dir = dataset_dir.expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    video_parts = sorted(dataset_dir.glob("videos.tar.part.*"))
    videos_tar = dataset_dir / "videos.tar"
    subtitles_tar = dataset_dir / "subtitles.tar"

    if video_parts and not videos_tar.exists():
        print(
            f"[INFO] Merging {len(video_parts)} video archive parts "
            f"into {videos_tar.name}"
        )

        with videos_tar.open("wb") as output_file:
            for part in video_parts:
                print(f"[INFO] Reading: {part.name}")

                with part.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output_file)

    if videos_tar.exists():
        print("[INFO] Extracting videos.tar")
        run_command(
            [
                "tar",
                "-xf",
                str(videos_tar),
                "-C",
                str(dataset_dir),
            ]
        )
    else:
        print("[WARN] videos.tar not found. Skipping video extraction.")

    if subtitles_tar.exists():
        print("[INFO] Extracting subtitles.tar")
        run_command(
            [
                "tar",
                "-xf",
                str(subtitles_tar),
                "-C",
                str(dataset_dir),
            ]
        )
    else:
        print("[WARN] subtitles.tar not found. Skipping subtitle extraction.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download long-video QA datasets from Hugging Face."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(SUPPORTED_DATASETS),
        help="Dataset name to download.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory in which raw datasets are stored.",
    )

    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded archives when supported.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = download_hf_dataset(
        dataset_name=args.dataset,
        output_root=args.output_root,
    )

    if args.extract:
        if args.dataset == "longvideobench":
            extract_longvideobench(dataset_dir)
        else:
            print(
                f"[INFO] No extraction step is required for "
                f"{args.dataset}."
            )


if __name__ == "__main__":
    main()