#!/usr/bin/env python3
"""Script to download the ETDataset for mr-Diff experiments."""

import argparse
import os
import urllib.request
from pathlib import Path


# ETDataset URLs from the official repository
DATASET_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}


def download_file(url: str, dest_path: Path) -> None:
    """Download a file from URL to destination path."""
    print(f"Downloading {url}...")

    # Create parent directory if needed
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Download with progress
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        print(f"\r  Progress: {percent}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
    print()  # Newline after progress


def main():
    parser = argparse.ArgumentParser(description="Download ETDataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ETDataset",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["ETTh1", "ETTm1"],
        choices=list(DATASET_URLS.keys()),
        help="Datasets to download",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading datasets to: {output_dir}")
    print(f"Datasets: {args.datasets}")
    print()

    for dataset_name in args.datasets:
        url = DATASET_URLS[dataset_name]
        dest_path = output_dir / f"{dataset_name}.csv"

        if dest_path.exists():
            print(f"{dataset_name} already exists at {dest_path}")
            continue

        try:
            download_file(url, dest_path)
            print(f"  Saved to: {dest_path}")

            # Verify the file
            with open(dest_path) as f:
                lines = sum(1 for _ in f)
            print(f"  Rows: {lines - 1} (excluding header)")

        except Exception as e:
            print(f"  Error downloading {dataset_name}: {e}")

    print("\nDownload complete!")
    print("\nDataset information:")
    print("- ETTh1/ETTh2: Hourly data (17,420 timesteps)")
    print("- ETTm1/ETTm2: 15-minute data (69,680 timesteps)")
    print("- Features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (7 total)")
    print("  - HUFL: High UseFul Load")
    print("  - HULL: High UseLess Load")
    print("  - MUFL: Middle UseFul Load")
    print("  - MULL: Middle UseLess Load")
    print("  - LUFL: Low UseFul Load")
    print("  - LULL: Low UseLess Load")
    print("  - OT: Oil Temperature (target for univariate)")


if __name__ == "__main__":
    main()
