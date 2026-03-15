"""Download and organize the ESC-50 dataset for animal sound classification."""

import os
import shutil
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

ANIMAL_CATEGORIES = [
    "dog", "rooster", "pig", "cow", "frog",
    "cat", "hen", "insects", "sheep", "crow",
]


def download_esc50(dest_dir: str) -> Path:
    """Download the ESC-50 zip archive to dest_dir."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "ESC-50-master.zip"

    if zip_path.exists():
        print(f"Archive already exists at {zip_path}, skipping download.")
        return zip_path

    print(f"Downloading ESC-50 from {ESC50_URL} ...")
    urllib.request.urlretrieve(ESC50_URL, str(zip_path))
    print(f"Downloaded to {zip_path}")
    return zip_path


def extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract the zip archive and return the extracted folder path."""
    extracted = dest_dir / "ESC-50-master"
    if extracted.exists():
        print(f"Already extracted at {extracted}, skipping extraction.")
        return extracted

    print("Extracting archive ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(dest_dir))
    print(f"Extracted to {extracted}")
    return extracted


def organize_animal_sounds(esc50_dir: Path, output_dir: Path) -> dict:
    """Filter animal categories from ESC-50 and organize into subdirectories.

    Returns a dict mapping category name -> number of files copied.
    """
    meta_path = esc50_dir / "meta" / "esc50.csv"
    audio_dir = esc50_dir / "audio"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    df = pd.read_csv(str(meta_path))

    output_dir = Path(output_dir)
    summary = {}

    for category in ANIMAL_CATEGORIES:
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        cat_files = df[df["category"] == category]
        count = 0
        for _, row in cat_files.iterrows():
            src = audio_dir / row["filename"]
            dst = cat_dir / row["filename"]
            if src.exists() and not dst.exists():
                shutil.copy2(str(src), str(dst))
            if dst.exists():
                count += 1

        summary[category] = count

    return summary


def main():
    """Download, extract, and organize the ESC-50 animal sound dataset."""
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    organized_dir = project_root / "data" / "organized"

    zip_path = download_esc50(str(raw_dir))
    esc50_dir = extract_zip(zip_path, raw_dir)
    summary = organize_animal_sounds(esc50_dir, organized_dir)

    print("\n--- Dataset Summary ---")
    total = 0
    for category, count in sorted(summary.items()):
        print(f"  {category:>10s}: {count} files")
        total += count
    print(f"  {'TOTAL':>10s}: {total} files")
    print(f"\nOrganized data saved to: {organized_dir}")


if __name__ == "__main__":
    main()
