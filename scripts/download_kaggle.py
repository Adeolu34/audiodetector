"""Download Kaggle animal sound datasets."""

import os
import zipfile
from pathlib import Path

os.environ['KAGGLE_KEY'] = 'KGAT_c14e1c22b8ea1347eb1040fc80b3b6e3'
os.environ['KAGGLE_USERNAME'] = '_'

from kaggle.api.kaggle_api_extended import KaggleApi

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "kaggle"

DATASETS = [
    "mmoreaux/audio-cats-and-dogs",
    "haithammoh/sounds-of-animals",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    for ds_slug in DATASETS:
        ds_name = ds_slug.split("/")[-1]
        ds_dir = OUTPUT_DIR / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        existing_files = list(ds_dir.rglob("*.wav")) + list(ds_dir.rglob("*.mp3")) + list(ds_dir.rglob("*.ogg"))
        if len(existing_files) > 10:
            print(f"{ds_slug}: already has {len(existing_files)} audio files, skipping.")
            continue

        print(f"Downloading {ds_slug}...")
        try:
            api.dataset_download_files(ds_slug, path=str(ds_dir), unzip=True)
            # Count what we got
            audio_files = list(ds_dir.rglob("*.wav")) + list(ds_dir.rglob("*.mp3")) + list(ds_dir.rglob("*.ogg"))
            print(f"  {ds_slug}: {len(audio_files)} audio files downloaded")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n--- Kaggle Downloads Summary ---")
    total = 0
    for ds_slug in DATASETS:
        ds_name = ds_slug.split("/")[-1]
        ds_dir = OUTPUT_DIR / ds_name
        if ds_dir.exists():
            audio = list(ds_dir.rglob("*.wav")) + list(ds_dir.rglob("*.mp3")) + list(ds_dir.rglob("*.ogg"))
            print(f"  {ds_name}: {len(audio)} audio files")
            total += len(audio)
    print(f"  TOTAL: {total} audio files")


if __name__ == "__main__":
    main()
