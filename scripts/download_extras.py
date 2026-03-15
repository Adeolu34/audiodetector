"""Download additional animal sound datasets from free sources."""

import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "external"


def download_animals_sound_kaggle_cao():
    """Download the caoofficial/animals-sound-dataset from Kaggle."""
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_c14e1c22b8ea1347eb1040fc80b3b6e3'
    from kaggle.api.kaggle_api_extended import KaggleApi

    ds_dir = OUTPUT_DIR / "kaggle" / "animals-sound-dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)

    existing = list(ds_dir.rglob("*.wav")) + list(ds_dir.rglob("*.mp3"))
    if len(existing) > 10:
        print(f"animals-sound-dataset: already has {len(existing)} files, skipping.")
        return

    api = KaggleApi()
    api.authenticate()

    print("Downloading caoofficial/animals-sound-dataset...")
    try:
        api.dataset_download_files("caoofficial/animals-sound-dataset", path=str(ds_dir), unzip=True)
        audio_files = list(ds_dir.rglob("*.wav")) + list(ds_dir.rglob("*.mp3"))
        print(f"  animals-sound-dataset: {len(audio_files)} audio files")
    except Exception as e:
        print(f"  ERROR: {e}")


def download_environmental_sound():
    """Download the environmental sound classification dataset from Kaggle."""
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_c14e1c22b8ea1347eb1040fc80b3b6e3'
    from kaggle.api.kaggle_api_extended import KaggleApi

    ds_dir = OUTPUT_DIR / "kaggle" / "environmental-sound"
    ds_dir.mkdir(parents=True, exist_ok=True)

    existing = list(ds_dir.rglob("*.wav")) + list(ds_dir.rglob("*.mp3")) + list(ds_dir.rglob("*.ogg"))
    if len(existing) > 10:
        print(f"environmental-sound: already has {len(existing)} files, skipping.")
        return

    api = KaggleApi()
    api.authenticate()

    print("Downloading ejlok1/toronto-emotional-speech-set-tess (skip)...")
    # Try animal-specific datasets
    extra_datasets = [
        "mexwell/bird-sounds",
        "vinayshanbhag/bird-song-data-set",
    ]

    for ds_slug in extra_datasets:
        ds_name = ds_slug.split("/")[-1]
        dd = OUTPUT_DIR / "kaggle" / ds_name
        dd.mkdir(parents=True, exist_ok=True)

        existing = list(dd.rglob("*.wav")) + list(dd.rglob("*.mp3")) + list(dd.rglob("*.ogg"))
        if len(existing) > 10:
            print(f"  {ds_name}: already has {len(existing)} files, skipping.")
            continue

        print(f"Downloading {ds_slug}...")
        try:
            api.dataset_download_files(ds_slug, path=str(dd), unzip=True)
            audio_files = list(dd.rglob("*.wav")) + list(dd.rglob("*.mp3")) + list(dd.rglob("*.ogg"))
            print(f"  {ds_name}: {len(audio_files)} audio files")
        except Exception as e:
            print(f"  {ds_name} ERROR: {e}")


def main():
    print("=== Downloading additional datasets ===\n")
    download_animals_sound_kaggle_cao()
    download_environmental_sound()

    # Summary
    print("\n--- Extra Downloads Summary ---")
    total = 0
    for p in OUTPUT_DIR.glob("kaggle/*"):
        if p.is_dir():
            audio = list(p.rglob("*.wav")) + list(p.rglob("*.mp3")) + list(p.rglob("*.ogg"))
            print(f"  {p.name}: {len(audio)} audio files")
            total += len(audio)
    print(f"  TOTAL: {total} audio files")


if __name__ == "__main__":
    main()
