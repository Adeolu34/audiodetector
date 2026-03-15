"""Download BirdCLEF+ 2025 competition data from Kaggle."""

import os
from pathlib import Path

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_c14e1c22b8ea1347eb1040fc80b3b6e3'

from kaggle.api.kaggle_api_extended import KaggleApi

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "birdclef2025"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    # Check if already downloaded
    existing = list(OUTPUT_DIR.rglob("*.ogg")) + list(OUTPUT_DIR.rglob("*.wav")) + list(OUTPUT_DIR.rglob("*.mp3"))
    if len(existing) > 100:
        print(f"BirdCLEF 2025: already has {len(existing)} audio files, skipping.")
        return

    print("Downloading BirdCLEF+ 2025 competition data...")
    print("(This is ~12GB, it will take a while)")
    try:
        api.competition_download_files("birdclef-2025", path=str(OUTPUT_DIR))
        print("Download complete. Extracting...")

        # Extract any zip files
        import zipfile
        for zf_path in OUTPUT_DIR.glob("*.zip"):
            print(f"  Extracting {zf_path.name}...")
            with zipfile.ZipFile(str(zf_path), "r") as zf:
                zf.extractall(str(OUTPUT_DIR))
            zf_path.unlink()
            print(f"  Extracted and removed {zf_path.name}")

        audio_files = list(OUTPUT_DIR.rglob("*.ogg")) + list(OUTPUT_DIR.rglob("*.wav")) + list(OUTPUT_DIR.rglob("*.mp3"))
        print(f"\nBirdCLEF 2025: {len(audio_files)} audio files ready")
    except Exception as e:
        print(f"ERROR: {e}")
        print("Note: You may need to accept the competition rules at:")
        print("  https://www.kaggle.com/competitions/birdclef-2025/rules")


if __name__ == "__main__":
    main()
