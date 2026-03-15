"""Download the cgeorgiaw/animal-sounds dataset from HuggingFace and export to WAV."""

import os
from pathlib import Path

import soundfile as sf
import numpy as np
from datasets import load_dataset

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "huggingface"

# Correct config names from the dataset
SPECIES = [
    "egyptian_fruit_bat",
    "egyptian_fruit_bat_extended",
    "giant_otter",
    "macaque",
    "orca",
    "zebra_finch",
]


def download_and_export():
    """Download each species split and export audio to WAV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for species in SPECIES:
        species_dir = OUTPUT_DIR / species
        species_dir.mkdir(parents=True, exist_ok=True)

        existing = list(species_dir.glob("*.wav"))
        if len(existing) > 0:
            print(f"  {species}: already has {len(existing)} files, skipping.")
            continue

        print(f"Downloading {species}...")
        try:
            ds = load_dataset("cgeorgiaw/animal-sounds", name=species, split="train",
                              trust_remote_code=True)
        except Exception as e:
            print(f"  ERROR downloading {species}: {e}")
            continue

        count = 0
        for i, sample in enumerate(ds):
            audio = sample.get("audio")
            if audio is None:
                continue
            waveform = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]

            wav_path = species_dir / f"{species}_{i:05d}.wav"
            sf.write(str(wav_path), waveform, sr)
            count += 1

            if count % 200 == 0:
                print(f"  {species}: exported {count} files...")

        print(f"  {species}: {count} files exported.")

    # Summary
    print("\n--- HuggingFace Dataset Summary ---")
    total = 0
    for species in SPECIES:
        species_dir = OUTPUT_DIR / species
        n = len(list(species_dir.glob("*.wav"))) if species_dir.exists() else 0
        print(f"  {species:>30s}: {n} files")
        total += n
    print(f"  {'TOTAL':>30s}: {total} files")


if __name__ == "__main__":
    download_and_export()
