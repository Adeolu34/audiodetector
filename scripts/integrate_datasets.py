"""Integrate all downloaded datasets into a unified data/organized structure."""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORGANIZED_DIR = PROJECT_ROOT / "data" / "organized"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

# Category mapping: source folder -> unified category name
# Turkish names from YashNita/animals-sound-dataset
TURKISH_MAP = {
    "Aslan": "lion",
    "Esek": "donkey",
    "Inek": "cow",
    "Kedi": "cat",
    "Kedi-Part1": "cat",
    "Kedi-Part2": "cat",
    "Kopek": "dog",
    "Kopek-Part1": "dog",
    "Kopek-Part2": "dog",
    "Koyun": "sheep",
    "Kurbaga": "frog",
    "Kus": "bird",
    "Kus-Part1": "bird",
    "Kus-Part2": "bird",
    "Maymun": "monkey",
    "Tavuk": "chicken",
}


def copy_audio_files(src_dir: Path, dest_dir: Path, prefix: str, extensions=(".wav", ".mp3", ".ogg", ".flac")):
    """Copy audio files from src to dest with a prefix to avoid name collisions."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in extensions:
            dest_name = f"{prefix}_{f.name}"
            dest_path = dest_dir / dest_name
            if not dest_path.exists():
                shutil.copy2(str(f), str(dest_path))
            count += 1
    return count


def copy_recursive_audio(src_dir: Path, dest_dir: Path, prefix: str, extensions=(".wav", ".mp3", ".ogg", ".flac")):
    """Recursively copy all audio files from src tree to flat dest."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in extensions:
            dest_name = f"{prefix}_{f.name}"
            dest_path = dest_dir / dest_name
            if not dest_path.exists():
                shutil.copy2(str(f), str(dest_path))
            count += 1
    return count


def integrate_yashnita():
    """Integrate YashNita Animal-Sound-Dataset."""
    print("\n--- Integrating YashNita dataset ---")
    base = EXTERNAL_DIR / "yashnita"
    if not base.exists():
        print("  Not found, skipping.")
        return

    total = 0
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or folder.name == ".git":
            continue
        category = TURKISH_MAP.get(folder.name)
        if category is None:
            print(f"  Unknown category: {folder.name}, skipping")
            continue
        n = copy_audio_files(folder, ORGANIZED_DIR / category, "yashnita")
        print(f"  {folder.name} -> {category}: {n} files")
        total += n
    print(f"  YashNita total: {total} files")


def integrate_animals_sound_dataset():
    """Integrate caoofficial/animals-sound-dataset (same Turkish names)."""
    print("\n--- Integrating animals-sound-dataset (Kaggle) ---")
    base = EXTERNAL_DIR / "kaggle" / "animals-sound-dataset" / "Animal-Sound-Dataset"
    if not base.exists():
        print("  Not found, skipping.")
        return

    total = 0
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        category = TURKISH_MAP.get(folder.name)
        if category is None:
            print(f"  Unknown category: {folder.name}, skipping")
            continue
        n = copy_audio_files(folder, ORGANIZED_DIR / category, "kaggle_asd")
        print(f"  {folder.name} -> {category}: {n} files")
        total += n
    print(f"  animals-sound-dataset total: {total} files")


def integrate_audio_cats_dogs():
    """Integrate mmoreaux/audio-cats-and-dogs."""
    print("\n--- Integrating audio-cats-and-dogs (Kaggle) ---")
    base = EXTERNAL_DIR / "kaggle" / "audio-cats-and-dogs" / "cats_dogs"
    if not base.exists():
        print("  Not found, skipping.")
        return

    mapping = {"cat": "cat", "cats": "cat", "dog": "dog", "test": None}
    total = 0
    for split_dir in sorted(base.iterdir()):
        if not split_dir.is_dir():
            continue
        for category_dir in sorted(split_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = mapping.get(category_dir.name)
            if category is None:
                # "test" folder has mixed content, skip
                continue
            n = copy_audio_files(category_dir, ORGANIZED_DIR / category, f"catdog_{split_dir.name}")
            print(f"  {split_dir.name}/{category_dir.name} -> {category}: {n} files")
            total += n
    print(f"  audio-cats-and-dogs total: {total} files")


def integrate_sounds_of_animals():
    """Integrate haithammoh/sounds-of-animals."""
    print("\n--- Integrating sounds-of-animals (Kaggle) ---")
    base = EXTERNAL_DIR / "kaggle" / "sounds-of-animals" / "new"
    if not base.exists():
        print("  Not found, skipping.")
        return

    mapping = {"Cow": "cow", "Dog": "dog", "Frog": "frog"}
    total = 0
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        category = mapping.get(folder.name)
        if category is None:
            print(f"  Unknown category: {folder.name}, skipping")
            continue
        n = copy_audio_files(folder, ORGANIZED_DIR / category, "asc24")
        print(f"  {folder.name} -> {category}: {n} files")
        total += n
    print(f"  sounds-of-animals total: {total} files")


def integrate_bird_songs():
    """Integrate vinayshanbhag/bird-song-data-set."""
    print("\n--- Integrating bird-song-data-set (Kaggle) ---")
    base = EXTERNAL_DIR / "kaggle" / "bird-song-data-set" / "wavfiles"
    if not base.exists():
        print("  Not found, skipping.")
        return

    # All bird songs go into the "bird" category
    # Also create per-species subcategories
    import pandas as pd
    meta_path = EXTERNAL_DIR / "kaggle" / "bird-song-data-set" / "bird_songs_metadata.csv"

    bird_dir = ORGANIZED_DIR / "bird"
    bird_dir.mkdir(parents=True, exist_ok=True)

    if meta_path.exists():
        meta = pd.read_csv(str(meta_path))
        filename_to_species = dict(zip(meta["filename"], meta["name"]))
    else:
        filename_to_species = {}

    total = 0
    species_counts = {}
    for f in sorted(base.iterdir()):
        if f.is_file() and f.suffix.lower() == ".wav":
            species = filename_to_species.get(f.name, "unknown_bird")
            species_key = species.lower().replace(" ", "_").replace("'", "")
            dest_name = f"birdsong_{species_key}_{f.name}"
            dest_path = bird_dir / dest_name
            if not dest_path.exists():
                shutil.copy2(str(f), str(dest_path))
            total += 1
            species_counts[species] = species_counts.get(species, 0) + 1

    for sp, n in sorted(species_counts.items()):
        print(f"  {sp}: {n} files")
    print(f"  bird-song-data-set total: {total} files -> bird")


def main():
    print("=" * 50)
    print("INTEGRATING ALL DATASETS")
    print("=" * 50)

    integrate_yashnita()
    integrate_animals_sound_dataset()
    integrate_audio_cats_dogs()
    integrate_sounds_of_animals()
    integrate_bird_songs()

    # Final summary
    print("\n" + "=" * 50)
    print("FINAL UNIFIED DATASET SUMMARY")
    print("=" * 50)
    grand_total = 0
    for category_dir in sorted(ORGANIZED_DIR.iterdir()):
        if category_dir.is_dir():
            audio = list(category_dir.glob("*.wav")) + list(category_dir.glob("*.mp3")) + list(category_dir.glob("*.ogg"))
            print(f"  {category_dir.name:>15s}: {len(audio):>6d} files")
            grand_total += len(audio)
    print(f"  {'TOTAL':>15s}: {grand_total:>6d} files")
    print(f"\nCategories: {len(list(ORGANIZED_DIR.iterdir()))}")


if __name__ == "__main__":
    main()
