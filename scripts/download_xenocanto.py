"""Download a curated subset of animal sounds from Xeno-canto API."""

import os
import json
import time
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "xenocanto"

# Target species with common names and xeno-canto query terms
# We want ~50 recordings per species for practical training
QUERIES = {
    "frog": [
        ("Lithobates catesbeianus", "bullfrog"),
        ("Hyla arborea", "tree_frog"),
        ("Rana temporaria", "common_frog"),
    ],
    "bird": [
        ("Turdus merula", "blackbird"),
        ("Parus major", "great_tit"),
        ("Corvus corax", "raven"),
        ("Columba palumbus", "woodpigeon"),
        ("Erithacus rubecula", "robin"),
    ],
    "insects": [
        ("Gryllus campestris", "field_cricket"),
        ("Tettigonia viridissima", "great_green_bushcricket"),
    ],
}

MAX_PER_SPECIES = 50  # max recordings per species query


def search_xenocanto(query: str, max_results: int = 50) -> list:
    """Search xeno-canto API v2 for recordings matching query."""
    import urllib.parse
    encoded = urllib.parse.quote(query)
    url = f"https://xeno-canto.org/api/2/recordings?query={encoded}&page=1"

    try:
        req = Request(url, headers={"User-Agent": "ZooAudioAI/1.0"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  API error for '{query}': {e}")
        return []

    recordings = data.get("recordings", [])
    # Filter for quality A/B and take up to max_results
    good = [r for r in recordings if r.get("q", "E") in ("A", "B")]
    if len(good) < max_results:
        good += [r for r in recordings if r.get("q", "E") == "C"]
    return good[:max_results]


def download_recording(rec: dict, dest_path: Path) -> bool:
    """Download a single xeno-canto recording."""
    file_url = rec.get("file")
    if not file_url:
        return False

    if file_url.startswith("//"):
        file_url = "https:" + file_url

    try:
        urlretrieve(file_url, str(dest_path))
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    for category, species_list in QUERIES.items():
        cat_dir = OUTPUT_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        existing = list(cat_dir.glob("*.*"))
        if len(existing) >= MAX_PER_SPECIES * len(species_list) * 0.8:
            print(f"{category}: already has {len(existing)} files, skipping.")
            continue

        cat_count = 0
        for scientific_name, common_name in species_list:
            print(f"Searching: {scientific_name} ({common_name})...")
            recordings = search_xenocanto(scientific_name, MAX_PER_SPECIES)
            print(f"  Found {len(recordings)} recordings")

            for i, rec in enumerate(recordings):
                xc_id = rec.get("id", f"unknown_{i}")
                ext = rec.get("file-name", "audio.mp3").split(".")[-1]
                dest = cat_dir / f"{common_name}_{xc_id}.{ext}"

                if dest.exists():
                    cat_count += 1
                    continue

                if download_recording(rec, dest):
                    cat_count += 1
                    total_downloaded += 1

                    if total_downloaded % 10 == 0:
                        print(f"  Downloaded {total_downloaded} total...")

                # Be polite to the API
                time.sleep(0.5)

        print(f"{category}: {cat_count} files total")

    # Summary
    print("\n--- Xeno-canto Download Summary ---")
    total = 0
    for category in QUERIES:
        cat_dir = OUTPUT_DIR / category
        n = len(list(cat_dir.glob("*.*"))) if cat_dir.exists() else 0
        print(f"  {category:>10s}: {n} files")
        total += n
    print(f"  {'TOTAL':>10s}: {total} files")


if __name__ == "__main__":
    main()
