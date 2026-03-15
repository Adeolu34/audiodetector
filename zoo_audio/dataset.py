"""PyTorch Dataset and DataLoader factory for animal sound classification."""

import os
import random as _random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import yaml

from zoo_audio.audio_utils import preprocess_file


class AnimalSoundDataset(Dataset):
    """PyTorch Dataset that loads audio on-the-fly and returns mel spectrograms."""

    def __init__(self, file_list: list, label_map: dict, config: dict,
                 augment: bool = False):
        """
        Args:
            file_list: List of (filepath, category_name) tuples.
            label_map: Dict mapping category_name -> int index.
            config: Audio configuration dict with keys like sample_rate, duration, etc.
            augment: If True, apply data augmentation during preprocessing.
        """
        self.file_list = file_list
        self.label_map = label_map
        self.config = config
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, category = self.file_list[idx]
        label = self.label_map[category]

        mel_spec = preprocess_file(
            filepath,
            sr=self.config["sample_rate"],
            duration=self.config["duration"],
            n_mels=self.config["n_mels"],
            n_fft=self.config["n_fft"],
            hop_length=self.config["hop_length"],
            augment=self.augment,
        )

        # Add channel dimension: (1, n_mels, T)
        tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
        return tensor, label


def build_label_map(data_dir: str) -> dict:
    """Scan subdirectories and return a sorted {category_name: int_index} map."""
    data_dir = Path(data_dir)
    categories = sorted(
        [d.name for d in data_dir.iterdir() if d.is_dir()]
    )
    return {name: idx for idx, name in enumerate(categories)}


def get_file_list(data_dir: str) -> list:
    """Return a list of (filepath, category_name) tuples for all audio files."""
    data_dir = Path(data_dir)
    file_list = []
    extensions = (".wav", ".mp3", ".ogg", ".flac")
    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for audio_file in sorted(category_dir.iterdir()):
            if audio_file.suffix.lower() in extensions:
                file_list.append((str(audio_file), category))
    return file_list


def _cap_class_samples(file_list: list, cap_map: dict, random_state: int = 42) -> list:
    """Subsample dominant classes according to *cap_map*.

    Args:
        file_list: List of (filepath, category_name) tuples.
        cap_map: Dict mapping category_name -> max_samples (e.g. {"bird": 1000}).
        random_state: Seed for reproducible subsampling.

    Returns a new file list with capped classes.
    """
    rng = _random.Random(random_state)
    by_class: dict[str, list] = {}
    for item in file_list:
        by_class.setdefault(item[1], []).append(item)

    result = []
    for cls, items in by_class.items():
        cap = cap_map.get(cls)
        if cap is not None and len(items) > cap:
            items = rng.sample(items, cap)
        result.extend(items)
    return result


def compute_class_weights(file_list: list, label_map: dict, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights: total / (num_classes * count).

    Returns a float tensor of shape (num_classes,).
    """
    counts = Counter(cat for _, cat in file_list)
    total = len(file_list)
    weights = torch.ones(num_classes)
    for cls, idx in label_map.items():
        if counts[cls] > 0:
            weights[idx] = total / (num_classes * counts[cls])
    return weights


def create_dataloaders(config: dict) -> tuple:
    """Create train/val/test DataLoaders with stratified 70/15/15 split.

    Args:
        config: Full config dict (loaded from config.yaml).

    Returns:
        (train_loader, val_loader, test_loader, label_map, class_weights)
    """
    data_dir = config["data"]["data_dir"]
    audio_config = config["audio"]
    batch_size = config["training"]["batch_size"]
    class_caps = config.get("training", {}).get("class_caps", {})

    label_map = build_label_map(data_dir)
    file_list = get_file_list(data_dir)
    num_classes = len(label_map)

    # Extract labels for stratification
    labels = [label_map[cat] for _, cat in file_list]

    # First split: 70% train, 30% temp
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_list, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Second split: 50/50 of temp -> 15% val, 15% test
    val_files, test_files, _, _ = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    # Cap dominant classes in training set
    if class_caps:
        train_files = _cap_class_samples(train_files, class_caps)

    # Compute class weights from (capped) training distribution
    class_weights = compute_class_weights(train_files, label_map, num_classes)

    # Build WeightedRandomSampler for training
    train_labels_capped = [label_map[cat] for _, cat in train_files]
    counts = Counter(train_labels_capped)
    sample_weights = [1.0 / counts[l] for l in train_labels_capped]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dataset = AnimalSoundDataset(train_files, label_map, audio_config, augment=True)
    val_dataset = AnimalSoundDataset(val_files, label_map, audio_config)
    test_dataset = AnimalSoundDataset(test_files, label_map, audio_config)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Dataset splits - Train: {len(train_dataset)}, "
          f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Classes ({len(label_map)}): {list(label_map.keys())}")
    if class_caps:
        print(f"Class caps applied: {class_caps}")

    return train_loader, val_loader, test_loader, label_map, class_weights
