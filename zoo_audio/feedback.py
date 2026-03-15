"""Self-learning feedback system for Zoo Audio AI.

Provides feedback collection, storage, and incremental fine-tuning so users
can correct wrong predictions and the model learns from those corrections.
"""

import csv
import json
import uuid
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from zoo_audio.audio_utils import preprocess_file
from zoo_audio.model import (
    create_model, get_final_fc_layer, set_final_fc_layer, freeze_backbone,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeedbackEntry:
    """A single feedback record linking a prediction to a user correction."""

    audio_path: str
    predicted_label: str
    predicted_confidence: float
    true_label: str
    is_correct: bool
    model_version: str
    checkpoint_path: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Persistent feedback store (JSONL)
# ---------------------------------------------------------------------------

class FeedbackStore:
    """Append-only store backed by a JSONL file."""

    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.feedback_dir / "feedback.jsonl"

    def add(self, entry: FeedbackEntry) -> None:
        """Append a single feedback entry."""
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def load_all(self) -> list[FeedbackEntry]:
        """Read every entry from the store."""
        if not self.filepath.exists():
            return []
        entries = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(FeedbackEntry(**json.loads(line)))
        return entries

    def load_pending(self, since_version: str | None = None) -> list[FeedbackEntry]:
        """Return entries not yet consumed by fine-tuning.

        If *since_version* is given, return entries whose ``model_version``
        matches that version (i.e. feedback collected on the latest model
        that has not been fine-tuned yet).  If ``None``, return all entries.
        """
        entries = self.load_all()
        if since_version is None:
            return entries
        return [e for e in entries if e.model_version == since_version]

    def summary(self) -> None:
        """Print human-readable statistics about collected feedback."""
        entries = self.load_all()
        if not entries:
            print("No feedback entries recorded yet.")
            return

        total = len(entries)
        correct = sum(1 for e in entries if e.is_correct)
        wrong = total - correct

        print(f"\n--- Feedback Summary ---")
        print(f"Total entries:  {total}")
        print(f"Confirmed correct:  {correct}")
        print(f"Corrections:  {wrong}")

        # Corrections broken down by true label
        corrections = [e for e in entries if not e.is_correct]
        if corrections:
            print(f"\nCorrections by true label:")
            counts = Counter(e.true_label for e in corrections)
            for label, count in counts.most_common():
                print(f"  {label:>15s}: {count}")

        # Detect new categories (labels that never appeared as predicted_label)
        predicted_labels = set(e.predicted_label for e in entries)
        true_labels = set(e.true_label for e in entries)
        new_cats = true_labels - predicted_labels
        if new_cats:
            print(f"\nNew categories seen in feedback: {sorted(new_cats)}")

    def list_entries(self) -> None:
        """Print every entry in a readable table."""
        entries = self.load_all()
        if not entries:
            print("No feedback entries recorded yet.")
            return

        print(f"\n{'ID':>8s}  {'Timestamp':>20s}  {'Predicted':>12s}  "
              f"{'Conf':>6s}  {'True':>12s}  {'Correct':>7s}")
        print("-" * 75)
        for e in entries:
            short_id = e.id[:8]
            ts = e.timestamp[:19]
            print(
                f"{short_id:>8s}  {ts:>20s}  {e.predicted_label:>12s}  "
                f"{e.predicted_confidence:6.2%}  {e.true_label:>12s}  "
                f"{'yes' if e.is_correct else 'no':>7s}"
            )

    def export_csv(self, csv_path: str) -> None:
        """Export all feedback entries as a CSV file."""
        entries = self.load_all()
        if not entries:
            print("No feedback entries to export.")
            return

        fieldnames = list(asdict(entries[0]).keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                writer.writerow(asdict(e))
        print(f"Exported {len(entries)} entries to {csv_path}")


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapping feedback entries
# ---------------------------------------------------------------------------

class FeedbackDataset(Dataset):
    """Wraps feedback entries into a PyTorch Dataset for fine-tuning."""

    def __init__(self, entries: list[FeedbackEntry], label_map: dict, audio_config: dict):
        self.label_map = label_map
        self.audio_config = audio_config

        # Filter to entries whose audio files still exist
        self.entries: list[FeedbackEntry] = []
        for entry in entries:
            if Path(entry.audio_path).exists():
                self.entries.append(entry)
            else:
                warnings.warn(
                    f"Audio file missing, skipping feedback entry: {entry.audio_path}"
                )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        mel_spec = preprocess_file(
            entry.audio_path,
            sr=self.audio_config["sample_rate"],
            duration=self.audio_config["duration"],
            n_mels=self.audio_config["n_mels"],
            n_fft=self.audio_config["n_fft"],
            hop_length=self.audio_config["hop_length"],
        )
        tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
        label = self.label_map[entry.true_label]
        return tensor, label

    def get_labels(self) -> list[int]:
        """Return integer labels for every entry (useful for samplers)."""
        return [self.label_map[e.true_label] for e in self.entries]


# ---------------------------------------------------------------------------
# Incremental fine-tuning
# ---------------------------------------------------------------------------

def _expand_fc_layer(model: nn.Module, old_num: int, new_num: int, device) -> None:
    """Expand the final Linear layer to accommodate new classes.

    Copies existing weights and initialises new rows with small random values.
    Works for any architecture supported by get/set_final_fc_layer.
    """
    old_layer = get_final_fc_layer(model)
    new_layer = nn.Linear(old_layer.in_features, new_num).to(device)

    # Copy old weights
    with torch.no_grad():
        new_layer.weight[:old_num] = old_layer.weight
        new_layer.bias[:old_num] = old_layer.bias
        # Xavier init for new class rows
        nn.init.xavier_uniform_(new_layer.weight[old_num:].unsqueeze(0))
        nn.init.zeros_(new_layer.bias[old_num:])

    set_final_fc_layer(model, new_layer)


def fine_tune(config: dict, feedback_entries: list[FeedbackEntry],
              base_checkpoint: str, epochs: int = 10, lr: float = 0.0001,
              freeze_conv_blocks: int = 2) -> str:
    """Incrementally fine-tune the model using collected feedback.

    Returns the path to the newly saved checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on device: {device}")

    # Load base checkpoint
    checkpoint = torch.load(base_checkpoint, map_location=device, weights_only=False)
    label_map: dict = dict(checkpoint["label_map"])
    audio_config: dict = checkpoint["config"]["audio"]
    model_config: dict = checkpoint["config"]["model"]

    # Determine base version
    base_version = checkpoint.get("version", "v1")
    new_version_num = int(base_version.lstrip("v")) + 1
    new_version = f"v{new_version_num}"

    # Discover new categories from feedback
    existing_classes = set(label_map.keys())
    feedback_classes = set(e.true_label for e in feedback_entries)
    new_classes = sorted(feedback_classes - existing_classes)

    if new_classes:
        print(f"New categories found in feedback: {new_classes}")
        for cls in new_classes:
            label_map[cls] = len(label_map)

    old_num_classes = len(existing_classes)
    num_classes = len(label_map)

    # Build model and load weights
    model_type = checkpoint.get("model_type", "cnn")
    model = create_model(
        model_type,
        num_classes=old_num_classes,
        dropout=model_config.get("dropout", 0.2),
        fc_dropout=model_config.get("fc_dropout", 0.3),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Expand FC if new categories
    if new_classes:
        _expand_fc_layer(model, old_num_classes, num_classes, device)
        print(f"Expanded output layer: {old_num_classes} -> {num_classes} classes")

    # Freeze early conv blocks to prevent catastrophic forgetting
    freeze_backbone(model, freeze_conv_blocks)
    print(f"Froze first {freeze_conv_blocks} conv block(s)")

    # Build dataset
    dataset = FeedbackDataset(feedback_entries, label_map, audio_config)
    if len(dataset) == 0:
        print("No valid feedback samples found. Aborting fine-tune.")
        return base_checkpoint

    # Train/val split (80/20) if enough data, else train-only
    val_loader = None
    if len(dataset) >= 10:
        val_size = max(1, len(dataset) // 5)
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Build weighted sampler from training subset
        train_labels = [dataset.get_labels()[i] for i in train_ds.indices]
        class_counts = Counter(train_labels)
        weights = [1.0 / class_counts[l] for l in train_labels]
        sampler = WeightedRandomSampler(weights, len(weights))

        train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    else:
        all_labels = dataset.get_labels()
        class_counts = Counter(all_labels)
        weights = [1.0 / class_counts[l] for l in all_labels]
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=0)

    # Class-weighted cross-entropy loss
    total_samples = sum(class_counts.values())
    class_weight = torch.zeros(num_classes, device=device)
    for cls_idx, count in class_counts.items():
        class_weight[cls_idx] = total_samples / count
    # Normalise so weights average to 1
    class_weight = class_weight / class_weight[class_weight > 0].mean()
    # Fill missing classes with weight 1.0
    class_weight[class_weight == 0] = 1.0

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    # Training loop
    print(f"\nFine-tuning for {epochs} epochs on {len(dataset)} feedback samples...")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for spectrograms, labels in train_loader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)

        if val_loader is not None:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for spectrograms, labels in val_loader:
                    spectrograms = spectrograms.to(device)
                    labels = labels.to(device)
                    outputs = model(spectrograms)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = correct / max(total, 1)
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

    # Use best validation state if available
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save new versioned checkpoint
    checkpoint_dir = Path(base_checkpoint).parent
    new_checkpoint_path = checkpoint_dir / f"model_{new_version}.pth"

    feedback_ids = [e.id for e in feedback_entries]
    new_checkpoint = {
        "model_state_dict": model.state_dict(),
        "label_map": label_map,
        "config": config,
        "model_type": model_type,
        "epoch": epochs,
        "val_accuracy": best_val_acc if val_loader else None,
        "val_loss": None,
        "version": new_version,
        "parent": base_version,
        "feedback_count": len(feedback_entries),
        "feedback_ids": feedback_ids,
    }
    torch.save(new_checkpoint, str(new_checkpoint_path))
    print(f"\nSaved fine-tuned checkpoint: {new_checkpoint_path}")
    print(f"  Version: {new_version} (parent: {base_version})")
    print(f"  Classes: {num_classes} | Feedback samples: {len(feedback_entries)}")

    return str(new_checkpoint_path)


# ---------------------------------------------------------------------------
# Model version inspector
# ---------------------------------------------------------------------------

def get_model_versions(checkpoint_dir: str = "checkpoints") -> list[dict]:
    """Scan checkpoint directory and return metadata for each version."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    versions = []
    for pth_file in sorted(checkpoint_dir.glob("*.pth")):
        try:
            checkpoint = torch.load(str(pth_file), map_location="cpu", weights_only=False)
        except Exception:
            continue

        info = {
            "file": str(pth_file),
            "version": checkpoint.get("version", "v1"),
            "parent": checkpoint.get("parent", None),
            "epoch": checkpoint.get("epoch"),
            "val_accuracy": checkpoint.get("val_accuracy"),
            "num_classes": len(checkpoint.get("label_map", {})),
            "feedback_count": checkpoint.get("feedback_count", 0),
        }
        versions.append(info)

    return versions
