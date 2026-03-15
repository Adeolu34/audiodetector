"""Training loop for the animal sound classification model."""

import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from zoo_audio.dataset import create_dataloaders
from zoo_audio.model import create_model


def load_config() -> dict:
    """Load configuration from config.yaml in the project root."""
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(str(config_path), "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch and return average loss."""
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

    return running_loss / max(num_batches, 1)


def validate(model, val_loader, criterion, device):
    """Validate and return (average loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for spectrograms, labels in val_loader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            num_batches += 1

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main():
    """Run the full training pipeline."""
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build dataloaders
    train_loader, val_loader, _, label_map, class_weights = create_dataloaders(config)

    # Initialize model
    num_classes = len(label_map)
    model_type = config["model"].get("type", "cnn")
    model = create_model(
        model_type,
        num_classes=num_classes,
        dropout=config["model"].get("dropout", 0.2),
        fc_dropout=config["model"].get("fc_dropout", 0.3),
        pretrained=config["model"].get("pretrained", True),
    ).to(device)

    print(f"\nModel type: {model_type}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer, scheduler, loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Training state
    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    checkpoint_dir = Path(config["data"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pth"

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...\n")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {lr:.6f}"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "label_map": label_map,
                "config": config,
                "model_type": model_type,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, str(checkpoint_path))
            print(f"  -> Saved best model (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\n--- Training Summary ---")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
