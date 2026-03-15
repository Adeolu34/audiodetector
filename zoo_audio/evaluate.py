"""Evaluation metrics and confusion matrix generation."""

from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from zoo_audio.dataset import create_dataloaders
from zoo_audio.model import create_model


def evaluate_model(model, test_loader, device):
    """Run inference on the test set and return (all_labels, all_preds)."""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for spectrograms, labels in test_loader:
            spectrograms = spectrograms.to(device)
            outputs = model(spectrograms)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm, class_names, save_path):
    """Generate and save a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def main():
    """Load the best checkpoint, evaluate on test set, and print results."""
    checkpoint_path = Path("checkpoints") / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}. Train a model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    label_map = checkpoint["label_map"]
    config = checkpoint["config"]

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_accuracy={checkpoint['val_accuracy']:.4f})")

    # Build test dataloader
    _, _, test_loader, _, _ = create_dataloaders(config)

    # Initialize model and load weights
    num_classes = len(label_map)
    model_type = checkpoint.get("model_type", "cnn")
    model = create_model(
        model_type,
        num_classes=num_classes,
        dropout=config["model"].get("dropout", 0.2),
        fc_dropout=config["model"].get("fc_dropout", 0.3),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    all_labels, all_preds = evaluate_model(model, test_loader, device)

    # Invert label_map for display
    idx_to_class = {v: k for k, v in label_map.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # Classification report
    print("\n--- Classification Report ---\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    save_path = Path("checkpoints") / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, save_path)

    # Overall accuracy
    accuracy = np.mean(all_labels == all_preds)
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
