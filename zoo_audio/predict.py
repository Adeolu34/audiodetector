"""Single-file prediction for animal sound classification."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from zoo_audio.audio_utils import preprocess_file
from zoo_audio.model import create_model


def predict(audio_path: str, checkpoint_path: str = None, top_k: int = 5) -> dict | None:
    """Predict the animal sound in an audio file.

    Args:
        audio_path: Path to the input WAV file.
        checkpoint_path: Path to model checkpoint. Defaults to checkpoints/best_model.pth.
        top_k: Number of top predictions to display.

    Returns:
        A dict with keys ``predictions``, ``audio_path``, ``checkpoint_path``,
        and ``model_version``, or ``None`` on error.
    """
    if checkpoint_path is None:
        checkpoint_path = str(Path("checkpoints") / "best_model.pth")

    if not Path(checkpoint_path).exists():
        print(f"No checkpoint found at {checkpoint_path}. Train a model first.")
        return None

    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_map = checkpoint["label_map"]
    config = checkpoint["config"]
    audio_config = config["audio"]
    model_version = checkpoint.get("version", "v1")

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
    model.eval()

    # Preprocess audio
    mel_spec = preprocess_file(
        audio_path,
        sr=audio_config["sample_rate"],
        duration=audio_config["duration"],
        n_mels=audio_config["n_mels"],
        n_fft=audio_config["n_fft"],
        hop_length=audio_config["hop_length"],
    )

    # Convert to tensor: (1, 1, n_mels, T)
    tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)

    # Get top-K predictions
    idx_to_class = {v: k for k, v in label_map.items()}
    top_k = min(top_k, num_classes)
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

    predictions = []
    print(f"\nPrediction for: {audio_path}")
    print("-" * 40)
    for i in range(top_k):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        class_name = idx_to_class[idx]
        bar = "#" * int(prob * 30)
        print(f"  {class_name:>10s}: {prob:6.2%}  {bar}")
        predictions.append({"class": class_name, "probability": prob})

    return {
        "predictions": predictions,
        "audio_path": str(audio_path),
        "checkpoint_path": str(checkpoint_path),
        "model_version": model_version,
    }


def predict_with_feedback(audio_path: str, checkpoint_path: str = None,
                          top_k: int = 5) -> None:
    """Run prediction and interactively collect user feedback.

    Prompts the user after displaying results:
      - ``y``  -> store as confirmed correct (top-1 prediction)
      - ``n``  -> ask for the true label, store as correction
      - Any other string -> treat as the true label (correction)
    """
    from zoo_audio.feedback import FeedbackEntry, FeedbackStore

    result = predict(audio_path, checkpoint_path, top_k)
    if result is None:
        return

    top_pred = result["predictions"][0]
    predicted_label = top_pred["class"]
    predicted_conf = top_pred["probability"]

    print()
    response = input(
        f"Is '{predicted_label}' correct?  (y / n / type the true label): "
    ).strip()

    if not response:
        print("No feedback provided, skipping.")
        return

    if response.lower() == "y":
        is_correct = True
        true_label = predicted_label
    elif response.lower() == "n":
        true_label = input("Enter the correct label: ").strip()
        if not true_label:
            print("No label provided, skipping.")
            return
        is_correct = False
    else:
        # Treat the response as the true label
        true_label = response
        is_correct = (true_label == predicted_label)

    entry = FeedbackEntry(
        audio_path=str(Path(audio_path).resolve()),
        predicted_label=predicted_label,
        predicted_confidence=predicted_conf,
        true_label=true_label,
        is_correct=is_correct,
        model_version=result["model_version"],
        checkpoint_path=result["checkpoint_path"],
    )

    store = FeedbackStore()
    store.add(entry)
    status = "confirmed" if is_correct else f"corrected -> {true_label}"
    print(f"Feedback saved ({status})")


def main():
    """CLI entry point for single-file prediction."""
    parser = argparse.ArgumentParser(description="Predict animal sound from audio file")
    parser.add_argument("--audio", required=True, help="Path to input WAV file")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions")
    args = parser.parse_args()

    predict(args.audio, args.checkpoint, args.top_k)


if __name__ == "__main__":
    main()
