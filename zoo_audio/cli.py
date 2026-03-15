"""Unified CLI entry point for Zoo Audio AI."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="zoo_audio",
        description="Zoo Audio AI - Animal sound classification",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # download
    subparsers.add_parser("download", help="Download and organize ESC-50 dataset")

    # train
    subparsers.add_parser("train", help="Train the classification model")

    # evaluate
    subparsers.add_parser("evaluate", help="Evaluate model on test set")

    # predict
    predict_parser = subparsers.add_parser("predict", help="Predict animal sound from audio file")
    predict_parser.add_argument("--audio", required=True, help="Path to input WAV file")
    predict_parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    predict_parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions")
    predict_parser.add_argument(
        "--feedback", action="store_true",
        help="Prompt for feedback after prediction",
    )

    # feedback
    feedback_parser = subparsers.add_parser("feedback", help="View feedback statistics")
    feedback_parser.add_argument("--list", action="store_true", help="List all feedback entries")
    feedback_parser.add_argument("--export", metavar="CSV_PATH", help="Export feedback as CSV")

    # app (Gradio web UI)
    app_parser = subparsers.add_parser("app", help="Launch Gradio web interface")
    app_parser.add_argument(
        "--port", type=int, default=7860, help="Port number (default: 7860)"
    )
    app_parser.add_argument(
        "--share", action="store_true", help="Create public Gradio link"
    )

    # fine-tune
    ft_parser = subparsers.add_parser("fine-tune", help="Fine-tune model from feedback")
    ft_parser.add_argument(
        "--checkpoint", default=None,
        help="Base checkpoint (default: checkpoints/best_model.pth)",
    )
    ft_parser.add_argument("--epochs", type=int, default=None, help="Fine-tune epochs")
    ft_parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    ft_parser.add_argument(
        "--min-feedback", type=int, default=None,
        help="Minimum feedback entries required to start fine-tuning",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "app":
        from zoo_audio.app import build_app
        build_app().launch(server_port=args.port, share=args.share)
    elif args.command == "download":
        from zoo_audio.download_dataset import main as download_main
        download_main()
    elif args.command == "train":
        from zoo_audio.train import main as train_main
        train_main()
    elif args.command == "evaluate":
        from zoo_audio.evaluate import main as evaluate_main
        evaluate_main()
    elif args.command == "predict":
        if args.feedback:
            from zoo_audio.predict import predict_with_feedback
            predict_with_feedback(args.audio, args.checkpoint, args.top_k)
        else:
            from zoo_audio.predict import predict
            predict(args.audio, args.checkpoint, args.top_k)
    elif args.command == "feedback":
        from zoo_audio.feedback import FeedbackStore
        store = FeedbackStore()
        if args.export:
            store.export_csv(args.export)
        elif args.list:
            store.list_entries()
        else:
            store.summary()
    elif args.command == "fine-tune":
        _run_fine_tune(args)


def _run_fine_tune(args):
    """Handle the fine-tune subcommand."""
    from pathlib import Path
    import yaml
    from zoo_audio.feedback import FeedbackStore, fine_tune, get_model_versions

    # Load config
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(str(config_path), "r") as f:
        config = yaml.safe_load(f)

    fb_config = config.get("feedback", {})

    # Resolve parameters: CLI args override config, which override defaults
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = str(Path(config["data"]["checkpoint_dir"]) / "best_model.pth")

    epochs = args.epochs if args.epochs is not None else fb_config.get("fine_tune_epochs", 10)
    lr = args.lr if args.lr is not None else fb_config.get("fine_tune_lr", 0.0001)
    min_feedback = (
        args.min_feedback if args.min_feedback is not None
        else fb_config.get("min_feedback", 5)
    )
    freeze_blocks = fb_config.get("freeze_conv_blocks", 2)

    if not Path(checkpoint).exists():
        print(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    # Load feedback
    feedback_dir = fb_config.get("feedback_dir", "data/feedback")
    store = FeedbackStore(feedback_dir)
    entries = store.load_all()

    if len(entries) < min_feedback:
        print(
            f"Not enough feedback to fine-tune. "
            f"Have {len(entries)}, need at least {min_feedback}."
        )
        sys.exit(1)

    print(f"Loaded {len(entries)} feedback entries")
    print(f"Base checkpoint: {checkpoint}")

    # Show existing versions
    versions = get_model_versions(config["data"]["checkpoint_dir"])
    if versions:
        print("\nExisting model versions:")
        for v in versions:
            acc = f"{v['val_accuracy']:.4f}" if v['val_accuracy'] is not None else "N/A"
            print(f"  {v['version']:>4s}  classes={v['num_classes']}  "
                  f"acc={acc}  feedback={v['feedback_count']}")

    new_path = fine_tune(
        config=config,
        feedback_entries=entries,
        base_checkpoint=checkpoint,
        epochs=epochs,
        lr=lr,
        freeze_conv_blocks=freeze_blocks,
    )
    print(f"\nFine-tuning complete. New checkpoint: {new_path}")


if __name__ == "__main__":
    main()
