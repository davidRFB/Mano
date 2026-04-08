#!/usr/bin/env python3
"""
Train gesture recognition models.

Usage:
    # Train static MLP on landmarks (first frame only)
    python scripts/04_train.py --data-dir data/raw_landmarks --model static

    # Train BiGRU on landmark sequences
    python scripts/04_train.py --data-dir data/raw_landmarks --model bigru

    # Train MobileNetV2 on images
    python scripts/04_train.py --data-dir data/raw --model mobilenet_v2

    # Train on specific letters only
    python scripts/04_train.py --data-dir data/raw_landmarks --model bigru --letters j,h,z,nn,s

Options:
    --data-dir: Data directory (required, e.g., data/raw_landmarks or data/raw)
    --model: static, gru, bigru, lstm, mobilenet, mobilenet_v2 (default: bigru)
    --letters: Comma-separated letters to train on (default: all)
    --epochs: Training epochs (default: 100)
    --lr: Learning rate (default: 0.001)
    --batch-size: Batch size (default: 32)
    --features: Feature mode for landmark models (default: xy_angles, ignored for mobilenet)
    --patience: Early stopping patience (default: 20)

Note: Uses stratified 70/15/15 train/val/test split for proper evaluation.
"""

import argparse
import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tempfile

import torch
import mlflow

from src.preprocessing.dataset import create_dataloaders, create_image_dataloaders, DYNAMIC_LETTERS
from src.models import get_model
from src.training.trainer import Trainer

# Paths
MLFLOW_URI = f"sqlite:///{Path('models/mlflow.db').absolute()}"
MLFLOW_ARTIFACTS = f"file://{Path('models/mlartifacts').absolute()}"


def main():
    parser = argparse.ArgumentParser(description="Train LSC gesture model")
    parser.add_argument("--model", type=str, default="bigru",
                        choices=["static", "gru", "bigru", "lstm", "mobilenet", "mobilenet_v2"])
    parser.add_argument("--letters", type=str, default=None,
                        help="Comma-separated letters (e.g., 'a,b,c' or 'j,h,z,nn,s')")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--features", type=str, default="xy_angles")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data directory (e.g., data/raw_landmarks or data/raw)")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--experiment", type=str, default="lsc_letters")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this specific run")
    parser.add_argument("--no-test", action="store_true",
                        help="Use 80/20 train/val split only (no test set)")
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                        help="Freeze MobileNet backbone (default: True)")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                        help="Train full MobileNet (unfreeze backbone)")
    parser.add_argument("--static-frame-override", type=str, default=None,
                        help="Override frame index for static mode per class (e.g., 'j:18,h:15')")
    args = parser.parse_args()

    # Parse letters
    letters = None
    if args.letters:
        letters = [l.strip().lower() for l in args.letters.split(",")]

    # Parse static frame overrides
    static_frame_override = None
    if args.static_frame_override:
        static_frame_override = {}
        for pair in args.static_frame_override.split(","):
            cls, idx = pair.strip().split(":")
            static_frame_override[cls.strip().lower()] = int(idx.strip())

    # Determine output mode
    output_mode = "static" if args.model == "static" else "sequence"

    # Image-based models
    is_image_model = args.model in ("mobilenet", "mobilenet_v2")
    data_dir = Path(args.data_dir)

    # Feature mode is only relevant for landmark-based models
    feature_display = "images (N/A)" if is_image_model else args.features

    print("=" * 60)
    print("LSC Gesture Recognition Training")
    print("=" * 60)
    # Split ratios
    if args.no_test:
        train_ratio, val_ratio, test_ratio = 0.80, 0.20, 0.0
        split_str = "80% train / 20% val (no test)"
    else:
        train_ratio, val_ratio, test_ratio = 0.70, 0.15, 0.15
        split_str = "70% train / 15% val / 15% test"

    print(f"Model: {args.model}")
    if is_image_model:
        print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Data: {data_dir}")
    print(f"Letters: {letters if letters else 'all'}")
    print(f"Features: {feature_display}")
    print(f"Output mode: {output_mode}")
    print(f"Split: {split_str}")
    print(f"Epochs: {args.epochs}")
    print(f"Early stop patience: {args.patience}")
    print("=" * 60)

    # Create data loaders
    if is_image_model:
        train_loader, val_loader, test_loader, num_classes, classes, feature_dim = create_image_dataloaders(
            data_dir=data_dir,
            batch_size=args.batch_size,
            letters=letters,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    else:
        train_loader, val_loader, test_loader, num_classes, classes, feature_dim = create_dataloaders(
            data_dir=data_dir,
            feature_mode=args.features,
            output_mode=output_mode,
            batch_size=args.batch_size,
            letters=letters,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            static_frame_override=static_frame_override,
        )

    print(f"\nClasses ({num_classes}): {classes}")
    print(f"Feature dim: {feature_dim}")

    # Create model
    model = get_model(
        model_type=args.model,
        num_classes=num_classes,
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        freeze_backbone=args.freeze_backbone,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Create experiment with artifact location if it doesn't exist
    if mlflow.get_experiment_by_name(args.experiment) is None:
        mlflow.create_experiment(args.experiment, artifact_location=MLFLOW_ARTIFACTS)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_params({
            "model": args.model,
            "data_dir": str(data_dir),
            "letters": args.letters or "all",
            "features": feature_display,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "patience": args.patience,
            "num_classes": num_classes,
            "feature_dim": feature_dim,
            "split": "70/15/15",
            "freeze_backbone": args.freeze_backbone if is_image_model else "N/A",
        })

        # Model config (saved in checkpoints for inference)
        model_config = {
            "model": args.model,
            "data_dir": str(data_dir),
            "num_classes": num_classes,
            "input_dim": feature_dim,
            "hidden_dim": args.hidden_dim,
            "classes": classes,
            "feature_mode": None if is_image_model else args.features,
            "static_frame_override": static_frame_override,
        }

        # Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=args.lr,
            config=model_config,
        )

        # Define callback for live MLflow logging
        def log_epoch_metrics(epoch: int, metrics: dict[str, float]) -> None:
            mlflow.log_metrics(metrics, step=epoch)

        # Use temp directory for intermediate checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            history = trainer.fit(
                epochs=args.epochs,
                checkpoint_dir=checkpoint_dir,
                early_stop_patience=args.patience,
                on_epoch_end=log_epoch_metrics,
            )

            # Log best validation metrics
            mlflow.log_metric("best_val_loss", trainer.best_val_loss)
            mlflow.log_metric("best_val_acc", trainer.best_val_acc)

            # Log best checkpoint (now includes config)
            best_path = checkpoint_dir / "best.pth"
            if best_path.exists():
                mlflow.log_artifact(best_path, artifact_path="model")

            # Load best model for test evaluation
            if best_path.exists():
                checkpoint = torch.load(best_path, map_location=trainer.device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])

            # Evaluate on test set (if available)
            test_acc = None
            if not args.no_test and len(test_loader.dataset) > 0:
                trainer.val_loader = test_loader
                test_loss, test_acc = trainer.validate()
                mlflow.log_metric("test_loss", test_loss)
                mlflow.log_metric("test_acc", test_acc)

        run_id = mlflow.active_run().info.run_id
        print(f"\nTraining complete!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Best validation accuracy: {trainer.best_val_acc:.1%}")
        if test_acc is not None:
            print(f"Test accuracy: {test_acc:.1%}")
        print(f"Model artifacts logged to MLflow run: {run_id}")


if __name__ == "__main__":
    main()
