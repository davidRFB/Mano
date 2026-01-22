#!/usr/bin/env python3
"""
Train gesture recognition models.

Usage:
    # Train static model (for non-movement letters)
    python scripts/04_train.py --model static

    # Train dynamic model (for J, H, Z, Ñ, S)
    python scripts/04_train.py --model bigru --letters j,h,z,nn,s

    # Train on all letters with sequence model
    python scripts/04_train.py --model bigru --epochs 100

Options:
    --model: static, gru, bigru, lstm (default: bigru)
    --letters: Comma-separated letters to train on (default: all)
    --epochs: Training epochs (default: 100)
    --lr: Learning rate (default: 0.001)
    --batch-size: Batch size (default: 32)
    --features: Feature mode (default: xy_angles)
"""

import argparse
from pathlib import Path

import torch
import mlflow

from src.data.dataset import create_dataloaders, DYNAMIC_LETTERS
from src.models import get_model
from src.training.trainer import Trainer

# Paths
DATA_DIR = Path("data/raw_landmarks")
CHECKPOINT_DIR = Path("models/checkpoints")
MLFLOW_URI = f"file://{Path('models/mlruns').absolute()}"


def main():
    parser = argparse.ArgumentParser(description="Train LSC gesture model")
    parser.add_argument("--model", type=str, default="bigru",
                        choices=["static", "gru", "bigru", "lstm"])
    parser.add_argument("--letters", type=str, default=None,
                        help="Comma-separated letters (e.g., 'a,b,c' or 'j,h,z,nn,s')")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--features", type=str, default="xy_angles")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--experiment", type=str, default="lsc_letters")
    args = parser.parse_args()

    # Parse letters
    letters = None
    if args.letters:
        letters = [l.strip().lower() for l in args.letters.split(",")]

    # Determine output mode
    output_mode = "static" if args.model == "static" else "sequence"

    print("=" * 60)
    print("LSC Gesture Recognition Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Letters: {letters if letters else 'all'}")
    print(f"Features: {args.features}")
    print(f"Output mode: {output_mode}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    # Create data loaders
    train_loader, val_loader, num_classes, classes, feature_dim = create_dataloaders(
        data_dir=DATA_DIR,
        feature_mode=args.features,
        output_mode=output_mode,
        batch_size=args.batch_size,
        letters=letters,
    )

    print(f"\nClasses ({num_classes}): {classes}")
    print(f"Feature dim: {feature_dim}")

    # Create model
    model = get_model(
        model_type=args.model,
        num_classes=num_classes,
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model": args.model,
            "letters": args.letters or "all",
            "features": args.features,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "num_classes": num_classes,
            "feature_dim": feature_dim,
        })

        # Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=args.lr,
        )

        # Create checkpoint directory
        run_checkpoint_dir = CHECKPOINT_DIR / mlflow.active_run().info.run_id
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history = trainer.fit(
            epochs=args.epochs,
            checkpoint_dir=run_checkpoint_dir,
            early_stop_patience=20,
        )

        # Log metrics
        mlflow.log_metric("best_val_acc", trainer.best_val_acc)
        for i, (train_loss, val_acc) in enumerate(zip(history["train_loss"], history["val_acc"])):
            mlflow.log_metrics({"train_loss": train_loss, "val_acc": val_acc}, step=i)

        # Save final checkpoint with config
        final_path = run_checkpoint_dir / "final.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "model": args.model,
                "num_classes": num_classes,
                "input_dim": feature_dim,
                "hidden_dim": args.hidden_dim,
                "classes": classes,
                "feature_mode": args.features,
            },
            "best_val_acc": trainer.best_val_acc,
        }, final_path)

        print(f"\nTraining complete!")
        print(f"Best validation accuracy: {trainer.best_val_acc:.1%}")
        print(f"Checkpoint saved: {final_path}")


if __name__ == "__main__":
    main()
