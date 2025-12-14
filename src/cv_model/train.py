"""
Training script for LSC gesture recognition model.

Usage:
    python -m src.cv_model.train --model mobilenet_v2 --epochs 30
    python -m src.cv_model.train --experiment my_experiment --model resnet18

Models available: mobilenet_v2, mobilenet_v3_small, efficientnet_b0, resnet18
"""

import argparse
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import mlflow.pytorch

from src.cv_model.preprocessing import create_dataloaders
from src.cv_model.model_factory import get_model


# Configuration
MODELS_DIR = Path("models")
MLFLOW_TRACKING_URI = f"file:///{MODELS_DIR.absolute()}/mlruns"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def save_checkpoint_to_mlflow(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    model_name: str,
    classes: list[str],
) -> str:
    """Save model checkpoint to MLflow artifacts only."""
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_acc{val_acc:.2f}_{timestamp}.pth"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
        "model_name": model_name,
        "classes": classes,
        "num_classes": len(classes),
    }

    # Save to temp file and log to MLflow
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / filename
        torch.save(checkpoint, filepath)
        mlflow.log_artifact(str(filepath), artifact_path="checkpoints")

    return filename


def train(
    model_name: str = "mobilenet_v2",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    experiment_name: str = "lsc_gesture_recognition",
) -> None:
    """
    Main training function with MLflow tracking.

    Args:
        model_name: Model architecture to use
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        experiment_name: MLflow experiment name
    """
    # Setup MLflow
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    print("=" * 60)
    print("LSC Gesture Recognition Training")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
    print("=" * 60)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, num_classes, classes = create_dataloaders(
        batch_size=batch_size
    )

    print(f"Classes ({num_classes}): {classes}")

    # Create model
    print(f"\nInitializing {model_name} with pretrained weights...")
    model = get_model(model_name, num_classes, pretrained=True)
    model = model.to(DEVICE)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate / 100)

    # Start MLflow run
    run_name = f"{model_name}_lr{learning_rate}_bs{batch_size}"
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "patience": patience,
                "num_classes": num_classes,
                "classes": ",".join(classes),
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "device": str(DEVICE),
                "pretrained": True,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "test_samples": len(test_loader.dataset),
            }
        )

        # Log model architecture info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_params(
            {
                "total_params": total_params,
                "trainable_params": trainable_params,
            }
        )

        # Training loop
        best_val_loss = float("inf")  # Track loss (lower is better)
        best_val_acc = 0.0  # Still track for reporting
        epochs_without_improvement = 0
        best_checkpoint_name = None

        print("\nStarting training...")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )

            # Validate
            val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

            # Update scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            # Logging
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best model to MLflow (based on validation LOSS - lower is better)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc  # Track best acc at best loss
                epochs_without_improvement = 0
                best_checkpoint_name = save_checkpoint_to_mlflow(
                    model, optimizer, epoch, val_acc, model_name, classes
                )
                print(f"  ↳ New best! (loss: {val_loss:.4f}) Saved to MLflow")
            else:
                epochs_without_improvement += 1

            # Early stopping (based on validation loss)
            if epochs_without_improvement >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch} (val_loss not improving for {patience} epochs)"
                )
                mlflow.log_param("early_stopped_epoch", epoch)
                break

        print("-" * 60)

        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

        # Log final metrics
        mlflow.log_metrics(
            {
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        # Log the final model to MLflow model registry
        if best_checkpoint_name:
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"lsc_{model_name}",
            )

        run_id = mlflow.active_run().info.run_id

        print("\n" + "=" * 60)
        print(f"Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"MLflow experiment: {experiment_name}")
        print(f"MLflow run ID: {run_id}")
        print(f"\nTo use this model for inference:")
        print(f"  python -m src.cv_model.inference --run-id {run_id}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train LSC gesture recognition model")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "mobilenet_v3_small", "efficientnet_b0", "resnet18"],
        help="Model architecture",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument(
        "--experiment",
        type=str,
        default="lsc_gesture_recognition",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        experiment_name=args.experiment,
    )


if __name__ == "__main__":
    main()

