"""
Training script for landmark-based LSC gesture recognition.

Usage:
    python -m src.cv_model.landmarks_train --model bigru --epochs 50
    python -m src.cv_model.landmarks_train --model gru_attention --hidden-dim 256

Models available: gru, bigru, gru_attention, lstm
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

from src.cv_model.landmarks_preprocessing import (
    create_dataloaders,
    FEATURE_MODES,
    DEFAULT_FEATURE_MODE,
)
from src.cv_model.landmarks_model import get_landmarks_model


# Configuration
MODELS_DIR = Path("models")
MLFLOW_TRACKING_URI = str((MODELS_DIR / "mlruns").absolute())
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

    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
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

    for sequences, labels in data_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * sequences.size(0)
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
    hidden_dim: int,
    num_layers: int,
    feature_mode: str,
    feature_dim: int,
) -> str:
    """Save model checkpoint to MLflow artifacts."""
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"landmarks_{model_name}_acc{val_acc:.2f}_{timestamp}.pth"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
        "model_name": model_name,
        "classes": classes,
        "num_classes": len(classes),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "feature_mode": feature_mode,
        "feature_dim": feature_dim,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / filename
        torch.save(checkpoint, filepath)
        mlflow.log_artifact(str(filepath), artifact_path="checkpoints")

    return filename


def train(
    model_name: str = "bigru",
    hidden_dim: int = 128,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    patience: int = 15,
    experiment_name: str = "lsc_landmarks_recognition",
    feature_mode: str = DEFAULT_FEATURE_MODE,
) -> None:
    """
    Main training function with MLflow tracking.

    Args:
        model_name: Model architecture (gru, bigru, gru_attention, lstm)
        hidden_dim: Hidden dimension for RNN
        num_layers: Number of RNN layers
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        dropout: Dropout rate
        patience: Early stopping patience
        experiment_name: MLflow experiment name
        feature_mode: Feature extraction mode (xy, xyz, xy_angles, etc.)
    """
    # Setup MLflow
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    print("=" * 60)
    print("LSC Landmarks Recognition Training")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Feature mode: {feature_mode} ({FEATURE_MODES[feature_mode]} features)")
    print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
    print("=" * 60)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, num_classes, classes, feature_dim = create_dataloaders(
        batch_size=batch_size,
        feature_mode=feature_mode,
    )

    print(f"Classes ({num_classes}): {classes}")
    print(f"Feature dimension: {feature_dim}")

    # Create model
    print(f"\nInitializing {model_name} model...")
    model = get_landmarks_model(
        model_name=model_name,
        num_classes=num_classes,
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate / 100)

    # Start MLflow run
    run_name = f"landmarks_{model_name}_h{hidden_dim}_l{num_layers}"
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(
            {
                "model_type": "landmarks_sequence",
                "model_name": model_name,
                "feature_mode": feature_mode,
                "feature_dim": feature_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
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
                "total_params": total_params,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "test_samples": len(test_loader.dataset),
            }
        )

        # Training loop
        best_val_loss = float("inf")
        best_val_acc = 0.0
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

            # Save best model (based on validation loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                epochs_without_improvement = 0
                best_checkpoint_name = save_checkpoint_to_mlflow(
                    model, optimizer, epoch, val_acc, model_name, classes,
                    hidden_dim, num_layers, feature_mode, feature_dim
                )
                print(f"  -> New best! (loss: {val_loss:.4f}) Saved to MLflow")
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(val_loss not improving for {patience} epochs)"
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

        # Log the final model
        if best_checkpoint_name:
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"lsc_landmarks_{model_name}",
            )

        run_id = mlflow.active_run().info.run_id

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"MLflow experiment: {experiment_name}")
        print(f"MLflow run ID: {run_id}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train landmark-based LSC gesture recognition model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bigru",
        choices=["gru", "bigru", "gru_attention", "lstm"],
        help="Model architecture",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for RNN",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of RNN layers",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument(
        "--experiment",
        type=str,
        default="lsc_landmarks_recognition",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=DEFAULT_FEATURE_MODE,
        choices=list(FEATURE_MODES.keys()),
        help=f"Feature mode: {', '.join(f'{k}({v})' for k, v in FEATURE_MODES.items())}",
    )

    args = parser.parse_args()

    train(
        model_name=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        experiment_name=args.experiment,
        feature_mode=args.features,
    )


if __name__ == "__main__":
    main()
