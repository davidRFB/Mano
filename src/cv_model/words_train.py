"""
Training script for word-level LSC gesture recognition.

Usage:
    python -m src.cv_model.words_train --model bigru --epochs 100
    python -m src.cv_model.words_train --model transformer --hidden-dim 256
    python -m src.cv_model.words_train --model gru_attention --epochs 50

Models available: gru, bigru, gru_attention, transformer, lstm
"""

import argparse
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import mlflow
import mlflow.pytorch

from src.cv_model.words_preprocessing import (
    create_dataloaders,
    FEATURE_MODES,
    DEFAULT_FEATURE_MODE,
    MAX_SEQUENCE_LENGTH,
)
from src.cv_model.words_model import get_words_model

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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model. Returns (loss, accuracy, top5_accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    for sequences, labels in data_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * sequences.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Top-5 accuracy
        _, top5_pred = outputs.topk(5, dim=1)
        correct_top5 += top5_pred.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

    return running_loss / total, correct / total, correct_top5 / total


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
    max_seq_len: int,
) -> str:
    """Save model checkpoint to MLflow artifacts."""
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"words_{model_name}_acc{val_acc:.2f}_{timestamp}.pth"

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
        "max_seq_len": max_seq_len,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / filename
        torch.save(checkpoint, filepath)
        mlflow.log_artifact(str(filepath), artifact_path="checkpoints")

    return filename


def train(
    model_name: str = "bigru",
    hidden_dim: int = 256,
    num_layers: int = 2,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    patience: int = 20,
    experiment_name: str = "lsc_words_recognition",
    feature_mode: str = DEFAULT_FEATURE_MODE,
    max_seq_len: int = MAX_SEQUENCE_LENGTH,
    min_samples: int = 1,
) -> None:
    """Main training function with MLflow tracking."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    print("=" * 60)
    print("LSC Word Recognition Training")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Feature mode: {feature_mode} ({FEATURE_MODES[feature_mode]} features)")
    print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
    print("=" * 60)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, num_classes, classes, feature_dim = (
        create_dataloaders(
            batch_size=batch_size,
            feature_mode=feature_mode,
            max_seq_len=max_seq_len,
            min_samples_per_class=min_samples,
        )
    )

    print(f"Vocabulary size: {num_classes} words")
    print(f"Feature dimension: {feature_dim}")
    print(f"Sample words: {classes[:10]}...")

    # Create model
    print(f"\nInitializing {model_name} model...")
    model = get_words_model(
        model_name=model_name,
        num_classes=num_classes,
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss with label smoothing for large vocabulary
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    run_name = f"words_{model_name}_h{hidden_dim}_l{num_layers}_{num_classes}cls"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_type": "words_sequence",
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
                "max_seq_len": max_seq_len,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingWarmRestarts",
                "device": str(DEVICE),
                "total_params": total_params,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "test_samples": len(test_loader.dataset),
            }
        )

        best_val_loss = float("inf")
        best_val_acc = 0.0
        epochs_without_improvement = 0
        best_checkpoint_name = None

        print("\nStarting training...")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )

            val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, DEVICE)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_top5_acc": val_top5,
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                f"Val: {val_loss:.4f}/{val_acc:.4f} (top5: {val_top5:.4f}) | "
                f"LR: {current_lr:.6f} | {elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                epochs_without_improvement = 0
                best_checkpoint_name = save_checkpoint_to_mlflow(
                    model,
                    optimizer,
                    epoch,
                    val_acc,
                    model_name,
                    classes,
                    hidden_dim,
                    num_layers,
                    feature_mode,
                    feature_dim,
                    max_seq_len,
                )
                print(f"  -> New best! Saved to MLflow")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                mlflow.log_param("early_stopped_epoch", epoch)
                break

        print("-" * 60)

        # Final evaluation
        print("\nEvaluating on test set...")
        test_loss, test_acc, test_top5 = evaluate(model, test_loader, criterion, DEVICE)
        print(f"Test: Loss={test_loss:.4f} | Acc={test_acc:.4f} | Top5={test_top5:.4f}")

        mlflow.log_metrics(
            {
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_top5_acc": test_top5,
            }
        )

        if best_checkpoint_name:
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"lsc_words_{model_name}",
            )

        run_id = mlflow.active_run().info.run_id

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f} | Top-5: {test_top5:.4f}")
        print(f"MLflow run ID: {run_id}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train word-level LSC gesture recognition model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bigru",
        choices=["gru", "bigru", "gru_attention", "transformer", "lstm"],
        help="Model architecture",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument(
        "--experiment",
        type=str,
        default="lsc_words_recognition",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=DEFAULT_FEATURE_MODE,
        choices=list(FEATURE_MODES.keys()),
        help=f"Feature mode: {', '.join(f'{k}({v})' for k, v in FEATURE_MODES.items())}",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQUENCE_LENGTH,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples per class to include",
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
        max_seq_len=args.max_seq_len,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    main()
