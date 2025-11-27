"""
Training script for LSC gesture recognition model.

Usage:
    python -m src.cv_model.train --model mobilenet_v2 --epochs 30

Models available: mobilenet_v2, mobilenet_v3_small, efficientnet_b0, resnet18
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from src.cv_model.preprocessing import create_dataloaders


# Configuration
MODELS_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Get a pretrained model with modified classifier for num_classes.

    Args:
        model_name: One of mobilenet_v2, mobilenet_v3_small, efficientnet_b0, resnet18
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        PyTorch model
    """
    weights = "DEFAULT" if pretrained else None

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            "Choose from: mobilenet_v2, mobilenet_v3_small, efficientnet_b0, resnet18"
        )

    return model


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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    model_name: str,
    classes: list[str],
    save_dir: Path = MODELS_DIR,
) -> Path:
    """Save model checkpoint."""
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_v1_acc{val_acc:.2f}_{timestamp}.pth"
    filepath = save_dir / filename

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
        "model_name": model_name,
        "classes": classes,
        "num_classes": len(classes),
    }

    torch.save(checkpoint, filepath)

    # Also save metadata as JSON
    metadata = {
        "model_name": model_name,
        "epoch": epoch,
        "val_acc": val_acc,
        "classes": classes,
        "num_classes": len(classes),
        "timestamp": timestamp,
    }
    json_path = filepath.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return filepath


def train(
    model_name: str = "mobilenet_v2",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
) -> None:
    """
    Main training function.

    Args:
        model_name: Model architecture to use
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
    """
    print("=" * 60)
    print("LSC Gesture Recognition Training")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
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

    # Training loop
    best_val_acc = 0.0
    epochs_without_improvement = 0

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

        # Logging
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, val_acc, model_name, classes
            )
            print(f"  ↳ New best! Saved to {checkpoint_path.name}")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print("-" * 60)

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    print("\n" + "=" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved to: {MODELS_DIR.absolute()}")
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

    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()

