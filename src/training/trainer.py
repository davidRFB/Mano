"""
Training loop and utilities.
"""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Simple training loop with checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device = DEVICE,
        config: dict | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0  # Track for logging purposes
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in self.val_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def fit(
        self,
        epochs: int,
        checkpoint_dir: Path | None = None,
        early_stop_patience: int = 20,
        verbose: bool = True,
        on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
    ) -> dict[str, list[float]]:
        """
        Train the model.

        Args:
            epochs: Number of training epochs
            checkpoint_dir: Directory to save checkpoints
            early_stop_patience: Stop if no improvement for N epochs
            verbose: Print progress
            on_epoch_end: Optional callback called after each epoch with
                          (epoch, metrics_dict) for live logging

        Returns:
            Training history dict
        """
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        no_improve = 0

        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Print detailed metrics periodically
            if verbose:
                best_str = f"{self.best_val_loss:.4f}" if self.best_val_loss < float("inf") else "-"
                print(
                    f"\nEpoch {epoch + 1}/{epochs} - "
                    f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.1%}, "
                    f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.1%}, "
                    f"best_loss: {best_str}"
                )

            # Call epoch callback for live logging (e.g., MLflow)
            if on_epoch_end:
                metrics = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                on_epoch_end(epoch, metrics)

            # Check for improvement (based on lowest validation loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc  # Track corresponding accuracy
                no_improve = 0

                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir / "best.pth")
            else:
                no_improve += 1

            # Early stopping
            if no_improve >= early_stop_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        return self.history

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint with config."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
