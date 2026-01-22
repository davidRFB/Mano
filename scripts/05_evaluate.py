#!/usr/bin/env python3
"""
Evaluate trained models and generate metrics.

Usage:
    python scripts/05_evaluate.py --checkpoint models/checkpoints/<run_id>/best.pth

Outputs:
    - Accuracy metrics
    - Confusion matrix
    - Classification report
    - Saves figures to blog/figures/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import create_dataloaders
from src.models import get_model
from src.training.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    print_classification_report,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_DIR = Path("blog/figures")


def load_model_from_checkpoint(checkpoint_path: Path):
    """Load model and config from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get("config", {})

    model = get_model(
        model_type=config.get("model", "bigru"),
        num_classes=config["num_classes"],
        input_dim=config["input_dim"],
        hidden_dim=config.get("hidden_dim", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, config


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], save_path: Path) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LSC model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/raw_landmarks")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path)
    classes = config.get("classes", [])
    feature_mode = config.get("feature_mode", "xy_angles")
    model_type = config.get("model", "bigru")

    print(f"Model: {model_type}")
    print(f"Classes: {classes}")
    print(f"Features: {feature_mode}")

    # Determine output mode
    output_mode = "static" if model_type == "static" else "sequence"

    # Load data
    train_loader, val_loader, _, _, _ = create_dataloaders(
        data_dir=Path(args.data_dir),
        feature_mode=feature_mode,
        output_mode=output_mode,
        letters=classes if classes else None,
    )

    # Compute metrics
    train_acc = compute_accuracy(model, train_loader, DEVICE)
    val_acc = compute_accuracy(model, val_loader, DEVICE)

    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.2%}")
    print(f"  Val:   {val_acc:.2%}")

    # Classification report
    print(f"\nClassification Report:")
    report = print_classification_report(model, val_loader, classes, DEVICE)
    print(report)

    # Confusion matrix
    cm = compute_confusion_matrix(model, val_loader, classes, DEVICE)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Save figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = FIGURES_DIR / f"confusion_matrix_{model_type}.png"
    plot_confusion_matrix(cm, classes, cm_path)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
