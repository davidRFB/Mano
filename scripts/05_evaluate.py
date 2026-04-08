#!/usr/bin/env python3
"""
Evaluate trained models and generate metrics.

Usage:
    python scripts/05_evaluate.py --run-id <mlflow_run_id>
    python scripts/05_evaluate.py --run-name <mlflow_run_name>
    python scripts/05_evaluate.py --list                    # List runs from default experiment
    python scripts/05_evaluate.py --list --experiment foo   # List runs from specific experiment
    python scripts/05_evaluate.py --list-all                # List runs from ALL experiments

Outputs:
    - Accuracy metrics
    - Confusion matrix
    - Classification report
    - Saves figures to blog/figures/
"""

import argparse
import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing.dataset import create_dataloaders
from src.training import load_model_from_mlflow, list_runs, list_all_runs, list_experiments
from src.training.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    print_classification_report,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_DIR = Path("blog/figures")


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
    parser.add_argument("--run-id", type=str, help="MLflow run ID")
    parser.add_argument("--run-name", type=str, help="MLflow run name")
    parser.add_argument("--list", action="store_true", help="List runs from experiment")
    parser.add_argument("--list-all", action="store_true", help="List runs from ALL experiments")
    parser.add_argument("--experiment", type=str, default="lsc_letters", help="Experiment name for --list")
    parser.add_argument("--data-dir", type=str, default="data/raw_landmarks")
    args = parser.parse_args()

    def fmt_acc(val: float | None) -> str:
        return f"{val:.1%}" if val is not None else "  -  "

    def print_run(run: dict, show_experiment: bool = False) -> None:
        train = fmt_acc(run["train_acc"])
        val = fmt_acc(run["val_acc"])
        test = fmt_acc(run["test_acc"])
        name = run["run_name"] or "unnamed"
        exp = f"{run['experiment']:<15}  " if show_experiment else ""
        print(f"  {exp}{name:<25}  train={train}  val={val}  test={test}  {run['status']:<10}  {run['run_id']}")

    # List all runs mode
    if args.list_all:
        print("All MLflow runs:")
        experiments = list_experiments()
        print(f"Experiments: {experiments}\n")
        print(f"  {'experiment':<15}  {'run_name':<25}  {'train':<10}  {'val':<10}  {'test':<10}  {'status':<10}  run_id")
        print("-" * 130)
        for run in list_all_runs(limit=20):
            print_run(run, show_experiment=True)
        return

    # List runs from specific experiment
    if args.list:
        print(f"MLflow runs (experiment: {args.experiment}):")
        print(f"  {'run_name':<25}  {'train':<10}  {'val':<10}  {'test':<10}  {'status':<10}  run_id")
        print("-" * 115)
        runs = list_runs(experiment_name=args.experiment, limit=10)
        if not runs:
            print(f"  No runs found. Available experiments: {list_experiments()}")
        for run in runs:
            print_run(run, show_experiment=False)
        return

    if not args.run_id and not args.run_name:
        print("Error: Must provide --run-id or --run-name (or use --list)")
        return

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Load model from MLflow
    try:
        model, config = load_model_from_mlflow(
            run_id=args.run_id,
            run_name=args.run_name,
            device=DEVICE,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    classes = config.get("classes", [])
    feature_mode = config.get("feature_mode", "xy_angles")
    model_type = config.get("model", "bigru")

    print(f"Model: {model_type}")
    print(f"Classes: {classes}")
    print(f"Features: {feature_mode}")

    # Determine output mode
    output_mode = "static" if model_type == "static" else "sequence"

    # Load data
    train_loader, val_loader, _, _, _, _ = create_dataloaders(
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
