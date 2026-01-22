"""
Evaluation metrics for gesture recognition.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = DEVICE,
) -> float:
    """Compute accuracy on a dataset."""
    model.eval()
    correct, total = 0, 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    """Get all predictions and true labels."""
    model.eval()
    all_preds, all_labels = [], []

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def compute_confusion_matrix(
    model: nn.Module,
    data_loader: DataLoader,
    classes: list[str],
    device: torch.device = DEVICE,
) -> np.ndarray:
    """Compute confusion matrix."""
    preds, labels = get_predictions(model, data_loader, device)
    return confusion_matrix(labels, preds, labels=range(len(classes)))


def print_classification_report(
    model: nn.Module,
    data_loader: DataLoader,
    classes: list[str],
    device: torch.device = DEVICE,
) -> str:
    """Print sklearn classification report."""
    preds, labels = get_predictions(model, data_loader, device)
    return classification_report(labels, preds, target_names=classes, zero_division=0)
