"""
Model factory module for LSC gesture recognition.

This module contains only model definition logic and has minimal dependencies
(torch, torchvision), making it safe to import in lightweight environments
(like API Docker containers) without pulling in heavy training dependencies.
"""

import torch.nn as nn
from torchvision import models


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
