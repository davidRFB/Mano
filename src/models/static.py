"""
Static MLP model for single-frame gesture classification.

Used for letters that don't require movement (A-I, K-R, T-Y, etc.)
Input: Single frame of extracted features (e.g., 56 features for xy_angles mode)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class StaticMLP(nn.Module):
    """
    Simple MLP for static gesture classification.

    Input: (batch, feature_dim) - single frame features
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 56,
        num_classes: int = 22,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.model = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Second hidden layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output layer
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class StaticMobileNetV2(nn.Module):
    """
    MobileNetV2 for static gesture classification from images.

    Simple classifier head - works best when fine-tuning full model.
    """

    def __init__(
        self,
        num_classes: int = 22,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)

        if freeze_backbone and pretrained:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # Simple head: just replace final layer (like original cv_model)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class StaticMobileNet(nn.Module):
    """
    MobileNetV3-Small for static gesture classification from images.

    Simple classifier head - better for transfer learning.
    """

    def __init__(
        self,
        num_classes: int = 22,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)

        if freeze_backbone and pretrained:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # Simple head: just replace final layer (better for transfer learning)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    # Test
    model = StaticMLP(input_dim=56, num_classes=22)
    x = torch.randn(8, 56)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
