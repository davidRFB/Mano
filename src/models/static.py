"""
Static MLP model for single-frame gesture classification.

Used for letters that don't require movement (A-I, K-R, T-Y, etc.)
Input: Single frame of extracted features (e.g., 56 features for xy_angles mode)
"""

import torch
import torch.nn as nn


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


if __name__ == "__main__":
    # Test
    model = StaticMLP(input_dim=56, num_classes=22)
    x = torch.randn(8, 56)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
