"""
RNN models for dynamic gesture classification.

Used for letters that require movement (J, H, Z, Ñ, S in LSC).
Input: Sequence of landmark features (seq_len, feature_dim)
"""

import torch
import torch.nn as nn


class DynamicGRU(nn.Module):
    """
    GRU model for sequence classification.

    Input: (batch, seq_len, input_dim)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 56,
        num_classes: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        # hidden: (num_layers, batch, hidden_dim)
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        return self.classifier(last_hidden)


class DynamicBiGRU(nn.Module):
    """
    Bidirectional GRU for sequence classification.

    Processes sequence in both directions for better context.
    """

    def __init__(
        self,
        input_dim: int = 56,
        num_classes: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Bidirectional doubles the hidden dimension
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        # hidden: (num_layers * 2, batch, hidden_dim)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.classifier(combined)


class DynamicLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence classification.
    """

    def __init__(
        self,
        input_dim: int = 56,
        num_classes: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.classifier(combined)


if __name__ == "__main__":
    # Test all models
    batch, seq_len, input_dim = 8, 20, 56
    num_classes = 5
    x = torch.randn(batch, seq_len, input_dim)

    for name, Model in [("GRU", DynamicGRU), ("BiGRU", DynamicBiGRU), ("LSTM", DynamicLSTM)]:
        model = Model(input_dim=input_dim, num_classes=num_classes)
        out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:8s} | Input: {x.shape} -> Output: {out.shape} | Params: {params:,}")
