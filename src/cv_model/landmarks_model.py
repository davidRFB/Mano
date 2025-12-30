"""
GRU-based models for landmark sequence classification.

Models available:
- landmarks_gru: Basic GRU classifier
- landmarks_bigru: Bidirectional GRU
- landmarks_gru_attention: GRU with attention mechanism
"""

import torch
import torch.nn as nn
from typing import Optional


class LandmarksGRU(nn.Module):
    """
    Basic GRU model for landmark sequence classification.

    Input: (batch, seq_len, 63) - 21 landmarks x 3 coords
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 27,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        gru_out, hidden = self.gru(x)
        # gru_out: (batch, seq_len, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim)

        # Use last hidden state
        last_hidden = hidden[-1]  # (batch, hidden_dim)

        # Classify
        out = self.classifier(last_hidden)
        return out


class LandmarksBiGRU(nn.Module):
    """
    Bidirectional GRU for landmark sequence classification.

    Processes sequence in both directions for better context.
    """

    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 27,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        gru_out, hidden = self.gru(x)
        # hidden: (num_layers * 2, batch, hidden_dim)

        # Concatenate final forward and backward hidden states
        forward_hidden = hidden[-2]  # (batch, hidden_dim)
        backward_hidden = hidden[-1]  # (batch, hidden_dim)
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        out = self.classifier(combined)
        return out


class LandmarksGRUAttention(nn.Module):
    """
    GRU with attention mechanism for landmark sequences.

    Learns to focus on important frames in the sequence.
    """

    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 27,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        gru_out, _ = self.gru(x)
        # gru_out: (batch, seq_len, hidden_dim * 2)

        # Compute attention weights
        attn_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Apply attention
        context = torch.sum(attn_weights * gru_out, dim=1)  # (batch, hidden_dim * 2)

        out = self.classifier(context)
        return out


class LandmarksLSTM(nn.Module):
    """
    LSTM alternative for landmark sequence classification.
    """

    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 27,
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (hidden, cell) = self.lstm(x)

        # Concatenate final forward and backward hidden states
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        out = self.classifier(combined)
        return out


def get_landmarks_model(
    model_name: str,
    num_classes: int,
    input_dim: int = 63,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Factory function to get a landmark sequence model.

    Args:
        model_name: One of 'gru', 'bigru', 'gru_attention', 'lstm'
        num_classes: Number of output classes
        input_dim: Input feature dimension (default 63 = 21 landmarks x 3)
        hidden_dim: GRU/LSTM hidden dimension
        num_layers: Number of recurrent layers
        dropout: Dropout rate

    Returns:
        PyTorch model
    """
    models = {
        "gru": LandmarksGRU,
        "bigru": LandmarksBiGRU,
        "gru_attention": LandmarksGRUAttention,
        "lstm": LandmarksLSTM,
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: {list(models.keys())}"
        )

    model_class = models[model_name]
    return model_class(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Test models
    batch_size = 4
    seq_len = 20
    input_dim = 63
    num_classes = 27

    x = torch.randn(batch_size, seq_len, input_dim)

    for model_name in ["gru", "bigru", "gru_attention", "lstm"]:
        model = get_landmarks_model(model_name, num_classes)
        out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{model_name:15s} | Output: {out.shape} | Params: {params:,}")
