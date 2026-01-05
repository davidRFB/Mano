"""
Models for word-level sign language recognition.

Handles longer sequences (holistic landmarks) and larger vocabulary.

Models available:
- gru: Basic GRU
- bigru: Bidirectional GRU
- gru_attention: GRU with attention
- transformer: Transformer encoder
"""

import math
import torch
import torch.nn as nn


class WordsGRU(nn.Module):
    """Basic GRU for word sequences."""

    def __init__(
        self,
        input_dim: int = 130,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        return self.classifier(last_hidden)


class WordsBiGRU(nn.Module):
    """Bidirectional GRU for word sequences."""

    def __init__(
        self,
        input_dim: int = 130,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
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

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.classifier(combined)


class WordsGRUAttention(nn.Module):
    """GRU with attention for word sequences."""

    def __init__(
        self,
        input_dim: int = 130,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
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
        gru_out, _ = self.gru(x)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        return self.classifier(context)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class WordsTransformer(nn.Module):
    """Transformer encoder for word sequences."""

    def __init__(
        self,
        input_dim: int = 130,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 100,
        dropout: float = 0.3,
        nhead: int = 8,
        max_seq_len: int = 100,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Global average pooling over sequence
        x = x.mean(dim=1)

        return self.classifier(x)


class WordsLSTM(nn.Module):
    """Bidirectional LSTM for word sequences."""

    def __init__(
        self,
        input_dim: int = 130,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
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
        _, (hidden, _) = self.lstm(x)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.classifier(combined)


def get_words_model(
    model_name: str,
    num_classes: int,
    input_dim: int = 130,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    max_seq_len: int = 90,
) -> nn.Module:
    """
    Factory function to get a word sequence model.

    Args:
        model_name: One of 'gru', 'bigru', 'gru_attention', 'transformer', 'lstm'
        num_classes: Number of output classes (vocabulary size)
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout rate
        max_seq_len: Maximum sequence length (for transformer)

    Returns:
        PyTorch model
    """
    models = {
        "gru": WordsGRU,
        "bigru": WordsBiGRU,
        "gru_attention": WordsGRUAttention,
        "transformer": WordsTransformer,
        "lstm": WordsLSTM,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models.keys())}")

    model_class = models[model_name]

    if model_name == "transformer":
        return model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

    return model_class(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )


if __name__ == "__main__":
    batch_size = 4
    seq_len = 90
    input_dim = 130
    num_classes = 500

    x = torch.randn(batch_size, seq_len, input_dim)

    for model_name in ["gru", "bigru", "gru_attention", "transformer", "lstm"]:
        model = get_words_model(model_name, num_classes, input_dim)
        out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{model_name:15s} | Output: {out.shape} | Params: {params:,}")
