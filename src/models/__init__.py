"""Neural network models for LSC gesture recognition."""

from src.models.static import StaticMLP, StaticMobileNet, StaticMobileNetV2
from src.models.dynamic import DynamicGRU, DynamicBiGRU, DynamicLSTM

__all__ = [
    "StaticMLP",
    "StaticMobileNet",
    "StaticMobileNetV2",
    "DynamicGRU",
    "DynamicBiGRU",
    "DynamicLSTM",
    "get_model",
]


def get_model(
    model_type: str,
    num_classes: int,
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
):
    """
    Factory function to get a model by name.

    Args:
        model_type: 'static', 'mobilenet', 'mobilenet_v2', 'gru', 'bigru', 'lstm'
        num_classes: Number of output classes
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers (RNN) or hidden layers (MLP)
        dropout: Dropout rate
        freeze_backbone: Freeze backbone for mobilenet models (transfer learning)
    """
    if model_type == "static":
        return StaticMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    elif model_type == "mobilenet":
        # MobileNetV3-Small with deeper head (good for frozen backbone)
        return StaticMobileNet(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "mobilenet_v2":
        # MobileNetV2 with simple head (good for fine-tuning full model)
        return StaticMobileNetV2(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "gru":
        return DynamicGRU(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "bigru":
        return DynamicBiGRU(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "lstm":
        return DynamicLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown model: {model_type}. "
            "Choose: static, mobilenet, mobilenet_v2, gru, bigru, lstm"
        )
