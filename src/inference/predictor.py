"""
Model wrapper for inference.
"""

from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from src.preprocessing import (
    normalize_landmarks,
    extract_features,
    extract_single_frame_features,
    DEFAULT_FEATURE_MODE,
)
from src.models import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    """
    Unified predictor for both static and dynamic models.

    Handles landmark extraction, buffering for sequences, and prediction.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        model_type: str = "bigru",
        classes: list[str] | None = None,
        feature_mode: str = DEFAULT_FEATURE_MODE,
        sequence_length: int = 20,
        device: torch.device = DEVICE,
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            model_type: Model type (static, gru, bigru, lstm)
            classes: List of class names
            feature_mode: Feature extraction mode
            sequence_length: Buffer size for sequence models
            device: Torch device
        """
        self.device = device
        self.model_type = model_type
        self.feature_mode = feature_mode
        self.sequence_length = sequence_length
        self.classes = classes or []
        self.is_sequence_model = model_type != "static"

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Get model config from checkpoint
        config = checkpoint.get("config", {})
        num_classes = config.get("num_classes", len(self.classes))
        input_dim = config.get("input_dim", 56)
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 2)

        if "classes" in config and not self.classes:
            self.classes = config["classes"]

        # Build and load model
        self.model = get_model(
            model_type=model_type,
            num_classes=num_classes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        # Frame buffer for sequence models
        self.buffer: deque = deque(maxlen=sequence_length)

    def reset_buffer(self) -> None:
        """Clear the frame buffer."""
        self.buffer.clear()

    def buffer_ready(self) -> bool:
        """Check if buffer has enough frames for prediction."""
        return len(self.buffer) >= self.sequence_length

    def add_frame(self, landmarks: np.ndarray) -> None:
        """
        Add a frame of landmarks to the buffer.

        Args:
            landmarks: (21, 3) raw landmarks from MediaPipe
        """
        self.buffer.append(landmarks.copy())

    @torch.no_grad()
    def predict_static(self, landmarks: np.ndarray) -> tuple[str, float]:
        """
        Predict from a single frame (for static model).

        Args:
            landmarks: (21, 3) raw landmarks

        Returns:
            (predicted_class, confidence)
        """
        # Normalize and extract features
        normalized = normalize_landmarks(landmarks)
        features = extract_single_frame_features(normalized, self.feature_mode)

        # Predict
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        outputs = self.model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = probs.max(1)

        return self.classes[pred_idx.item()], confidence.item()

    @torch.no_grad()
    def predict_sequence(self) -> tuple[str, float]:
        """
        Predict from buffered sequence (for RNN models).

        Returns:
            (predicted_class, confidence)
        """
        if not self.buffer_ready():
            raise ValueError(f"Buffer not ready: {len(self.buffer)}/{self.sequence_length}")

        # Convert buffer to sequence
        sequence = np.stack(list(self.buffer), axis=0)  # (seq_len, 21, 3)
        normalized = normalize_landmarks(sequence)
        features = extract_features(normalized, self.feature_mode)

        # Predict
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        outputs = self.model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = probs.max(1)

        return self.classes[pred_idx.item()], confidence.item()

    def predict(self, landmarks: np.ndarray | None = None) -> tuple[str, float] | None:
        """
        Unified predict interface.

        For static models: pass landmarks directly.
        For sequence models: landmarks are added to buffer, prediction when ready.

        Args:
            landmarks: (21, 3) raw landmarks (required for static, optional for sequence)

        Returns:
            (predicted_class, confidence) or None if buffer not ready
        """
        if self.model_type == "static":
            if landmarks is None:
                raise ValueError("landmarks required for static model")
            return self.predict_static(landmarks)
        else:
            if landmarks is not None:
                self.add_frame(landmarks)

            if self.buffer_ready():
                return self.predict_sequence()
            return None
