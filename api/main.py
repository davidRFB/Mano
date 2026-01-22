"""
LSC Gesture Recognition API

FastAPI service for landmark-based gesture recognition.

Usage:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /              - API info
    GET  /health        - Health check
    POST /predict       - Predict from landmarks JSON
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data.preprocessing import (
    normalize_landmarks,
    extract_features,
    DEFAULT_FEATURE_MODE,
)
from src.models import get_model

# Configuration
CHECKPOINT_PATH = Path("models/checkpoints/latest/best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global state
model = None
config = {}

app = FastAPI(
    title="LSC Gesture Recognition API",
    description="Colombian Sign Language letter recognition from hand landmarks",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LandmarkFrame(BaseModel):
    """Single frame of 21 landmarks."""
    landmarks: list[list[float]]  # Shape: (21, 3)


class LandmarkSequence(BaseModel):
    """Sequence of landmark frames."""
    frames: list[list[list[float]]]  # Shape: (seq_len, 21, 3)


class PredictionResponse(BaseModel):
    """Prediction result."""
    letter: str
    confidence: float
    all_probs: Optional[dict[str, float]] = None


def load_model(checkpoint_path: Path):
    """Load model from checkpoint."""
    if not checkpoint_path.exists():
        return None, {}

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint.get("config", {})

    mdl = get_model(
        model_type=cfg.get("model", "bigru"),
        num_classes=cfg["num_classes"],
        input_dim=cfg["input_dim"],
        hidden_dim=cfg.get("hidden_dim", 128),
    )
    mdl.load_state_dict(checkpoint["model_state_dict"])
    mdl.to(DEVICE)
    mdl.eval()

    return mdl, cfg


@app.on_event("startup")
def startup():
    """Load model at startup."""
    global model, config
    print("=" * 50)
    print("Loading model...")

    model, config = load_model(CHECKPOINT_PATH)

    if model is None:
        print(f"Warning: No model at {CHECKPOINT_PATH}")
        print("API running without model. Train first.")
    else:
        print(f"Model: {config.get('model', 'unknown')}")
        print(f"Classes: {config.get('classes', [])}")
        print("API ready!")
    print("=" * 50)


@app.get("/")
def root():
    """API info."""
    return {
        "name": "LSC Gesture Recognition API",
        "version": "2.0.0",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "model_type": config.get("model"),
        "num_classes": config.get("num_classes"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: LandmarkSequence):
    """
    Predict gesture from landmark sequence.

    Input: JSON with 'frames' containing sequence of landmark arrays.
    Each frame is (21, 3) = 21 landmarks × 3 coordinates (x, y, z).

    Example:
    {
        "frames": [
            [[0.5, 0.5, 0.0], ...],  // Frame 1: 21 landmarks
            [[0.5, 0.5, 0.0], ...],  // Frame 2: 21 landmarks
            ...
        ]
    }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to numpy
    try:
        sequence = np.array(data.frames, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid landmark data: {e}")

    # Validate shape
    if sequence.ndim != 3 or sequence.shape[1] != 21 or sequence.shape[2] != 3:
        raise HTTPException(
            status_code=400,
            detail=f"Expected shape (seq_len, 21, 3), got {sequence.shape}"
        )

    # Preprocess
    feature_mode = config.get("feature_mode", DEFAULT_FEATURE_MODE)
    normalized = normalize_landmarks(sequence)
    features = extract_features(normalized, mode=feature_mode)

    # Predict
    with torch.no_grad():
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]

    # Get results
    classes = config.get("classes", [])
    confidence, pred_idx = probs.max(0)
    letter = classes[pred_idx.item()] if classes else str(pred_idx.item())

    # Format letter display
    display_letter = "Ñ" if letter == "nn" else letter.upper()

    # All probabilities (top 5)
    top_probs = {}
    top_k = min(5, len(classes))
    top_vals, top_idx = probs.topk(top_k)
    for val, idx in zip(top_vals, top_idx):
        cls = classes[idx.item()] if classes else str(idx.item())
        top_probs[cls.upper() if cls != "nn" else "Ñ"] = round(val.item(), 4)

    return PredictionResponse(
        letter=display_letter,
        confidence=round(confidence.item(), 4),
        all_probs=top_probs,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
