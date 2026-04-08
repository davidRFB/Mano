"""
LSC Gesture Recognition API

FastAPI service for landmark-based gesture recognition.

Usage:
    # Local development
    uvicorn api.main:app --reload --port 8000

    # With custom model path
    MODEL_PATH=path/to/best.pth uvicorn api.main:app --port 8000

Endpoints:
    GET  /              - API info
    GET  /health        - Health check
    POST /predict       - Predict from landmarks JSON
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.preprocessing import (
    normalize_landmarks,
    extract_features,
    DEFAULT_FEATURE_MODE,
)
from src.models import get_model

# Configuration
MODEL_PATH = Path(os.getenv("MODEL_PATH", "api/model.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global state
model = None
config: dict = {}

app = FastAPI(
    title="LSC Gesture Recognition API",
    description="Colombian Sign Language letter recognition from hand landmarks",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://davidrfb.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class LandmarkRequest(BaseModel):
    """Landmarks input: single frame or sequence.

    - Single frame: "landmarks" with shape (21, 3)
    - Sequence: "frames" with shape (seq_len, 21, 3)
    Send one or the other.
    """
    landmarks: list[list[float]] | None = None  # (21, 3) single frame
    frames: list[list[list[float]]] | None = None  # (seq_len, 21, 3) sequence


class PredictionResponse(BaseModel):
    """Prediction result."""
    letter: str
    confidence: float
    all_probs: Optional[dict[str, float]] = None


def load_model(path: Path) -> tuple[torch.nn.Module, dict]:
    """Load model directly from checkpoint file."""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]

    mdl = get_model(
        model_type=cfg["model"],
        num_classes=cfg["num_classes"],
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
    )
    mdl.load_state_dict(checkpoint["model_state_dict"])
    mdl.to(DEVICE)
    mdl.eval()

    return mdl, cfg


@app.on_event("startup")
def startup() -> None:
    """Load model at startup."""
    global model, config
    print("=" * 50)
    print(f"Loading model from {MODEL_PATH}...")

    if not MODEL_PATH.exists():
        print(f"Warning: {MODEL_PATH} not found. API running without model.")
        print("=" * 50)
        return

    try:
        model, config = load_model(MODEL_PATH)
        print(f"Model: {config.get('model')} | Classes: {config.get('num_classes')}")
        print(f"Features: {config.get('feature_mode')}")
        print(f"Val acc: {config.get('best_val_acc', 'N/A')}")
        print("API ready!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API running without model.")
    print("=" * 50)


@app.get("/")
def root() -> dict:
    """API info."""
    return {
        "name": "LSC Gesture Recognition API",
        "version": "2.1.0",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: LandmarkRequest) -> PredictionResponse:
    """
    Predict gesture from landmarks.

    Send ONE of:
    - Single frame: {"landmarks": [[x,y,z], ...]}  (21 landmarks)
    - Sequence:     {"frames": [[[x,y,z], ...], ...]}  (N frames x 21 landmarks)

    Static models use single frame. Sequence models need 20 frames.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Parse input: single frame or sequence
    try:
        if data.landmarks is not None:
            sequence = np.array(data.landmarks, dtype=np.float32)
            if sequence.shape != (21, 3):
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected landmarks shape (21, 3), got {sequence.shape}",
                )
            sequence = sequence[np.newaxis, :, :]
        elif data.frames is not None:
            sequence = np.array(data.frames, dtype=np.float32)
            if sequence.ndim != 3 or sequence.shape[1] != 21 or sequence.shape[2] != 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected frames shape (N, 21, 3), got {sequence.shape}",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Send 'landmarks' (single frame) or 'frames' (sequence)",
            )
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid landmark data: {e}")

    # Preprocess
    feature_mode = config.get("feature_mode", DEFAULT_FEATURE_MODE)
    model_type = config.get("model", "bigru")
    normalized = normalize_landmarks(sequence)
    features = extract_features(normalized, mode=feature_mode)

    # Static models expect a single frame
    if model_type == "static":
        features = features[0]

    # Predict
    with torch.no_grad():
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]

    # Get results
    classes = config.get("classes", [])
    confidence, pred_idx = probs.max(0)
    letter = classes[pred_idx.item()] if classes else str(pred_idx.item())

    display_letter = "NN" if letter == "nn" else letter.upper()

    # Top 5 probabilities
    top_probs = {}
    top_k = min(5, len(classes))
    top_vals, top_idx = probs.topk(top_k)
    for val, idx in zip(top_vals, top_idx):
        cls = classes[idx.item()] if classes else str(idx.item())
        top_probs[cls.upper() if cls != "nn" else "NN"] = round(val.item(), 4)

    return PredictionResponse(
        letter=display_letter,
        confidence=round(confidence.item(), 4),
        all_probs=top_probs,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
