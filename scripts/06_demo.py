#!/usr/bin/env python3
"""
Real-time gesture recognition demo.

Usage:
    python scripts/06_demo.py --checkpoint models/checkpoints/<run_id>/best.pth

Controls:
    - ESC or Q: Quit
"""

import argparse
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

from src.data.preprocessing import (
    normalize_landmarks,
    extract_features,
    DEFAULT_FEATURE_MODE,
    FEATURE_MODES,
)
from src.models import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get("config", {})

    model = get_model(
        model_type=config.get("model", "bigru"),
        num_classes=config["num_classes"],
        input_dim=config["input_dim"],
        hidden_dim=config.get("hidden_dim", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, config


def extract_mp_landmarks(hand_landmarks) -> np.ndarray:
    """Extract landmarks from MediaPipe result."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )


@torch.no_grad()
def predict_sequence(
    model: torch.nn.Module,
    buffer: deque,
    classes: list[str],
    feature_mode: str,
) -> tuple[str, float]:
    """Run prediction on buffered sequence."""
    sequence = np.stack(list(buffer), axis=0)
    normalized = normalize_landmarks(sequence)
    features = extract_features(normalized, mode=feature_mode)

    tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
    outputs = model(tensor)
    probs = F.softmax(outputs, dim=1)
    conf, pred_idx = probs.max(1)

    return classes[pred_idx.item()], conf.item()


def main():
    parser = argparse.ArgumentParser(description="Real-time LSC demo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--predict-every", type=int, default=5)
    args = parser.parse_args()

    print("=" * 50)
    print("LSC Real-time Demo")
    print("=" * 50)

    # Load model
    model, config = load_model(Path(args.checkpoint))
    classes = config.get("classes", [])
    feature_mode = config.get("feature_mode", DEFAULT_FEATURE_MODE)
    model_type = config.get("model", "bigru")

    print(f"Model: {model_type}")
    print(f"Classes: {classes}")
    print(f"Features: {feature_mode}")
    print(f"Device: {DEVICE}")

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State
    buffer: deque = deque(maxlen=args.sequence_length)
    frame_count = 0
    current_pred = None
    current_conf = 0.0

    print("\nCamera ready! Show hand gestures.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Buffer landmarks
                landmarks = extract_mp_landmarks(hand_landmarks)
                buffer.append(landmarks)

                # Predict when buffer is full
                frame_count += 1
                if len(buffer) >= args.sequence_length and frame_count >= args.predict_every:
                    frame_count = 0
                    current_pred, current_conf = predict_sequence(
                        model, buffer, classes, feature_mode
                    )
        else:
            buffer.clear()
            current_pred = None
            current_conf = 0.0

        # Draw UI
        # Status bar
        cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)

        status = "Hand: OK" if hand_detected else "Hand: --"
        color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Buffer progress
        buf_text = f"Buffer: {len(buffer)}/{args.sequence_length}"
        cv2.putText(frame, buf_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Prediction
        if current_pred:
            display = "Ñ" if current_pred == "nn" else current_pred.upper()

            # Large letter
            font_scale = 4
            (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 8)
            x = w - tw - 30
            y = 70

            pred_color = (0, 255, 0) if current_conf > 0.5 else (0, 255, 255)
            cv2.rectangle(frame, (x - 10, 10), (w - 10, y + 10), pred_color, -1)
            cv2.putText(frame, display, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 8)
            cv2.putText(frame, f"{current_conf:.0%}", (x, y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("LSC Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo ended.")


if __name__ == "__main__":
    main()
