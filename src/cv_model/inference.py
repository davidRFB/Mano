"""
Real-time inference script for LSC gesture recognition.

Usage:
    python -m src.cv_model.inference
    python -m src.cv_model.inference --model models/mobilenet_v2_v1_acc1.00_20251127_204309.pth

Controls:
    - Press ESC or 'q' to quit
    - Hand gesture predictions displayed in real-time
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.cv_model.train import get_model


# Configuration
WINDOW_NAME = "LSC Gesture Recognition - Press ESC to quit"
IMAGE_SIZE = 224
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
CROP_PADDING = 50

# ImageNet normalization (must match training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path) -> tuple[torch.nn.Module, list[str]]:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file

    Returns:
        Tuple of (model, class_names)
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    model_name = checkpoint["model_name"]
    classes = checkpoint["classes"]
    num_classes = checkpoint["num_classes"]

    model = get_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded {model_name} with {num_classes} classes")
    print(f"Classes: {classes}")

    return model, classes


def get_inference_transform() -> transforms.Compose:
    """Get transforms for inference (same as validation)."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_hand_bbox(
    hand_landmarks, frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    """Extract bounding box from hand landmarks with padding."""
    x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
    y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]

    x_min = int(max(0, min(x_coords) - CROP_PADDING))
    x_max = int(min(frame_width, max(x_coords) + CROP_PADDING))
    y_min = int(max(0, min(y_coords) - CROP_PADDING))
    y_max = int(min(frame_height, max(y_coords) + CROP_PADDING))

    return x_min, y_min, x_max, y_max


def crop_hand(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop hand region from frame."""
    x_min, y_min, x_max, y_max = bbox
    return frame[y_min:y_max, x_min:x_max].copy()


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    transform: transforms.Compose,
    classes: list[str],
) -> tuple[str, float]:
    """
    Run inference on a cropped hand image.

    Args:
        model: Trained PyTorch model
        image: BGR image (cropped hand region)
        transform: Preprocessing transforms
        classes: List of class names

    Returns:
        Tuple of (predicted_letter, confidence)
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transforms and add batch dimension
    tensor = transform(rgb_image).unsqueeze(0).to(DEVICE)

    # Forward pass
    outputs = model(tensor)
    probabilities = F.softmax(outputs, dim=1)

    # Get prediction
    confidence, predicted_idx = probabilities.max(1)
    predicted_letter = classes[predicted_idx.item()]

    return predicted_letter, confidence.item()


def draw_prediction(
    frame: np.ndarray,
    letter: str,
    confidence: float,
    bbox: tuple[int, int, int, int],
) -> None:
    """Draw prediction overlay on frame."""
    x_min, y_min, x_max, y_max = bbox

    # Draw bounding box
    color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    # Draw letter above bbox
    label = f"{letter.upper()} ({confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3

    # Get text size for background
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x_min, y_min - text_h - 15),
        (x_min + text_w + 10, y_min),
        color,
        -1,
    )

    # Draw text
    cv2.putText(
        frame,
        label,
        (x_min + 5, y_min - 10),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )


def draw_overlay(
    frame: np.ndarray,
    hand_detected: bool,
    prediction: str | None,
    confidence: float | None,
) -> None:
    """Draw status overlay on frame."""
    h, w = frame.shape[:2]

    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)

    # Title
    cv2.putText(
        frame,
        "LSC Gesture Recognition",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Instructions
    cv2.putText(
        frame,
        "Press ESC to quit",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )

    # Hand detection status
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    status_text = "Hand: DETECTED" if hand_detected else "Hand: NOT FOUND"
    cv2.putText(
        frame,
        status_text,
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        status_color,
        1,
    )

    # Current prediction (large display)
    if prediction and confidence:
        # Large letter display on right side
        display_text = prediction.upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4
        thickness = 8

        (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)

        x_pos = w - text_w - 30
        y_pos = 70

        # Background
        cv2.rectangle(
            frame,
            (x_pos - 10, 10),
            (w - 10, y_pos + 10),
            (0, 255, 0) if confidence > 0.8 else (0, 255, 255),
            -1,
        )

        # Letter
        cv2.putText(
            frame,
            display_text,
            (x_pos, y_pos),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )


def find_latest_model(models_dir: Path = Path("models")) -> Path | None:
    """Find the most recent model checkpoint."""
    checkpoints = list(models_dir.glob("*.pth"))
    if not checkpoints:
        return None

    # Sort by modification time, newest first
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def main(model_path: str | None = None) -> None:
    """Main inference loop."""
    print("=" * 60)
    print("LSC Gesture Recognition - Real-time Inference")
    print("=" * 60)

    # Find model
    if model_path:
        checkpoint_path = Path(model_path)
    else:
        checkpoint_path = find_latest_model()
        if checkpoint_path is None:
            print("Error: No model found in models/ directory")
            print("Train a model first with: python -m src.cv_model.train")
            return

    print(f"Using model: {checkpoint_path}")
    print(f"Device: {DEVICE}")

    # Load model
    model, classes = load_model(checkpoint_path)
    transform = get_inference_transform()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nCamera ready! Show your hand to see predictions.")
    print("=" * 60)

    current_prediction = None
    current_confidence = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hand_detected = False
        current_bbox = None

        # Process hand landmarks
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Get bounding box
                current_bbox = get_hand_bbox(hand_landmarks, w, h)

                # Crop hand and run inference
                hand_crop = crop_hand(frame, current_bbox)

                # Ensure crop is valid
                if hand_crop.size > 0 and hand_crop.shape[0] > 10 and hand_crop.shape[1] > 10:
                    current_prediction, current_confidence = predict(
                        model, hand_crop, transform, classes
                    )

                    # Draw prediction on frame
                    draw_prediction(frame, current_prediction, current_confidence, current_bbox)
        else:
            current_prediction = None
            current_confidence = None

        # Draw overlay
        draw_overlay(frame, hand_detected, current_prediction, current_confidence)

        # Show frame
        cv2.imshow(WINDOW_NAME, frame)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            print("\nExiting...")
            break

    # Cleanup
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    print("Inference session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time LSC gesture recognition")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (default: latest in models/)",
    )

    args = parser.parse_args()
    main(model_path=args.model)

