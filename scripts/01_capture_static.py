#!/usr/bin/env python3
"""
Unified static gesture capture for LSC.

Usage:
    # Photo only (cropped hand)
    python scripts/01_capture_static.py --mode photo --output data/raw

    # Photo with landmarks overlay
    python scripts/01_capture_static.py --mode photo_landmarks --output data/raw_with_landmarks

    # Landmarks only (.npy)
    python scripts/01_capture_static.py --mode landmarks --output data/raw_landmarks_static

Controls:
    - Press A-Z to capture that letter
    - Press DOWN ARROW for Ñ (saved as 'nn')
    - Press ESC to quit

Output structure:
    {output_dir}/{letter}/{letter}_{count}_{timestamp}.{png|npy}
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np

# Constants
LETTERS = list("abcdefghijklmnopqrstuvwxyz") + ["nn"]
CROP_PADDING = 50
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

CaptureMode = Literal["photo", "photo_landmarks", "landmarks"]


def setup_directories(output_dir: Path) -> None:
    """Create letter subdirectories."""
    for letter in LETTERS:
        (output_dir / letter).mkdir(parents=True, exist_ok=True)


def get_next_filename(output_dir: Path, letter: str, ext: str) -> Path:
    """Get next available filename for a letter."""
    letter_dir = output_dir / letter
    existing = list(letter_dir.glob(f"*.{ext}"))
    count = len(existing) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return letter_dir / f"{letter}_{count:04d}_{timestamp}.{ext}"


def extract_landmarks(hand_landmarks) -> np.ndarray:
    """Extract 21 landmarks as (21, 3) array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )


def get_hand_bbox(
    hand_landmarks, width: int, height: int
) -> tuple[int, int, int, int]:
    """Get bounding box around hand with padding."""
    x_coords = [lm.x * width for lm in hand_landmarks.landmark]
    y_coords = [lm.y * height for lm in hand_landmarks.landmark]

    x_min = int(max(0, min(x_coords) - CROP_PADDING))
    x_max = int(min(width, max(x_coords) + CROP_PADDING))
    y_min = int(max(0, min(y_coords) - CROP_PADDING))
    y_max = int(min(height, max(y_coords) + CROP_PADDING))

    return x_min, y_min, x_max, y_max


def draw_landmarks(
    frame: np.ndarray,
    hand_landmarks,
    mp_hands,
    mp_drawing,
    mp_drawing_styles,
) -> None:
    """Draw hand landmarks on frame."""
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )


def draw_ui(
    frame: np.ndarray,
    mode: CaptureMode,
    hand_detected: bool,
    last_letter: str,
    last_count: int,
) -> None:
    """Draw UI overlay on frame."""
    h, w = frame.shape[:2]

    # Status bar
    cv2.rectangle(frame, (0, 0), (w, 70), (40, 40, 40), -1)

    # Mode indicator
    mode_text = f"Mode: {mode}"
    cv2.putText(frame, mode_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

    # Hand status
    status = "Hand: OK" if hand_detected else "Hand: --"
    color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(frame, status, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Last capture
    if last_letter:
        display = "Ñ" if last_letter == "nn" else last_letter.upper()
        cv2.putText(
            frame, f"Last: {display} ({last_count})", (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    # Instructions
    cv2.putText(
        frame, "A-Z: capture | DOWN: Ñ | ESC: quit",
        (w - 280, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
    )


def capture_frame(
    cap: cv2.VideoCapture,
    hands,
    mode: CaptureMode,
    mp_hands,
    mp_drawing,
    mp_drawing_styles,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple | None]:
    """
    Capture and process a frame.

    Returns: (cropped_image, landmarks, bbox) or (None, None, None) if no hand.
    """
    ret, frame = cap.read()
    if not ret:
        return None, None, None

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None, None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = extract_landmarks(hand_landmarks)
    bbox = get_hand_bbox(hand_landmarks, w, h)

    # Prepare output based on mode
    if mode == "photo":
        # Crop raw frame
        x_min, y_min, x_max, y_max = bbox
        cropped = frame[y_min:y_max, x_min:x_max].copy()
        return cropped, landmarks, bbox

    elif mode == "photo_landmarks":
        # Draw landmarks then crop
        frame_with_landmarks = frame.copy()
        draw_landmarks(frame_with_landmarks, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles)
        x_min, y_min, x_max, y_max = bbox
        cropped = frame_with_landmarks[y_min:y_max, x_min:x_max].copy()
        return cropped, landmarks, bbox

    else:  # landmarks
        return None, landmarks, bbox


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture static gestures")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["photo", "photo_landmarks", "landmarks"],
        default="landmarks",
        help="Capture mode: photo (raw), photo_landmarks (with overlay), landmarks (npy only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw_landmarks_static",
        help="Output directory",
    )
    args = parser.parse_args()

    mode: CaptureMode = args.mode
    output_dir = Path(args.output)
    ext = "npy" if mode == "landmarks" else "png"

    print("=" * 50)
    print(f"LSC Static Capture - Mode: {mode}")
    print("=" * 50)
    print(f"Output: {output_dir}")
    print(f"Format: .{ext}")
    print("Press A-Z to capture | DOWN for Ñ | ESC to quit")
    print("=" * 50)

    setup_directories(output_dir)

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State
    last_letter = ""
    last_count = 0

    print("\nCamera ready!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Process for display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = False
        current_bbox = None

        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            current_bbox = get_hand_bbox(hand_landmarks, w, h)

            # Draw landmarks for preview
            draw_landmarks(frame, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles)

            # Draw bbox
            x_min, y_min, x_max, y_max = current_bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

        # Draw UI
        draw_ui(frame, mode, hand_detected, last_letter, last_count)

        cv2.imshow("LSC Capture", frame)

        # Handle keys
        key = cv2.waitKeyEx(1)

        if key == 27:  # ESC
            break

        # Detect letter key
        letter = None
        if ord("a") <= key <= ord("z"):
            letter = chr(key)
        elif key in (65364, 2621440, 84):  # Down arrow
            letter = "nn"

        if letter:
            if not hand_detected:
                display = "Ñ" if letter == "nn" else letter.upper()
                print(f"[!] No hand detected - cannot capture '{display}'")
                continue

            # Capture clean frame
            cropped, landmarks, _ = capture_frame(
                cap, hands, mode, mp_hands, mp_drawing, mp_drawing_styles
            )

            if landmarks is None:
                print("[!] Hand lost during capture")
                continue

            filepath = get_next_filename(output_dir, letter, ext)

            if mode == "landmarks":
                np.save(filepath, landmarks)
            else:
                cv2.imwrite(str(filepath), cropped)

            last_letter = letter
            last_count = len(list((output_dir / letter).glob(f"*.{ext}")))

            display = "Ñ" if letter == "nn" else letter.upper()
            if mode == "landmarks":
                print(f"Saved: {filepath.name} shape={landmarks.shape} | Total {display}: {last_count}")
            else:
                print(f"Saved: {filepath.name} size={cropped.shape[1]}x{cropped.shape[0]} | Total {display}: {last_count}")

            # Visual feedback
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 255, 0), 8)
            cv2.imshow("LSC Capture", frame)
            cv2.waitKey(100)

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSaved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
