#!/usr/bin/env python3
"""
Capture gesture sequences for LSC (all letters).

Records multi-frame sequences for training sequence models.
Works for both static and dynamic letters - uniform approach.

Usage:
    # Landmarks only (default)
    python scripts/02_capture_dynamic.py --output data/raw_landmarks

    # Video only
    python scripts/02_capture_dynamic.py --mode video --output data/raw_video

    # Both video and landmarks
    python scripts/02_capture_dynamic.py --mode both --output data/raw_sequences

Controls:
    - Press A-Z to record that letter (N-frame sequence)
    - Press DOWN ARROW for Ñ (saved as 'nn')
    - Press ESC to quit

Output structure:
    {output_dir}/{letter}/{letter}_{count}_{timestamp}.{npy|avi}
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
DEFAULT_FRAMES = 20
VIDEO_FPS = 15

CaptureMode = Literal["landmarks", "video", "both"]


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
    recording: bool,
    record_progress: int,
    total_frames: int,
    last_letter: str,
    last_count: int,
) -> None:
    """Draw UI overlay on frame."""
    h, w = frame.shape[:2]

    # Red border when recording
    if recording:
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 8)

    # Status bar
    cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)

    if recording:
        # Recording status
        progress_text = f"RECORDING: {record_progress}/{total_frames} frames"
        cv2.putText(frame, progress_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Progress bar
        bar_width = int((w - 20) * record_progress / total_frames)
        cv2.rectangle(frame, (10, 55), (w - 10, 70), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 55), (10 + bar_width, 70), (0, 0, 255), -1)
    else:
        # Mode indicator
        mode_text = f"Mode: {mode} | Frames: {total_frames}"
        cv2.putText(frame, mode_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)

        # Hand status
        status = "Hand: OK" if hand_detected else "Hand: --"
        color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, status, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # Last capture
        if last_letter:
            display = "Ñ" if last_letter == "nn" else last_letter.upper()
            cv2.putText(
                frame, f"Last: {display} ({last_count})", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        # Instructions
        cv2.putText(
            frame, "A-Z: record | DOWN: Ñ | ESC: quit",
            (w - 270, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
        )


def save_video(frames: list[np.ndarray], filepath: Path, fps: int = VIDEO_FPS) -> None:
    """Save frames as AVI video."""
    if not frames:
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(filepath), fourcc, fps, (w, h))

    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        out.write(frame)

    out.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture gesture sequences")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["landmarks", "video", "both"],
        default="landmarks",
        help="Capture mode: landmarks (.npy), video (.avi), or both",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw_landmarks",
        help="Output directory",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"Frames per sequence (default: {DEFAULT_FRAMES})",
    )
    args = parser.parse_args()

    mode: CaptureMode = args.mode
    output_dir = Path(args.output)
    num_frames = args.frames

    print("=" * 50)
    print(f"LSC Sequence Capture - Mode: {mode}")
    print("=" * 50)
    print(f"Output: {output_dir}")
    print(f"Frames per sequence: {num_frames}")
    if mode == "landmarks":
        print("Format: .npy (21, 3) per frame")
    elif mode == "video":
        print("Format: .avi (cropped hand)")
    else:
        print("Format: .npy + .avi")
    print("Press A-Z to record | DOWN for Ñ | ESC to quit")
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
    recording = False
    record_letter = ""
    record_landmarks: list[np.ndarray] = []
    record_frames: list[np.ndarray] = []
    record_bbox: tuple[int, int, int, int] | None = None

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
        current_landmarks = None
        current_bbox = None

        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            current_landmarks = extract_landmarks(hand_landmarks)
            current_bbox = get_hand_bbox(hand_landmarks, w, h)

            # Draw landmarks for preview
            draw_landmarks(frame, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles)

            # Draw bbox
            x_min, y_min, x_max, y_max = current_bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

        # Recording logic
        if recording:
            if not hand_detected:
                print("[!] Hand lost - aborting recording")
                recording = False
                record_landmarks = []
                record_frames = []
                record_bbox = None
            else:
                # Store landmarks
                record_landmarks.append(current_landmarks)

                # Store video frame (cropped) if needed
                if mode in ("video", "both"):
                    x_min, y_min, x_max, y_max = record_bbox
                    cropped = frame[y_min:y_max, x_min:x_max].copy()
                    record_frames.append(cropped)

                # Check if recording complete
                if len(record_landmarks) >= num_frames:
                    display = "Ñ" if record_letter == "nn" else record_letter.upper()

                    # Save landmarks
                    if mode in ("landmarks", "both"):
                        landmarks_path = get_next_filename(output_dir, record_letter, "npy")
                        arr = np.stack(record_landmarks, axis=0)
                        np.save(landmarks_path, arr)
                        print(f"Saved: {landmarks_path.name} shape={arr.shape}")

                    # Save video
                    if mode in ("video", "both"):
                        video_path = get_next_filename(output_dir, record_letter, "avi")
                        save_video(record_frames, video_path)
                        crop_h, crop_w = record_frames[0].shape[:2]
                        print(f"Saved: {video_path.name} size={crop_w}x{crop_h}")

                    last_letter = record_letter
                    # Count based on primary output
                    ext = "npy" if mode in ("landmarks", "both") else "avi"
                    last_count = len(list((output_dir / record_letter).glob(f"*.{ext}")))

                    recording = False
                    record_landmarks = []
                    record_frames = []
                    record_bbox = None

        # Draw UI
        draw_ui(
            frame, mode, hand_detected, recording,
            len(record_landmarks), num_frames,
            last_letter, last_count
        )

        cv2.imshow("LSC Sequence Capture", frame)

        # Handle keys
        key = cv2.waitKeyEx(1)

        if key == 27:  # ESC
            break

        # Detect letter key (only when not recording)
        if not recording:
            letter = None
            if ord("a") <= key <= ord("z"):
                letter = chr(key)
            elif key in (65364, 2621440, 84):  # Down arrow
                letter = "nn"

            if letter:
                if not hand_detected:
                    display = "Ñ" if letter == "nn" else letter.upper()
                    print(f"[!] No hand detected - cannot record '{display}'")
                else:
                    recording = True
                    record_letter = letter
                    record_landmarks = []
                    record_frames = []
                    record_bbox = current_bbox
                    display = "Ñ" if letter == "nn" else letter.upper()
                    print(f"Recording '{display}'...")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSaved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
