#!/usr/bin/env python3
"""
Capture word gestures for LSC.

Uses MediaPipe Holistic for upper body + both hands.
Variable-length recording (press SPACE to stop).

Usage:
    # Landmarks only (default)
    python scripts/03_capture_words.py --output data/raw_words

    # Video only
    python scripts/03_capture_words.py --mode video --output data/raw_words_video

    # Both video and landmarks
    python scripts/03_capture_words.py --mode both --output data/raw_words

Flow:
    1. Type word name and press ENTER
    2. Countdown 3-2-1
    3. Recording starts (red border)
    4. Press SPACE to stop recording
    5. Saved to {output}/{word}/

Landmarks captured (51 total):
    - Upper body pose: 9 points
    - Left hand: 21 points
    - Right hand: 21 points
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np

# Constants
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
VIDEO_FPS = 15

# Upper body pose landmark indices (from MediaPipe Pose 33 landmarks)
UPPER_BODY_INDICES = [
    0,   # nose
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
]

CaptureMode = Literal["landmarks", "video", "both"]


def get_next_filename(output_dir: Path, word: str, ext: str) -> Path:
    """Get next available filename for a word."""
    word_dir = output_dir / word.lower()
    word_dir.mkdir(parents=True, exist_ok=True)
    existing = list(word_dir.glob(f"*.{ext}"))
    count = len(existing) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return word_dir / f"{word}_{count:04d}_{timestamp}.{ext}"


def extract_pose_landmarks(pose_landmarks) -> np.ndarray:
    """Extract upper body pose landmarks (9 x 3)."""
    if pose_landmarks is None:
        return np.zeros((len(UPPER_BODY_INDICES), 3), dtype=np.float32)

    landmarks = []
    for idx in UPPER_BODY_INDICES:
        lm = pose_landmarks.landmark[idx]
        landmarks.append([lm.x, lm.y, lm.z])
    return np.array(landmarks, dtype=np.float32)


def extract_hand_landmarks(hand_landmarks) -> np.ndarray:
    """Extract hand landmarks (21 x 3)."""
    if hand_landmarks is None:
        return np.zeros((21, 3), dtype=np.float32)

    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )


def draw_status(
    frame: np.ndarray,
    mode: CaptureMode,
    state: str,
    word: str = "",
    input_text: str = "",
    countdown: int = 0,
    frame_count: int = 0,
    detection_status: dict = None,
) -> None:
    """Draw status overlay on frame."""
    h, w = frame.shape[:2]

    # Red border when recording
    if state == "recording":
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)

    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 85), (40, 40, 40), -1)

    if state == "input":
        cv2.putText(
            frame, f"Type word: {input_text}_",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            frame, "Press ENTER to start | ESC to quit",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1
        )
        cv2.putText(
            frame, f"Mode: {mode}",
            (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1
        )

    elif state == "countdown":
        cv2.putText(
            frame, f"Recording '{word}' in: {countdown}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        # Big countdown in center
        cv2.putText(
            frame, str(countdown),
            (w // 2 - 40, h // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10
        )

    elif state == "recording":
        cv2.putText(
            frame, f"RECORDING: '{word}' | Frames: {frame_count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        cv2.putText(
            frame, "Press SPACE to stop recording",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

    elif state == "idle":
        cv2.putText(
            frame, "Press any key to start typing word",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        if word:
            cv2.putText(
                frame, f"Last saved: {word}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        cv2.putText(
            frame, f"Mode: {mode}",
            (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1
        )

    # Detection status (bottom of frame)
    if detection_status:
        y_pos = h - 20
        parts = []
        parts.append(("Pose", detection_status.get("pose", False)))
        parts.append(("L-Hand", detection_status.get("left_hand", False)))
        parts.append(("R-Hand", detection_status.get("right_hand", False)))

        x_pos = 10
        for name, detected in parts:
            color = (0, 255, 0) if detected else (100, 100, 100)
            text = f"{name}: {'OK' if detected else '--'}"
            cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            x_pos += 120


def save_video(frames: list[np.ndarray], filepath: Path, fps: int = VIDEO_FPS) -> None:
    """Save frames as AVI video."""
    if not frames:
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(filepath), fourcc, fps, (w, h))

    for frame in frames:
        out.write(frame)

    out.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture word gestures")
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
        default="data/raw_words",
        help="Output directory",
    )
    args = parser.parse_args()

    mode: CaptureMode = args.mode
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print(f"LSC Word Capture - Mode: {mode}")
    print("=" * 50)
    print(f"Output: {output_dir}")
    print("Landmarks: Pose (9) + L-Hand (21) + R-Hand (21) = 51")
    print("1. Type word → ENTER")
    print("2. Countdown 3-2-1")
    print("3. Recording → SPACE to stop")
    print("ESC to quit")
    print("=" * 50)

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State machine
    state = "idle"
    input_text = ""
    current_word = ""
    last_saved_word = ""
    countdown_start = 0
    recorded_landmarks: list[np.ndarray] = []
    recorded_frames: list[np.ndarray] = []

    print("\nCamera ready! Press any key to start.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Process with Holistic
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Detection status
        detection_status = {
            "pose": results.pose_landmarks is not None,
            "left_hand": results.left_hand_landmarks is not None,
            "right_hand": results.right_hand_landmarks is not None,
        }

        # Draw landmarks on preview
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        # State: Recording
        if state == "recording":
            # Extract and store landmarks
            pose = extract_pose_landmarks(results.pose_landmarks)
            left = extract_hand_landmarks(results.left_hand_landmarks)
            right = extract_hand_landmarks(results.right_hand_landmarks)

            # Combine: (51, 3)
            frame_landmarks = np.concatenate([pose, left, right], axis=0)
            recorded_landmarks.append(frame_landmarks)

            # Store video frame if needed
            if mode in ("video", "both"):
                recorded_frames.append(frame.copy())

        # State: Countdown
        elif state == "countdown":
            elapsed = time.time() - countdown_start
            remaining = 3 - int(elapsed)
            if remaining <= 0:
                state = "recording"
                recorded_landmarks = []
                recorded_frames = []
                print(f"Recording '{current_word}'... Press SPACE to stop.")

        # Draw status
        countdown_val = max(0, 3 - int(time.time() - countdown_start)) if state == "countdown" else 0
        draw_status(
            frame, mode, state,
            word=current_word or last_saved_word,
            input_text=input_text,
            countdown=countdown_val,
            frame_count=len(recorded_landmarks),
            detection_status=detection_status,
        )

        cv2.imshow("LSC Word Capture", frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("\nExiting...")
            break

        if state == "idle":
            if key != 255:
                state = "input"
                input_text = ""
                if 32 <= key <= 126:
                    input_text = chr(key)

        elif state == "input":
            if key == 13:  # ENTER
                if input_text.strip():
                    current_word = input_text.strip().lower()
                    state = "countdown"
                    countdown_start = time.time()
                    print(f"Starting countdown for '{current_word}'...")
            elif key == 8:  # Backspace
                input_text = input_text[:-1]
            elif key == 27:  # ESC - cancel
                state = "idle"
                input_text = ""
            elif 32 <= key <= 126:
                input_text += chr(key)

        elif state == "recording":
            if key == 32:  # SPACE to stop
                if len(recorded_landmarks) > 0:
                    # Save landmarks
                    if mode in ("landmarks", "both"):
                        landmarks_path = get_next_filename(output_dir, current_word, "npy")
                        arr = np.stack(recorded_landmarks, axis=0)
                        np.save(landmarks_path, arr)
                        print(f"Saved: {landmarks_path.name} shape={arr.shape}")

                    # Save video
                    if mode in ("video", "both"):
                        video_path = get_next_filename(output_dir, current_word, "avi")
                        save_video(recorded_frames, video_path)
                        print(f"Saved: {video_path.name} frames={len(recorded_frames)}")

                    last_saved_word = current_word
                else:
                    print("No frames recorded!")

                state = "idle"
                recorded_landmarks = []
                recorded_frames = []
                current_word = ""

    holistic.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSaved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
