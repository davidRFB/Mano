"""
Word capture script for Colombian Sign Language.

Uses MediaPipe Holistic for upper body + hands landmarks.

Usage:
    python scripts/capture_words.py

Flow:
    1. Type word name and press ENTER
    2. Countdown 3-2-1
    3. Recording starts (red border)
    4. Press SPACE to stop recording
    5. Landmarks saved to data/raw_words/{word}/

Landmarks captured:
    - Upper body pose: 13 points (shoulders, elbows, wrists, head, torso)
    - Left hand: 21 points
    - Right hand: 21 points
    - Total: 55 landmarks x 3 coords = 165 features per frame
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# Configuration
DATA_DIR = Path("./data/raw_words")
DATA_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_NAME = "LSC Word Capture"
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Upper body pose landmark indices (from MediaPipe Pose 33 landmarks)
# We only need upper body for sign language
UPPER_BODY_INDICES = [
    0,   # nose
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip (for torso reference)
    24,  # right hip (for torso reference)
]

# For reference - all we capture:
# Pose: 9 upper body landmarks
# Left hand: 21 landmarks (if detected)
# Right hand: 21 landmarks (if detected)
# Total max: 9 + 21 + 21 = 51 landmarks


def get_next_filename(word: str) -> Path:
    """Get next available filename for a word."""
    word_dir = DATA_DIR / word.lower()
    word_dir.mkdir(parents=True, exist_ok=True)
    existing = list(word_dir.glob("*.npy"))
    next_num = len(existing) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return word_dir / f"{word}_{next_num:04d}_{timestamp}.npy"


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

    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )
    return landmarks


def draw_status(
    frame: np.ndarray,
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
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)

    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)

    if state == "input":
        cv2.putText(
            frame,
            f"Type word: {input_text}_",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Press ENTER to start | ESC to quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )

    elif state == "countdown":
        cv2.putText(
            frame,
            f"Recording '{word}' in: {countdown}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        # Big countdown in center
        cv2.putText(
            frame,
            str(countdown),
            (w // 2 - 40, h // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (0, 255, 255),
            10,
        )

    elif state == "recording":
        cv2.putText(
            frame,
            f"RECORDING: '{word}' | Frames: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "Press SPACE to stop recording",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    elif state == "idle":
        cv2.putText(
            frame,
            "Press any key to start typing word",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        if word:
            cv2.putText(
                frame,
                f"Last saved: {word}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    # Detection status (bottom of frame)
    if detection_status:
        y_pos = h - 20
        status_parts = []
        if detection_status.get("pose"):
            status_parts.append(("Pose: OK", (0, 255, 0)))
        else:
            status_parts.append(("Pose: --", (100, 100, 100)))
        if detection_status.get("left_hand"):
            status_parts.append(("L-Hand: OK", (0, 255, 0)))
        else:
            status_parts.append(("L-Hand: --", (100, 100, 100)))
        if detection_status.get("right_hand"):
            status_parts.append(("R-Hand: OK", (0, 255, 0)))
        else:
            status_parts.append(("R-Hand: --", (100, 100, 100)))

        x_pos = 10
        for text, color in status_parts:
            cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            x_pos += 120


def main() -> None:
    """Main capture loop."""
    print("=" * 50)
    print("LSC Word Capture Tool (Holistic)")
    print("=" * 50)
    print("1. Type word name → ENTER")
    print("2. Countdown 3-2-1")
    print("3. Recording starts → SPACE to stop")
    print("ESC to quit anytime")
    print("=" * 50)
    print(f"\nLandmarks: Upper body (9) + Left hand (21) + Right hand (21) = 51 points")
    print(f"Saving to: {DATA_DIR.absolute()}")

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

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State machine
    state = "idle"  # idle, input, countdown, recording
    input_text = ""
    current_word = ""
    last_saved_word = ""
    countdown_start = 0
    recorded_frames: list[np.ndarray] = []

    print("\nCamera ready! Press any key to start.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Process with Holistic
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Detection status
        detection_status = {
            "pose": results.pose_landmarks is not None,
            "left_hand": results.left_hand_landmarks is not None,
            "right_hand": results.right_hand_landmarks is not None,
        }

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        # State: Recording
        if state == "recording":
            # Extract and store landmarks
            pose = extract_pose_landmarks(results.pose_landmarks)
            left_hand = extract_hand_landmarks(results.left_hand_landmarks)
            right_hand = extract_hand_landmarks(results.right_hand_landmarks)

            # Combine: (9 + 21 + 21, 3) = (51, 3)
            frame_landmarks = np.concatenate([pose, left_hand, right_hand], axis=0)
            recorded_frames.append(frame_landmarks)

        # State: Countdown
        elif state == "countdown":
            elapsed = time.time() - countdown_start
            remaining = 3 - int(elapsed)
            if remaining <= 0:
                state = "recording"
                recorded_frames = []
                print(f"Recording '{current_word}'... Press SPACE to stop.")

        # Draw status
        draw_status(
            frame,
            state=state,
            word=current_word or last_saved_word,
            input_text=input_text,
            countdown=max(0, 3 - int(time.time() - countdown_start)) if state == "countdown" else 0,
            frame_count=len(recorded_frames),
            detection_status=detection_status,
        )

        cv2.imshow(WINDOW_NAME, frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF

        # ESC to quit
        if key == 27:
            print("\nExiting...")
            break

        # State-specific key handling
        if state == "idle":
            if key != 255:  # Any key pressed
                state = "input"
                input_text = ""
                if 32 <= key <= 126:  # Printable ASCII
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
            elif key == 27:  # ESC - cancel input
                state = "idle"
                input_text = ""
            elif 32 <= key <= 126:  # Printable ASCII
                input_text += chr(key)

        elif state == "recording":
            if key == 32:  # SPACE to stop
                if len(recorded_frames) > 0:
                    # Save recording
                    filepath = get_next_filename(current_word)
                    landmarks_array = np.stack(recorded_frames, axis=0)
                    np.save(filepath, landmarks_array)

                    print(f"Saved: {filepath.name} | Shape: {landmarks_array.shape} | Frames: {len(recorded_frames)}")
                    last_saved_word = current_word
                else:
                    print("No frames recorded!")

                state = "idle"
                recorded_frames = []
                current_word = ""

    # Cleanup
    holistic.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\nCapture session ended.")
    print(f"Words saved to: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
