"""
Landmark capture script for Colombian Sign Language gestures.

Usage:
    python scripts/capture_landmarks.py

Controls:
    - Press any letter key (a-z) to start recording landmarks for that letter
    - Press DOWN ARROW to record Ñ (saved as nn/)
    - Press ESC to quit
    - Landmarks are saved to data/raw_landmarks/{letter}/

Features:
    - Hand detection with MediaPipe
    - Captures 21 landmark points (x, y, z) per frame
    - Records 20-frame sequences per gesture
    - Saves as .npy files (shape: 20 x 21 x 3)
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime
import os

# Configuration
DATA_DIR = Path("../data/raw_landmarks")
os.makedirs(DATA_DIR, exist_ok=True)
WINDOW_NAME = "LSC Landmark Capture - Press letter keys to record"
CROP_PADDING = 50  # pixels of padding around hand for display
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Recording settings
RECORD_FRAMES = 20  # number of frames per recording


def ensure_letter_dirs() -> None:
    """Create directories for each letter a-z plus nn for ñ."""
    letters = list("abcdefghijklmnopqrstuvwxyz") + ["nn"]
    for letter in letters:
        letter_dir = DATA_DIR / letter
        letter_dir.mkdir(parents=True, exist_ok=True)


def get_next_filename(letter: str) -> Path:
    """Get next available filename for a letter."""
    letter_dir = DATA_DIR / letter
    existing = list(letter_dir.glob("*.npy"))
    next_num = len(existing) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return letter_dir / f"{letter}_{next_num:04d}_{timestamp}.npy"


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


def extract_landmarks(hand_landmarks) -> np.ndarray:
    """Extract 21 landmarks as numpy array (21 x 3)."""
    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )
    return landmarks


def draw_hand_landmarks(
    frame: np.ndarray,
    hand_landmarks,
    mp_hands,
    mp_drawing,
    mp_drawing_styles,
) -> None:
    """Draw hand landmarks and connections on frame."""
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )


def draw_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int], color: tuple) -> None:
    """Draw bounding box on frame."""
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)


def draw_overlay(
    frame: np.ndarray,
    letter: str,
    count: int,
    recording: bool,
    record_progress: int,
    hand_detected: bool,
) -> None:
    """Draw status overlay on frame."""
    h, w = frame.shape[:2]

    # Red border when recording
    if recording:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)

    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 70), (40, 40, 40), -1)

    # Instructions or recording status
    if recording:
        progress_text = f"RECORDING: {record_progress}/{RECORD_FRAMES} frames"
        cv2.putText(
            frame,
            progress_text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
        # Progress bar
        bar_width = int((w - 20) * record_progress / RECORD_FRAMES)
        cv2.rectangle(frame, (10, 50), (w - 10, 65), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 50), (10 + bar_width, 65), (0, 0, 255), -1)
    else:
        cv2.putText(
            frame,
            "Press A-Z to record | ESC to quit",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        # Hand detection status
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        status_text = "Hand: DETECTED" if hand_detected else "Hand: NOT FOUND"
        cv2.putText(
            frame,
            status_text,
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            1,
        )

        # Last capture info
        if letter:
            display = "Ñ" if letter == "nn" else letter.upper()
            cv2.putText(
                frame,
                f"Last: {display} | Total for {display}: {count}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )


def main() -> None:
    """Main capture loop."""
    print("=" * 50)
    print("LSC Landmark Capture Tool")
    print("=" * 50)
    print(f"Press any letter key (a-z) to record {RECORD_FRAMES}-frame landmark sequence")
    print("Press DOWN ARROW for Ñ")
    print("Press ESC to quit")
    print("=" * 50)
    print(f"\nData format: {RECORD_FRAMES} frames x 21 landmarks x 3 coords (x,y,z)")

    # Setup directories
    ensure_letter_dirs()

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

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_letter = ""
    letter_count = 0

    # Recording state
    recording = False
    record_landmarks: list[np.ndarray] = []
    record_letter = ""

    print("\nCamera ready! Position your hand and press a letter key.")
    print("Hand landmarks will be shown when detected.")

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
        current_landmarks = None

        # Process hand landmarks
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on display
                draw_hand_landmarks(
                    frame, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles
                )

                # Get and draw bounding box
                bbox = get_hand_bbox(hand_landmarks, w, h)
                draw_bbox(frame, bbox, (255, 0, 255))  # Magenta bbox

                # Extract landmark coordinates
                current_landmarks = extract_landmarks(hand_landmarks)

        # Handle recording
        if recording:
            if not hand_detected or current_landmarks is None:
                print("[!] Hand lost during recording - aborting")
                recording = False
                record_landmarks = []
            else:
                # Store landmarks for this frame
                record_landmarks.append(current_landmarks)

                # Check if recording is complete
                if len(record_landmarks) >= RECORD_FRAMES:
                    filepath = get_next_filename(record_letter)

                    # Stack into array (RECORD_FRAMES x 21 x 3)
                    landmarks_array = np.stack(record_landmarks, axis=0)
                    np.save(filepath, landmarks_array)

                    # Update state
                    last_letter = record_letter
                    letter_count = len(list((DATA_DIR / record_letter).glob("*.npy")))

                    display_letter = "Ñ" if record_letter == "nn" else record_letter.upper()
                    print(
                        f"Saved: {filepath.name} (shape: {landmarks_array.shape}) "
                        f"| Total {display_letter}: {letter_count}"
                    )

                    # Reset recording state
                    recording = False
                    record_landmarks = []

        # Draw overlay
        record_progress = len(record_landmarks) if recording else 0
        draw_overlay(
            frame, last_letter, letter_count, recording, record_progress, hand_detected
        )

        # Show frame
        cv2.imshow(WINDOW_NAME, frame)

        # Handle key press (use waitKeyEx for arrow keys)
        key = cv2.waitKeyEx(1)

        # Quit on ESC
        if key == 27:
            print("\nExiting...")
            break

        # Start recording on letter keys (a-z) or DOWN ARROW for ñ
        letter = None
        if ord("a") <= key <= ord("z") and not recording:
            letter = chr(key)
        # Down arrow: 65364 (Linux GTK), 2621440 (Windows), 84 (some systems)
        elif key in (65364, 2621440, 84) and not recording:
            letter = "nn"

        if letter is not None:
            if not hand_detected or current_landmarks is None:
                display_letter = "Ñ" if letter == "nn" else letter.upper()
                print(f"[!] No hand detected - cannot record '{display_letter}'")
                continue

            # Start recording
            recording = True
            record_letter = letter
            record_landmarks = []
            display_letter = "Ñ" if letter == "nn" else letter.upper()
            print(f"Recording '{display_letter}'...")

    # Cleanup
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\nCapture session ended.")
    print(f"Landmarks saved to: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
