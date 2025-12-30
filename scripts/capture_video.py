"""
Video capture script for Colombian Sign Language gestures.

Usage:
    python scripts/capture_video.py

Controls:
    - Press any letter key (a-z) to start recording a video for that letter
    - Press DOWN ARROW to record Ñ (saved as nn/)
    - Press ESC to quit
    - Videos are saved to data/raw_video/{letter}/

Features:
    - Hand detection with MediaPipe
    - 21 landmark points visualization
    - Auto-crops to hand region with padding
    - Records 50-frame videos per gesture
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime
import os

# Configuration
DATA_DIR = Path("../data/raw_video")
os.makedirs(DATA_DIR, exist_ok=True)
WINDOW_NAME = "LSC Video Capture - Press letter keys to record"
CROP_PADDING = 50  # pixels of padding around hand
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Video settings
VIDEO_FRAMES = 50  # number of frames per video
VIDEO_FPS = 15  # frames per second for saved video


def ensure_letter_dirs() -> None:
    """Create directories for each letter a-z plus nn for ñ."""
    letters = list("abcdefghijklmnopqrstuvwxyz") + ["nn"]
    for letter in letters:
        letter_dir = DATA_DIR / letter
        letter_dir.mkdir(parents=True, exist_ok=True)


def get_next_filename(letter: str) -> Path:
    """Get next available filename for a letter (nn for ñ)."""
    letter_dir = DATA_DIR / letter
    existing = list(letter_dir.glob("*.avi"))
    next_num = len(existing) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return letter_dir / f"{letter}_{next_num:04d}_{timestamp}.avi"


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
        progress_text = f"RECORDING: {record_progress}/{VIDEO_FRAMES} frames"
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
        bar_width = int((w - 20) * record_progress / VIDEO_FRAMES)
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
    print("LSC Video Capture Tool (with Hand Detection)")
    print("=" * 50)
    print(f"Press any letter key (a-z) to record {VIDEO_FRAMES}-frame video")
    print("Press DOWN ARROW for Ñ")
    print("Press ESC to quit")
    print("=" * 50)

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
    record_frames: list[np.ndarray] = []
    record_letter = ""
    record_bbox: tuple[int, int, int, int] | None = None

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
        current_bbox = None

        # Process hand landmarks
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                draw_hand_landmarks(
                    frame, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles
                )

                # Get and draw bounding box
                current_bbox = get_hand_bbox(hand_landmarks, w, h)
                draw_bbox(frame, current_bbox, (255, 0, 255))  # Magenta bbox

        # Handle recording
        if recording:
            if not hand_detected:
                print(f"[!] Hand lost during recording - aborting")
                recording = False
                record_frames = []
                record_bbox = None
            else:
                # Crop frame using the initial bbox for consistency
                hand_crop = crop_hand(frame.copy(), record_bbox)
                record_frames.append(hand_crop)

                # Check if recording is complete
                if len(record_frames) >= VIDEO_FRAMES:
                    filepath = get_next_filename(record_letter)

                    # Determine video size from first frame
                    crop_h, crop_w = record_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    out = cv2.VideoWriter(
                        str(filepath), fourcc, VIDEO_FPS, (crop_w, crop_h)
                    )

                    for f in record_frames:
                        # Resize if needed to match first frame
                        if f.shape[:2] != (crop_h, crop_w):
                            f = cv2.resize(f, (crop_w, crop_h))
                        out.write(f)

                    out.release()

                    # Update state
                    last_letter = record_letter
                    letter_count = len(list((DATA_DIR / record_letter).glob("*.avi")))

                    display_letter = "Ñ" if record_letter == "nn" else record_letter.upper()
                    print(
                        f"Saved: {filepath.name} ({crop_w}x{crop_h}, {VIDEO_FRAMES} frames) "
                        f"| Total {display_letter}: {letter_count}"
                    )

                    # Reset recording state
                    recording = False
                    record_frames = []
                    record_bbox = None

        # Draw overlay
        record_progress = len(record_frames) if recording else 0
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
            if not hand_detected or current_bbox is None:
                display_letter = "Ñ" if letter == "nn" else letter.upper()
                print(f"[!] No hand detected - cannot record '{display_letter}'")
                continue

            # Start recording
            recording = True
            record_letter = letter
            record_bbox = current_bbox
            record_frames = []
            display_letter = "Ñ" if letter == "nn" else letter.upper()
            print(f"Recording '{display_letter}'...")

    # Cleanup
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\nCapture session ended.")
    print(f"Videos saved to: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
