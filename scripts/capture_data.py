"""
Data capture script for Colombian Sign Language gestures.

Usage:
    python scripts/capture_data.py

Controls:
    - Press any letter key (a-z) to capture a photo for that letter
    - Press ESC or 'q' to quit
    - Photos are saved to data/raw/{letter}/

Features:
    - Hand detection with MediaPipe
    - 21 landmark points visualization
    - Auto-crops to hand region with padding
"""

import cv2
import time
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime


# Configuration
DATA_DIR = Path("../data/raw")
WINDOW_NAME = "LSC Data Capture - Press letter keys to capture"
CAPTURE_FEEDBACK_DURATION = 0.3  # seconds to show green border
CROP_PADDING = 50  # pixels of padding around hand
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5


def ensure_letter_dirs() -> None:
    """Create directories for each letter a-z."""
    for letter in "abcdefghijklmnopqrstuvwxyz":
        letter_dir = DATA_DIR / letter
        letter_dir.mkdir(parents=True, exist_ok=True)


def get_next_filename(letter: str) -> Path:
    """Get next available filename for a letter."""
    letter_dir = DATA_DIR / letter
    existing = list(letter_dir.glob("*.jpg"))
    next_num = len(existing) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return letter_dir / f"{letter}_{next_num:04d}_{timestamp}.jpg"


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
    capturing: bool,
    hand_detected: bool,
) -> None:
    """Draw status overlay on frame."""
    h, w = frame.shape[:2]
    
    # Green border when capturing
    if capturing:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 10)
    
    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 70), (40, 40, 40), -1)
    
    # Instructions
    cv2.putText(
        frame,
        "Press A-Z to capture | ESC/Q to quit",
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
        cv2.putText(
            frame,
            f"Last: {letter.upper()} | Total for {letter.upper()}: {count}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def main() -> None:
    """Main capture loop."""
    print("=" * 50)
    print("LSC Data Capture Tool (with Hand Detection)")
    print("=" * 50)
    print("Press any letter key (a-z) to capture a photo")
    print("Press ESC  to quit")
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
    capture_time = 0
    
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
        
        # Check if we're in capture feedback period
        capturing = (time.time() - capture_time) < CAPTURE_FEEDBACK_DURATION
        
        # Draw overlay
        draw_overlay(frame, last_letter, letter_count, capturing, hand_detected)
        
        # Show frame
        cv2.imshow(WINDOW_NAME, frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        
        # Quit on ESC or 'q'
        if key == 27 :
            print("\nExiting...")
            break
        
        # Capture on letter keys (a-z) - only if hand is detected
        if ord('a') <= key <= ord('z'):
            letter = chr(key)
            
            if not hand_detected or current_bbox is None:
                print(f"[!] No hand detected - cannot capture '{letter.upper()}'")
                continue
            
            filepath = get_next_filename(letter)
            
            # Get a clean frame for saving (without overlay)
            ret, clean_frame = cap.read()
            if ret:
                clean_frame = cv2.flip(clean_frame, 1)
                
                # Process clean frame to get landmarks
                rgb_clean = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
                clean_results = hands.process(rgb_clean)
                
                if clean_results.multi_hand_landmarks:
                    clean_landmarks = clean_results.multi_hand_landmarks[0]
                    clean_bbox = get_hand_bbox(clean_landmarks, w, h)
                    
                    # Crop to hand region
                    hand_crop = crop_hand(clean_frame, clean_bbox)
                    
                    # Save cropped hand image
                    cv2.imwrite(str(filepath), hand_crop)
                    
                    # Update state
                    last_letter = letter
                    letter_count = len(list((DATA_DIR / letter).glob("*.jpg")))
                    capture_time = time.time()
                    
                    crop_size = f"{hand_crop.shape[1]}x{hand_crop.shape[0]}"
                    print(f"Captured: {filepath.name} ({crop_size}) | Total {letter.upper()}: {letter_count}")
                else:
                    print(f"[!] Hand lost during capture - try again")
    
    # Cleanup
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nCapture session ended.")
    print(f"Images saved to: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
