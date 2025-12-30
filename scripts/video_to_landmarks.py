"""
Extract landmarks from video files.

Usage:
    # Process videos - filename = label (e.g., hola.mp4 -> word "hola")
    python scripts/video_to_landmarks.py videos/        # folder of videos
    python scripts/video_to_landmarks.py hola.mp4       # single video

    # For letters (hand only, no pose)
    python scripts/video_to_landmarks.py videos/ --letters

Output:
    Words:   data/raw_words/{word}/*.npy   (default, holistic)
    Letters: data/raw_landmarks/{letter}/*.npy
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime

# Output directories
LANDMARKS_DIR = Path("./data/raw_landmarks")
WORDS_DIR = Path("./data/raw_words")

# Upper body indices for holistic mode
UPPER_BODY_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]


def extract_hand_landmarks(hand_landmarks) -> np.ndarray:
    """Extract 21 hand landmarks (21 x 3)."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )


def extract_pose_landmarks(pose_landmarks) -> np.ndarray:
    """Extract upper body pose landmarks (9 x 3)."""
    if pose_landmarks is None:
        return np.zeros((len(UPPER_BODY_INDICES), 3), dtype=np.float32)
    landmarks = []
    for idx in UPPER_BODY_INDICES:
        lm = pose_landmarks.landmark[idx]
        landmarks.append([lm.x, lm.y, lm.z])
    return np.array(landmarks, dtype=np.float32)


def process_video_hands(video_path: Path, min_confidence: float = 0.5) -> np.ndarray | None:
    """
    Process video and extract hand landmarks.

    Returns:
        (num_frames, 21, 3) array or None if no hand detected
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Cannot open {video_path}")
        return None

    all_landmarks = []
    frame_count = 0
    detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            landmarks = extract_hand_landmarks(results.multi_hand_landmarks[0])
            all_landmarks.append(landmarks)
            detected_count += 1
        # Skip frames with no detection

    cap.release()
    hands.close()

    if not all_landmarks:
        print(f"  No hands detected in {frame_count} frames")
        return None

    print(f"  Detected hand in {detected_count}/{frame_count} frames")
    return np.stack(all_landmarks, axis=0)


def process_video_holistic(video_path: Path, min_confidence: float = 0.5) -> np.ndarray | None:
    """
    Process video with holistic (pose + hands).

    Returns:
        (num_frames, 51, 3) array - 9 pose + 21 left + 21 right
    """
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Cannot open {video_path}")
        return None

    all_landmarks = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Extract all parts (zeros if not detected)
        pose = extract_pose_landmarks(results.pose_landmarks)

        if results.left_hand_landmarks:
            left = extract_hand_landmarks(results.left_hand_landmarks)
        else:
            left = np.zeros((21, 3), dtype=np.float32)

        if results.right_hand_landmarks:
            right = extract_hand_landmarks(results.right_hand_landmarks)
        else:
            right = np.zeros((21, 3), dtype=np.float32)

        # Combine: (51, 3)
        frame_landmarks = np.concatenate([pose, left, right], axis=0)
        all_landmarks.append(frame_landmarks)

    cap.release()
    holistic.close()

    if not all_landmarks:
        return None

    print(f"  Processed {frame_count} frames")
    return np.stack(all_landmarks, axis=0)


def get_output_path(label: str, is_word: bool) -> Path:
    """Get output path for landmarks."""
    if is_word:
        out_dir = WORDS_DIR / label.lower()
    else:
        out_dir = LANDMARKS_DIR / label.lower()

    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.npy"))
    next_num = len(existing) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return out_dir / f"{label}_{next_num:04d}_{timestamp}.npy"


def process_single(
    video_path: Path,
    label: str,
    is_word: bool = False,
    holistic: bool = False,
    min_confidence: float = 0.5,
) -> bool:
    """Process a single video file."""
    print(f"\nProcessing: {video_path.name}")

    if holistic or is_word:
        landmarks = process_video_holistic(video_path, min_confidence)
    else:
        landmarks = process_video_hands(video_path, min_confidence)

    if landmarks is None:
        return False

    output_path = get_output_path(label, is_word)
    np.save(output_path, landmarks)
    print(f"  Saved: {output_path.name} | Shape: {landmarks.shape}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract landmarks from video files")
    parser.add_argument("input", type=str, help="Video file or folder of videos")
    parser.add_argument("--letters", action="store_true", help="Letter mode (hand only, no pose)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Min detection confidence")

    args = parser.parse_args()

    input_path = Path(args.input)
    is_word = not args.letters  # Default: word mode (holistic)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if input_path.is_file():
        # Single video - filename is the label
        label = input_path.stem.lower()  # hola.mp4 -> "hola"
        process_single(input_path, label, is_word, holistic=is_word, min_confidence=args.confidence)

    elif input_path.is_dir():
        # Folder of videos - each filename is its label
        videos = [f for f in input_path.iterdir() if f.suffix.lower() in video_extensions]

        if not videos:
            print(f"No video files found in {input_path}")
            return

        print(f"Found {len(videos)} videos")
        print(f"Mode: {'words (holistic)' if is_word else 'letters (hand only)'}")
        print("-" * 40)

        success = 0
        for video in sorted(videos):
            label = video.stem.lower()  # filename without extension
            if process_single(video, label, is_word, holistic=is_word, min_confidence=args.confidence):
                success += 1

        print(f"\nProcessed {success}/{len(videos)} videos successfully")
    else:
        print(f"Error: {input_path} not found")


if __name__ == "__main__":
    main()
