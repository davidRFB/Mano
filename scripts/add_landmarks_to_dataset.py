"""
Batch process dataset to add MediaPipe hand landmarks.

Usage:
    python scripts/add_landmarks_to_dataset.py
    python scripts/add_landmarks_to_dataset.py --confidence 0.3

Input:  data/raw/{a-z}/*.jpg (cropped hand images)
Output: data/raw_landmarks/{a-z}/*.jpg (same images + MediaPipe landmarks)

Skips images where no hand is detected and logs them for review.

Note: MediaPipe's hand detector is trained on full frames. Cropped hand images
may require lower detection confidence (0.3-0.5) for better detection rates.
"""

import argparse
import cv2
import mediapipe as mp
from pathlib import Path
from datetime import datetime


# Configuration
INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/raw_landmarks")
DEFAULT_CONFIDENCE = 0.1  # Lower than capture_data.py due to cropped images

# Statistics
stats = {
    "processed": 0,
    "skipped": 0,
    "failed_files": [],
}


def ensure_output_dirs() -> None:
    """Create output directories for each letter a-z."""
    for letter in "abcdefghijklmnopqrstuvwxyz":
        letter_dir = OUTPUT_DIR / letter
        letter_dir.mkdir(parents=True, exist_ok=True)


def process_image(
    input_path: Path,
    output_path: Path,
    hands: mp.solutions.hands.Hands,
    mp_drawing: mp.solutions.drawing_utils,
    mp_hands: mp.solutions.hands,
    mp_drawing_styles: mp.solutions.drawing_styles,
) -> bool:
    """
    Process a single image: detect hand and draw landmarks.

    Args:
        input_path: Path to input image
        output_path: Path to save output image
        hands: MediaPipe Hands instance
        mp_drawing: MediaPipe drawing utils
        mp_hands: MediaPipe hands module
        mp_drawing_styles: MediaPipe drawing styles

    Returns:
        True if successful, False if hand not detected
    """
    # Read image
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"  [ERROR] Could not read: {input_path}")
        return False

    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(rgb_image)

    # Check if hand detected
    if not results.multi_hand_landmarks:
        return False

    # Draw landmarks on the image (same style as inference.py)
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    # Save output image
    cv2.imwrite(str(output_path), image)
    return True


def process_dataset(confidence: float = DEFAULT_CONFIDENCE) -> None:
    """Process all images in the dataset."""
    print("=" * 60)
    print("LSC Dataset - Add MediaPipe Landmarks")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR.absolute()}")
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print(f"Detection confidence: {confidence}")
    print("=" * 60)

    # Create output directories
    ensure_output_dirs()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=True,  # For batch processing (no tracking)
        max_num_hands=1,
        min_detection_confidence=confidence,
    )

    # Get all letter directories
    letter_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])

    if not letter_dirs:
        print(f"[ERROR] No letter directories found in {INPUT_DIR}")
        return

    print(f"\nProcessing {len(letter_dirs)} letter directories...\n")

    # Process each letter directory
    for letter_dir in letter_dirs:
        letter = letter_dir.name
        output_letter_dir = OUTPUT_DIR / letter

        # Get all images in this letter directory
        images = sorted(letter_dir.glob("*.jpg"))

        if not images:
            print(f"[{letter.upper()}] No images found, skipping")
            continue

        letter_processed = 0
        letter_skipped = 0

        for img_path in images:
            output_path = output_letter_dir / img_path.name

            success = process_image(
                img_path,
                output_path,
                hands,
                mp_drawing,
                mp_hands,
                mp_drawing_styles,
            )

            if success:
                letter_processed += 1
                stats["processed"] += 1
            else:
                letter_skipped += 1
                stats["skipped"] += 1
                stats["failed_files"].append(str(img_path))

        # Progress report for this letter
        total = letter_processed + letter_skipped
        skip_pct = (letter_skipped / total * 100) if total > 0 else 0
        print(
            f"[{letter.upper()}] {letter_processed}/{total} processed "
            f"({letter_skipped} skipped, {skip_pct:.1f}%)"
        )

    # Cleanup
    hands.close()

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = stats["processed"] + stats["skipped"]
    rejection_rate = (stats["skipped"] / total * 100) if total > 0 else 0

    print(f"Total images:     {total}")
    print(f"Processed:        {stats['processed']}")
    print(f"Skipped:          {stats['skipped']}")
    print(f"Rejection rate:   {rejection_rate:.2f}%")

    if rejection_rate > 5:
        print(f"\n⚠️  WARNING: Rejection rate ({rejection_rate:.2f}%) exceeds 5% threshold")
    else:
        print(f"\n✅ Rejection rate within acceptable threshold (<5%)")

    # Log failed files if any
    if stats["failed_files"]:
        log_path = OUTPUT_DIR / "failed_detections.txt"
        with open(log_path, "w") as f:
            f.write(f"# Failed hand detections - {datetime.now().isoformat()}\n")
            f.write(f"# Total: {len(stats['failed_files'])} files\n\n")
            for path in stats["failed_files"]:
                f.write(f"{path}\n")
        print(f"\nFailed files logged to: {log_path}")

    print(f"\nOutput saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add MediaPipe landmarks to hand gesture dataset"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Min detection confidence (default: {DEFAULT_CONFIDENCE})",
    )
    args = parser.parse_args()

    process_dataset(confidence=args.confidence)

