"""
Process YouTube Colombian Sign Language dataset.

Cuts first 3 seconds of each video (normal speed portion).
Saves processed video with cleaned filename.

Usage:
    python scripts/process_youtube_dataset.py data/webscrapping/you_dataset/ --limit 5
    python scripts/process_youtube_dataset.py data/webscrapping/you_dataset/ --workers 8
"""

import argparse
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

OUTPUT_DIR = Path("./data/webscrapping/youtube_dataset_proc")
CUT_DURATION =  2.5


def clean_filename(filename: str) -> str | None:
    """
    Extract clean label from YouTube video filename.

    "Abandonar - Lengua de señas colombiana.mp4" → "abandonar"
    "Abiótico (biología) - Lengua..." → "abiotico"
    "Acabar 1 - Lengua..." → "acabar_1"
    """
    name = Path(filename).stem

    if " - Lengua de señas" not in name:
        return None

    word_part = name.split(" - Lengua de señas")[0].strip()
    word_part = re.sub(r'\s*\([^)]+\)\s*', ' ', word_part).strip()
    word_part = re.sub(r'\s+', '_', word_part)
    word_part = unicodedata.normalize('NFD', word_part)
    word_part = ''.join(c for c in word_part if unicodedata.category(c) != 'Mn')
    word_part = word_part.lower()
    word_part = re.sub(r'[^a-z0-9_]', '', word_part)

    return word_part if word_part else None


def cut_video(input_path: Path, output_path: Path, duration: float) -> tuple[bool, int, int]:
    """Cut first N seconds of video. Returns (success, cut_frames, total_frames)."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cut_frames = int(fps * duration)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for _ in range(cut_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    return True, cut_frames, total_frames


def process_single_video(
    args: tuple[Path, Path, float]
) -> tuple[str, Path | None, bool, int, int]:
    """Worker function for parallel processing. Returns (label, output_path, success, cut, total)."""
    video, output_path, duration = args
    label = clean_filename(video.name)

    if label is None:
        return video.name, None, False, 0, 0

    ok, cut, total = cut_video(video, output_path, duration)
    return label, output_path, ok, cut, total


def main():
    parser = argparse.ArgumentParser(description="Process YouTube sign language dataset")
    parser.add_argument("input", type=str, help="Folder of videos")
    parser.add_argument("--duration", type=float, default=CUT_DURATION, help="Cut duration in seconds")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"Error: {input_path} is not a directory")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted([f for f in input_path.iterdir() if f.suffix.lower() == ".mp4"])

    if args.limit > 0:
        videos = videos[:args.limit]

    # Pre-compute output paths (handle duplicates sequentially)
    tasks = []
    used_paths: set[Path] = set()
    for video in videos:
        label = clean_filename(video.name)
        if label is None:
            tasks.append((video, None, args.duration))
            continue

        output_path = OUTPUT_DIR / f"{label}.mp4"
        if output_path.exists() or output_path in used_paths:
            n = 2
            while (OUTPUT_DIR / f"{label}_{n}.mp4").exists() or (OUTPUT_DIR / f"{label}_{n}.mp4") in used_paths:
                n += 1
            output_path = OUTPUT_DIR / f"{label}_{n}.mp4"

        used_paths.add(output_path)
        tasks.append((video, output_path, args.duration))

    print(f"Videos: {len(videos)} | Cut: {args.duration}s | Workers: {args.workers} | Output: {OUTPUT_DIR}")
    print("-" * 60)

    success = 0
    skipped = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit only valid tasks
        future_to_idx = {}
        for i, (video, output_path, duration) in enumerate(tasks):
            if output_path is None:
                skipped += 1
                print(f"SKIP: {video.name[:50]}...")
                continue
            future = executor.submit(process_single_video, (video, output_path, duration))
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            label, output_path, ok, cut, total = future.result()
            if ok:
                print(f"✓ {label} → {cut}/{total} frames → {output_path.name}")
                success += 1
            else:
                print(f"✗ ERROR: {label[:50]}...")
                errors += 1

    print(f"\n{'='*60}")
    print(f"Done: {success} success | {skipped} skipped | {errors} errors")


if __name__ == "__main__":
    main()
