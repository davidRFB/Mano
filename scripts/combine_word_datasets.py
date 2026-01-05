"""
Combine word video datasets via symlinks.

Creates symlinks from INSOR and YouTube videos to a combined folder.
Keeps all samples (including duplicates) for more training data.

Usage:
    python scripts/combine_word_datasets.py --dry-run   # preview
    python scripts/combine_word_datasets.py             # execute
    python scripts/combine_word_datasets.py --stats     # show statistics only
"""

import argparse
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

INSOR_DIR = Path("./data/webscrapping/insor_dataset/videos")
YOUTUBE_DIR = Path("./data/webscrapping/youtube_dataset_proc")
OUTPUT_DIR = Path("./data/webscrapping/combined_words")


def normalize_name(filename: str) -> str:
    """Normalize filename: lowercase, remove accents, clean."""
    name = Path(filename).stem
    # Remove _2, _3 suffixes from YouTube duplicates
    name = re.sub(r"_\d+$", "", name)
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower()
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def is_single_word(filename: str) -> bool:
    """Check if filename represents a single word (no underscores, except trailing numbers)."""
    stem = Path(filename).stem
    # Remove trailing _2, _3 etc (YouTube duplicates)
    stem = re.sub(r"_\d+$", "", stem)
    return "_" not in stem


def get_unique_path(base_path: Path, used_paths: set[Path]) -> Path:
    """Get unique path, adding suffix if needed."""
    if not base_path.exists() and base_path not in used_paths:
        return base_path

    n = 2
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    while True:
        new_path = parent / f"{stem}_{n}{suffix}"
        if not new_path.exists() and new_path not in used_paths:
            return new_path
        n += 1


def collect_videos() -> tuple[list[tuple[Path, str, str]], dict[str, list[str]]]:
    """
    Collect all videos from both sources.

    Returns:
        videos: List of (path, normalized_label, source)
        word_samples: Dict mapping word -> list of sources
    """
    videos = []
    word_samples: dict[str, list[str]] = defaultdict(list)

    # INSOR single-word videos
    if INSOR_DIR.exists():
        for f in INSOR_DIR.iterdir():
            if f.suffix.lower() in {".mp4", ".m4v", ".mov"} and is_single_word(f.name):
                label = normalize_name(f.name)
                if label:
                    videos.append((f, label, "insor"))
                    word_samples[label].append("insor")

    # YouTube processed videos
    if YOUTUBE_DIR.exists():
        for f in YOUTUBE_DIR.glob("*.mp4"):
            label = normalize_name(f.name)
            if label:
                videos.append((f, label, "youtube"))
                word_samples[label].append("youtube")

    return videos, dict(word_samples)


def show_statistics(word_samples: dict[str, list[str]]) -> None:
    """Display dataset statistics."""
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    # Count by source
    insor_only = sum(1 for sources in word_samples.values() if sources == ["insor"])
    youtube_only = sum(1 for sources in word_samples.values() if all(s == "youtube" for s in sources))
    overlap = sum(1 for sources in word_samples.values() if "insor" in sources and "youtube" in sources)

    print(f"\nWords by source:")
    print(f"  INSOR only:    {insor_only}")
    print(f"  YouTube only:  {youtube_only}")
    print(f"  Both sources:  {overlap}")
    print(f"  Total unique:  {len(word_samples)}")

    # Sample distribution
    sample_counts = [len(sources) for sources in word_samples.values()]
    print(f"\nSamples per word:")
    for n in sorted(set(sample_counts)):
        count = sample_counts.count(n)
        pct = count / len(word_samples) * 100
        bar = "█" * min(50, count // 10)
        print(f"  {n} sample(s): {count:4d} words ({pct:5.1f}%) {bar}")

    total_videos = sum(sample_counts)
    print(f"\nTotal videos: {total_videos}")

    # Words with most samples
    print(f"\nWords with most samples:")
    sorted_words = sorted(word_samples.items(), key=lambda x: len(x[1]), reverse=True)
    for word, sources in sorted_words[:10]:
        source_str = ", ".join(f"{s}({sources.count(s)})" for s in set(sources))
        print(f"  {word}: {len(sources)} ({source_str})")


def main():
    parser = argparse.ArgumentParser(description="Combine word datasets via symlinks")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    args = parser.parse_args()

    # Collect all videos
    videos, word_samples = collect_videos()

    if not videos:
        print("No videos found!")
        return

    # Show statistics
    show_statistics(word_samples)

    if args.stats:
        return

    print("\n" + "=" * 60)
    print("CREATING SYMLINKS")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)

    if not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    used_paths: set[Path] = set()
    created = {"insor": 0, "youtube": 0}
    skipped = 0

    for video_path, label, source in sorted(videos, key=lambda x: (x[1], x[2])):
        output_path = get_unique_path(OUTPUT_DIR / f"{label}.mp4", used_paths)
        used_paths.add(output_path)

        if args.dry_run:
            # Only show first few per source
            if created[source] < 5:
                print(f"[{source:7s}] {label} -> {output_path.name}")
            elif created[source] == 5:
                print(f"[{source:7s}] ...")
            created[source] += 1
        else:
            try:
                output_path.symlink_to(video_path.resolve())
                created[source] += 1
            except FileExistsError:
                skipped += 1

    print("-" * 60)

    if args.dry_run:
        print(f"Would create: INSOR={created['insor']}, YouTube={created['youtube']}")
        print(f"Total: {created['insor'] + created['youtube']} symlinks")
    else:
        print(f"Created: INSOR={created['insor']}, YouTube={created['youtube']}")
        print(f"Total: {created['insor'] + created['youtube']} | Skipped: {skipped}")

    print(f"\nNext steps:")
    print(f"  1. Extract landmarks:")
    print(f"     python scripts/video_to_landmarks.py {OUTPUT_DIR}")
    print(f"  2. Train model:")
    print(f"     python -m src.cv_model.words_train --model bigru")


if __name__ == "__main__":
    main()
