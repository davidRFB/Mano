"""
INSOR Colombian Sign Language Dictionary Scraper

Scrapes sign language videos and annotations from:
https://educativo.insor.gov.co/diccionario/

Each word entry contains:
- Word (palabra)
- Definition (definición)
- Example sentence (ejemplo)
- 3 videos: word sign, definition in sign, example in sign
"""

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup


def sanitize_filename(text: str, max_length: int = 200) -> str:
    """
    Convert text to a valid filename.

    - Replace spaces with underscores
    - Remove or replace invalid characters
    - Truncate if too long
    """
    if not text:
        return "unknown"

    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # Replace spaces with underscores
    text = text.replace(" ", "_")

    # Remove invalid filename characters
    invalid_chars = r'[<>:"/\\|?*\n\r\t]'
    text = re.sub(invalid_chars, "", text)

    # Remove trailing periods and dots (Windows issue)
    text = text.strip("._")

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length].rstrip("_")

    return text or "unknown"


@dataclass
class WordEntry:
    """Represents a dictionary entry with word, definition, example and video URLs."""

    slug: str
    word: str
    definition: str
    example: str
    video_word: Optional[str] = None
    video_definition: Optional[str] = None
    video_example: Optional[str] = None


class INSORScraper:
    """Scraper for INSOR Colombian Sign Language dictionary."""

    BASE_URL = "https://educativo.insor.gov.co"
    CATEGORY_URL = f"{BASE_URL}/catdiccionario"
    DICTIONARY_URL = f"{BASE_URL}/diccionario"
    LETTERS = list("abcdefghijklmnopqrstuvwxyz")

    def __init__(self, output_dir: str = "output", delay: float = 0.5):
        """
        Initialize scraper.

        Args:
            output_dir: Directory to save videos and annotations
            delay: Delay between requests in seconds
        """
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            }
        )

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch URL and return BeautifulSoup object."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            time.sleep(self.delay)
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def get_word_links_for_letter(self, letter: str) -> list[str]:
        """Get all word page URLs for a given letter."""
        word_links = set()
        page = 1

        while True:
            if page == 1:
                url = f"{self.CATEGORY_URL}/{letter}/"
            else:
                url = f"{self.CATEGORY_URL}/{letter}/page/{page}/"

            soup = self._get_soup(url)
            if not soup:
                break

            # Find all dictionary word links
            pattern = re.compile(rf"{self.DICTIONARY_URL}/[^/]+/$")
            links = soup.find_all("a", href=pattern)

            if not links:
                break

            new_links = {link["href"] for link in links}

            # Check if we got any new links (pagination end)
            if not new_links - word_links:
                break

            word_links.update(new_links)
            print(f"  Letter {letter.upper()}, page {page}: found {len(new_links)} words")
            page += 1

        return list(word_links)

    def get_all_word_links(self, letters: Optional[list[str]] = None) -> list[str]:
        """Get all word page URLs for specified letters."""
        if letters is None:
            letters = self.LETTERS

        all_links = []
        for letter in letters:
            print(f"Fetching words for letter: {letter.upper()}")
            links = self.get_word_links_for_letter(letter)
            all_links.extend(links)
            print(f"  Total for {letter.upper()}: {len(links)} words")

        return all_links

    def parse_word_page(self, url: str) -> Optional[WordEntry]:
        """Parse a word page and extract all information."""
        soup = self._get_soup(url)
        if not soup:
            return None

        # Extract slug from URL
        slug = url.rstrip("/").split("/")[-1]

        # Extract text content from texto-normal divs
        text_divs = soup.find_all("div", class_="texto-normal")
        texts = [div.get_text(strip=True) for div in text_divs]

        # Expected order: word, definition, example
        word = texts[0] if len(texts) > 0 else ""
        definition = texts[1] if len(texts) > 1 else ""
        example = texts[2] if len(texts) > 2 else ""

        # Extract video URLs from vid-cargado attributes
        video_inputs = soup.find_all("input", attrs={"vid-cargado": True})
        videos = [inp.get("vid-cargado", "") for inp in video_inputs]

        # Expected order: word video, definition video, example video
        video_word = videos[0] if len(videos) > 0 else None
        video_definition = videos[1] if len(videos) > 1 else None
        video_example = videos[2] if len(videos) > 2 else None

        return WordEntry(
            slug=slug,
            word=word,
            definition=definition,
            example=example,
            video_word=video_word,
            video_definition=video_definition,
            video_example=video_example,
        )

    def download_video(self, url: str, output_path: Path) -> bool:
        """Download a video file."""
        if output_path.exists():
            return True

        try:
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            time.sleep(self.delay)
            return True
        except requests.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return False

    def download_entry_videos(self, entry: WordEntry) -> dict[str, str]:
        """
        Download all videos for a word entry.

        Videos are named based on the annotation text:
        - Word video: word text (e.g., "Hola.m4v" or "A_veces.m4v")
        - Definition video: definition text with underscores
        - Example video: example sentence with underscores

        Returns:
            Dict mapping video type to output filename (or empty if failed)
        """
        video_dir = self.output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        for video_type, url, text in [
            ("word", entry.video_word, entry.word),
            ("definition", entry.video_definition, entry.definition),
            ("example", entry.video_example, entry.example),
        ]:
            if url and text:
                ext = Path(url).suffix or ".m4v"
                filename = sanitize_filename(text) + ext
                output_path = video_dir / filename

                if self.download_video(url, output_path):
                    results[video_type] = filename
                else:
                    results[video_type] = ""
            else:
                results[video_type] = ""

        return results

    def scrape_and_save(
        self,
        letters: Optional[list[str]] = None,
        download_videos: bool = True,
        limit: Optional[int] = None,
    ) -> list[WordEntry]:
        """
        Scrape dictionary entries and save to files.

        Args:
            letters: List of letters to scrape (default: all)
            download_videos: Whether to download video files
            limit: Maximum number of words to scrape (for testing)

        Returns:
            List of scraped WordEntry objects
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get all word links
        print("=== Fetching word links ===")
        word_links = self.get_all_word_links(letters)
        print(f"\nTotal words found: {len(word_links)}")

        if limit:
            word_links = word_links[:limit]
            print(f"Limited to: {limit} words")

        # Parse each word page
        print("\n=== Parsing word pages ===")
        entries = []
        saved_files = []  # Track actual saved filenames
        for i, url in enumerate(word_links, 1):
            print(f"[{i}/{len(word_links)}] Parsing: {url}")
            entry = self.parse_word_page(url)
            if entry:
                entries.append(entry)

                # Download videos if requested
                if download_videos:
                    results = self.download_entry_videos(entry)
                    saved_files.append(
                        {
                            "slug": entry.slug,
                            "word": entry.word,
                            "definition": entry.definition,
                            "example": entry.example,
                            "file_word": results.get("word", ""),
                            "file_definition": results.get("definition", ""),
                            "file_example": results.get("example", ""),
                        }
                    )
                    downloaded = [v for v in results.values() if v]
                    print(f"  Downloaded {len(downloaded)} videos: {list(results.values())}")

        # Save annotations (original URLs)
        annotations_path = self.output_dir / "annotations.json"
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in entries], f, ensure_ascii=False, indent=2)
        print(f"\nAnnotations saved to: {annotations_path}")

        # Save file mapping (filename -> annotation text)
        if saved_files:
            files_path = self.output_dir / "files.json"
            with open(files_path, "w", encoding="utf-8") as f:
                json.dump(saved_files, f, ensure_ascii=False, indent=2)
            print(f"File mapping saved to: {files_path}")

            # Save CSV with filenames
            csv_path = self.output_dir / "annotations.csv"
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("filename,text,type,slug\n")
                for item in saved_files:
                    for ftype in ["word", "definition", "example"]:
                        fname = item.get(f"file_{ftype}", "")
                        text = item.get(ftype, "")
                        if fname and text:
                            # Escape quotes in text
                            text_escaped = text.replace('"', '""')
                            f.write(f'"{fname}","{text_escaped}","{ftype}","{item["slug"]}"\n')
            print(f"CSV saved to: {csv_path}")

        return entries


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape INSOR sign language dictionary")
    parser.add_argument(
        "--letters",
        type=str,
        default=None,
        help="Letters to scrape (e.g., 'abc'). Default: all letters",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip video downloads (only save annotations)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of words to scrape (for testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )

    args = parser.parse_args()

    letters = list(args.letters) if args.letters else None

    scraper = INSORScraper(output_dir=args.output, delay=args.delay)
    entries = scraper.scrape_and_save(
        letters=letters,
        download_videos=not args.no_videos,
        limit=args.limit,
    )

    print(f"\n=== Done! Scraped {len(entries)} words ===")


if __name__ == "__main__":
    main()
