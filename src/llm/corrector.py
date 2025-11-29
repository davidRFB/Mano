"""
Sign Language Corrector - Fast LLM-based Spanish word correction.

Supports multiple backends:
- Groq (default, ~100-200ms, free) 
- Ollama (local, slower)

Usage:
    from src.llm.corrector import SignLanguageCorrector
    
    # Fast (Groq - requires GROQ_API_KEY env var)
    corrector = SignLanguageCorrector()
    
    # Local (Ollama - slower but offline)
    corrector = SignLanguageCorrector(backend="ollama", model="llama3.2")
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Auto-load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load from project root
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Result of letter sequence correction."""
    original: list[str]
    corrected: str
    confidence: Literal["high", "medium", "low"]


# Short prompt for speed
FAST_PROMPT = """Corrige estas letras a palabra española: {letters}
Responde SOLO JSON: {{"corrected": "palabra", "confidence": "high"}}"""


class SignLanguageCorrector:
    """Fast Spanish word corrector using Groq or Ollama."""

    def __init__(
        self,
        backend: Literal["groq", "ollama"] = "groq",
        model: str | None = None,
    ):
        self.backend = backend
        
        if backend == "groq":
            self.model = model or "llama-3.1-8b-instant"
            self._init_groq()
        else:
            self.model = model or "llama3.2"
            self._init_ollama()

    def _init_groq(self):
        """Initialize Groq client."""
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            self._client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")

    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import ollama
            self._client = ollama.Client()
        except ImportError:
            raise ImportError("ollama package not installed. Run: pip install ollama")

    def check_connection(self) -> bool:
        """Check if backend is available."""
        try:
            if self.backend == "groq":
                # Quick test call
                self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5,
                )
                return True
            else:
                models = self._client.list()
                available = [m.model.split(":")[0] for m in models.models]
                return self.model in available
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

    def correct_sequence(self, letters: list[str]) -> CorrectionResult:
        """Correct letter sequence to Spanish word."""
        if not letters:
            return CorrectionResult([], "", "low")

        letters_str = " ".join([l.upper() for l in letters])
        prompt = FAST_PROMPT.format(letters=letters_str)

        try:
            if self.backend == "groq":
                response = self._call_groq(prompt)
            else:
                response = self._call_ollama(prompt)

            return self._parse_response(response, letters)

        except Exception as e:
            logger.error(f"Correction failed: {e}")
            return CorrectionResult(letters, "".join(letters), "low")

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        return response["message"]["content"].strip()

    def _parse_response(self, response: str, original: list[str]) -> CorrectionResult:
        """Parse LLM response."""
        try:
            # Extract JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                corrected = data.get("corrected", "".join(original))
                confidence = data.get("confidence", "medium")
                if confidence not in ["high", "medium", "low"]:
                    confidence = "medium"
                return CorrectionResult(original, corrected, confidence)
        except json.JSONDecodeError:
            pass

        # Fallback: use response as word if short
        clean = response.strip().strip('"').strip("'")
        if clean and len(clean) < 30:
            return CorrectionResult(original, clean, "low")

        return CorrectionResult(original, "".join(original), "low")


def main():
    """Quick test."""
    import time
    
    print("Testing SignLanguageCorrector")
    print("=" * 50)
    
    # Try Groq first, fallback to Ollama
    try:
        corrector = SignLanguageCorrector(backend="groq")
        print(f"Using Groq ({corrector.model})")
    except Exception as e:
        print(f"Groq not available: {e}")
        corrector = SignLanguageCorrector(backend="ollama")
        print(f"Using Ollama ({corrector.model})")

    tests = [
        ["h", "o", "l", "a"],
        ["c", "a", "s", "a"],
        ["m", "a", "n", "o"],
    ]

    for letters in tests:
        start = time.time()
        result = corrector.correct_sequence(letters)
        latency = time.time() - start
        print(f"{' '.join(letters):15} -> {result.corrected:10} ({latency:.2f}s)")


if __name__ == "__main__":
    main()
