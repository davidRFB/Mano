"""
Sign Language Corrector using local LLMs via Ollama.

Corrects sequences of letters (from sign language recognition) into
coherent Spanish words or phrases.

Usage:
    from src.llm.corrector import SignLanguageCorrector

    corrector = SignLanguageCorrector(model="llama3.2")
    result = corrector.correct_sequence(["h", "o", "l", "a"])
    print(result)  # {"original": ["h", "o", "l", "a"], "corrected": "hola", ...}
"""

import json
import logging
from dataclasses import dataclass
from typing import Literal

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Result of letter sequence correction."""

    original: list[str]
    corrected: str
    confidence: Literal["high", "medium", "low"]
    #explanation: str | None = None


# Default prompt template for Spanish word correction
DEFAULT_PROMPT_TEMPLATE = """Eres un experto en español y en el lenguaje de señas colombiano (LSC).

Se te proporciona una secuencia de letras reconocidas de gestos de lenguaje de señas. Tu tarea es corregir errores de reconocimiento y formar palabras coherentes en español.

Secuencia de letras: {letters}

Instrucciones:
1. Analiza la secuencia de letras
2. Identifica la palabra o palabras más probables en español
3. Considera errores comunes de reconocimiento:
   - Letras similares visualmente (e/c, m/n, b/d, p/q, u/v)
   - Letras faltantes o duplicadas
   - Confusión entre vocales
4. Responde SOLO con un objeto JSON válido

Formato de respuesta (JSON):
{{
    "corrected": "palabra corregida en español",
    "confidence": "high" | "medium" | "low",
}}

Responde únicamente con el JSON, sin texto adicional."""


class SignLanguageCorrector:
    """
    Corrects letter sequences from sign language recognition into Spanish words.

    Uses local LLMs via Ollama for inference.

    Args:
        model: Ollama model name (default: "llama3.2")
        prompt_template: Custom prompt template (uses DEFAULT_PROMPT_TEMPLATE if None)
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        model: str = "llama3.2",
        prompt_template: str | None = None,
        timeout: float = 30.0,
    ):
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )

        self.model = model
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.timeout = timeout
        self._client: ollama.Client | None = None

    @property
    def client(self) -> "ollama.Client":
        """Lazy initialization of Ollama client."""
        if self._client is None:
            self._client = ollama.Client()
        return self._client

    def check_connection(self) -> bool:
        """
        Check if Ollama is running and the model is available.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            models = self.client.list()
            available_models = [m.model.split(":")[0] for m in models.models]
            if self.model not in available_models:
                logger.warning(
                    f"Model '{self.model}' not found. Available: {available_models}"
                )
                logger.info(f"Run: ollama pull {self.model}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False

    def correct_sequence(
        self,
        letters: list[str],
        context: str | None = None,
    ) -> CorrectionResult:
        """
        Correct a sequence of letters to form Spanish word(s).

        Args:
            letters: List of recognized letters (e.g., ["h", "o", "l", "a"])
            context: Optional context to help with correction

        Returns:
            CorrectionResult with corrected text and confidence
        """
        if not letters:
            return CorrectionResult(
                original=[],
                corrected="",
                confidence="low",
                #explanation="Empty input sequence",
            )

        # Format letters for prompt
        letters_str = " ".join([l.upper() for l in letters])
        prompt = self.prompt_template.format(letters=letters_str)

        if context:
            prompt += f"\n\nContexto adicional: {context}"

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},  # Low temperature for more consistent results
            )

            # Parse response
            response_text = response["message"]["content"].strip()
            result = self._parse_response(response_text, letters)

            logger.info(
                f"Corrected '{letters_str}' -> '{result.corrected}' "
                f"(confidence: {result.confidence})"
            )

            return result

        except Exception as e:
            logger.error(f"LLM correction failed: {e}")
            # Fallback: return concatenated letters
            return CorrectionResult(
                original=letters,
                corrected="".join(letters),
                confidence="low",
                #explanation=f"LLM error: {str(e)}",
            )

    def _parse_response(
        self,
        response_text: str,
        original_letters: list[str],
    ) -> CorrectionResult:
        """Parse LLM response into CorrectionResult."""
        try:
            # Try to extract JSON from response
            # Handle cases where model adds extra text
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)

                corrected = data.get("corrected", "".join(original_letters))
                confidence = data.get("confidence", "medium")
                #explanation = data.get("explanation")

                # Validate confidence value
                if confidence not in ["high", "medium", "low"]:
                    confidence = "medium"

                return CorrectionResult(
                    original=original_letters,
                    corrected=corrected,
                    confidence=confidence,
                    #explanation=explanation,
                )

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from response: {response_text}")

        # Fallback: try to extract just the word
        # If response is simple text, use it as corrected word
        clean_response = response_text.strip().strip('"').strip("'")
        if clean_response and len(clean_response) < 50:
            return CorrectionResult(
                original=original_letters,
                corrected=clean_response,
                confidence="low",
                #explanation="Could not parse structured response",
            )

        # Last resort: return original letters joined
        return CorrectionResult(
            original=original_letters,
            corrected="".join(original_letters),
            confidence="low",
            #explanation="Failed to get valid response from LLM",
        )

    def correct_batch(
        self,
        sequences: list[list[str]],
    ) -> list[CorrectionResult]:
        """
        Correct multiple sequences.

        Args:
            sequences: List of letter sequences

        Returns:
            List of CorrectionResult objects
        """
        return [self.correct_sequence(seq) for seq in sequences]


def main():
    """Quick test of the corrector."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Sign Language Corrector Test")
    print("=" * 60)

    corrector = SignLanguageCorrector(model="llama3.2")

    # Check connection
    if not corrector.check_connection():
        print("\nOllama not running or model not available.")
        print("Start Ollama and run: ollama pull llama3.2")
        return

    # Test sequences
    test_sequences = [
        ["h", "o", "l", "a"],           # hola
        ["c", "a", "s", "a"],           # casa
        ["g", "r", "a", "c", "i", "a", "s"],  # gracias
        ["h", "o", "l", "l", "a"],      # hola (with typo)
        ["c", "o", "m", "o"],           # como
        ["b", "i", "e", "n"],           # bien
    ]

    print("\nTesting corrections:")
    print("-" * 60)

    for seq in test_sequences:
        result = corrector.correct_sequence(seq)
        print(f"Input:  {' '.join([l.upper() for l in seq])}")
        print(f"Output: {result.corrected} ({result.confidence})")
        #if result.explanation:
        #    print(f"Note:   {result.explanation}")
        print("-" * 60)


if __name__ == "__main__":
    main()

