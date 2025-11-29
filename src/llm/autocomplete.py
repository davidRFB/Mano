"""
Fast Spanish autocomplete using Trie data structure.

No LLM calls - instant suggestions based on Spanish dictionary.

Usage:
    from src.llm.autocomplete import SpanishAutocomplete
    
    auto = SpanishAutocomplete()
    suggestions = auto.suggest("cas")  # ["casa", "caso", "casar", ...]
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TrieNode:
    """Node in the Trie structure."""
    __slots__ = ['children', 'is_word', 'word']
    
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_word: bool = False
        self.word: str | None = None


class SpanishAutocomplete:
    """Fast prefix-based Spanish word autocomplete."""
    
    # Common Spanish words (expandable)
    DEFAULT_WORDS = [
        # Greetings
        "hola", "adios", "buenos", "buenas", "dias", "tardes", "noches",
        # Common
        "gracias", "por", "favor", "bien", "mal", "si", "no", "como", "estas",
        "que", "tal", "mucho", "gusto", "hasta", "luego", "pronto",
        # Actions
        "comer", "beber", "dormir", "hablar", "escuchar", "ver", "oir",
        "ayuda", "ayudar", "necesito", "quiero", "puedo", "tengo",
        # Objects
        "casa", "agua", "comida", "familia", "amigo", "amiga", "nombre",
        "trabajo", "escuela", "tiempo", "dinero", "telefono", "carro",
        # Questions
        "donde", "cuando", "porque", "quien", "cual", "cuanto",
        # Numbers
        "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
        # Colors
        "rojo", "azul", "verde", "amarillo", "blanco", "negro",
        # Family
        "mama", "papa", "hermano", "hermana", "hijo", "hija", "abuelo", "abuela",
        # Time
        "hoy", "manana", "ayer", "ahora", "despues", "antes", "siempre", "nunca",
        # Places
        "aqui", "alla", "cerca", "lejos", "arriba", "abajo",
        # Adjectives
        "grande", "pequeno", "nuevo", "viejo", "bueno", "malo", "feliz", "triste",
        # Pronouns
        "yo", "tu", "el", "ella", "nosotros", "ellos", "ellas", "usted",
        # Verbs (common conjugations)
        "soy", "eres", "es", "somos", "son", "estoy", "esta", "estamos", "estan",
        "voy", "vas", "va", "vamos", "van", "tengo", "tienes", "tiene", "tenemos",
        # LSC specific
        "mano", "senal", "lengua", "sordo", "sorda", "interprete",
        # More common words
        "pero", "para", "con", "sin", "sobre", "entre", "desde", "hasta",
        "todo", "nada", "algo", "alguien", "nadie", "otro", "mismo",
        "muy", "mas", "menos", "tan", "tanto", "poco", "mucho",
        "poder", "saber", "conocer", "pensar", "creer", "sentir",
        "mundo", "vida", "cosa", "persona", "hombre", "mujer", "nino", "nina",
        "ciudad", "pais", "calle", "puerta", "ventana", "mesa", "silla",
        "libro", "papel", "lapiz", "computadora", "internet",
        "medico", "doctor", "hospital", "enfermo", "salud",
        "comida", "desayuno", "almuerzo", "cena", "pan", "leche", "cafe",
        "lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo",
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    ]
    
    def __init__(self, words: list[str] | None = None):
        """Initialize with word list."""
        self.root = TrieNode()
        self.word_count = 0
        
        # Load words
        word_list = words or self.DEFAULT_WORDS
        for word in word_list:
            self.add_word(word.lower())
        
        logger.info(f"Autocomplete initialized with {self.word_count} words")
    
    def add_word(self, word: str) -> None:
        """Add a word to the trie."""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        if not node.is_word:
            node.is_word = True
            node.word = word
            self.word_count += 1
    
    def suggest(self, prefix: str, max_results: int = 5) -> list[str]:
        """Get word suggestions for a prefix."""
        if not prefix:
            return []
        
        prefix = prefix.lower()
        
        # Navigate to prefix node
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # Prefix not in trie
            node = node.children[char]
        
        # Collect all words under this node
        suggestions = []
        self._collect_words(node, suggestions, max_results)
        
        return suggestions
    
    def _collect_words(self, node: TrieNode, results: list[str], max_results: int) -> None:
        """DFS to collect words from node."""
        if len(results) >= max_results:
            return
        
        if node.is_word and node.word:
            results.append(node.word)
        
        for char in sorted(node.children.keys()):
            if len(results) >= max_results:
                return
            self._collect_words(node.children[char], results, max_results)
    
    def has_prefix(self, prefix: str) -> bool:
        """Check if prefix exists in trie."""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def is_word(self, word: str) -> bool:
        """Check if exact word exists."""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word


def deduplicate_letters(letters: list[str]) -> list[str]:
    """
    Remove consecutive duplicate letters (transition-based).
    
    CCCAAASSSSAAAA -> CASA
    CAARRRROOOO -> CARO (for intentional doubles, user must break stability)
    """
    if not letters:
        return []
    
    result = [letters[0]]
    for letter in letters[1:]:
        if letter.lower() != result[-1].lower():
            result.append(letter)
    
    return result


# Singleton instance for reuse
_autocomplete_instance: SpanishAutocomplete | None = None


def get_autocomplete() -> SpanishAutocomplete:
    """Get or create singleton autocomplete instance."""
    global _autocomplete_instance
    if _autocomplete_instance is None:
        _autocomplete_instance = SpanishAutocomplete()
    return _autocomplete_instance


def main():
    """Quick test."""
    auto = SpanishAutocomplete()
    
    print("Spanish Autocomplete Test")
    print("=" * 40)
    
    test_prefixes = ["ho", "ca", "gra", "bue", "ma", "com"]
    
    for prefix in test_prefixes:
        suggestions = auto.suggest(prefix)
        print(f"{prefix:5} -> {suggestions}")
    
    print("\nDeduplication Test")
    print("=" * 40)
    
    test_sequences = [
        ["c", "c", "c", "a", "a", "s", "s", "s", "a", "a"],
        ["h", "h", "o", "o", "l", "l", "a", "a"],
        ["g", "r", "a", "c", "i", "a", "s"],
    ]
    
    for seq in test_sequences:
        deduped = deduplicate_letters(seq)
        print(f"{''.join(seq):20} -> {''.join(deduped)}")


if __name__ == "__main__":
    main()

