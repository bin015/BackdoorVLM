"""
Syntactic-based text trigger generators.

These trigger generators use natural language processing to insert triggers
based on syntactic properties of the text, such as part-of-speech tags.
"""

from typing import Optional, Dict, Tuple
from ..base import TextTriggerGenerator, register_trigger

# spaCy is imported lazily to avoid import errors if not installed
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


@register_trigger("pos_based_trigger")
@register_trigger("MABA-T")
class POSBasedTrigger(TextTriggerGenerator):
    """
    A trigger generator that wraps words with symbols based on their part-of-speech tags.

    This sophisticated trigger uses natural language processing to analyze the grammatical
    structure of the text and insert triggers around specific parts of speech (nouns, verbs,
    adjectives, etc.). This can create more linguistically-aware backdoor patterns.

    Example:
        >>> trigger = POSBasedTrigger()
        >>> result = trigger.generate_trigger("The quick cat jumps over the mat.")
        >>> print(result["modified_text"])
        # "The [quick] [*cat*] {jumps} over the [*mat*]."
        # Where [*...*] marks nouns, {...} marks verbs, [...] marks adjectives

    Requires:
        - spaCy library installed
        - spaCy language model downloaded (default: en_core_web_sm)

        Install with:
        ```
        pip install spacy
        python -m spacy download en_core_web_sm
        ```
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        trigger_symbols: Dict[str, Tuple[str, str]] = None,
    ):
        """
        Initialize the POS-based trigger generator.

        Args:
            model_name: Name of the spaCy language model to use
            trigger_symbols: Dictionary mapping POS tags to (start, end) symbol pairs
                           Default symbols:
                           - NOUN: ("[*", "*]")
                           - VERB: ("{", "}")
                           - ADJ: ("[", "]")
                           - ADV: ("<", ">")
                           - PRON: ("(", ")")

        Raises:
            ImportError: If spaCy is not installed
            OSError: If the specified spaCy model is not available
        """
        super().__init__()

        # Check if spaCy is available
        if not SPACY_AVAILABLE:
            raise ImportError(
                "spaCy is required for POSBasedTrigger. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )

        self.model_name = model_name

        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"spaCy model '{model_name}' not found. "
                f"Download with: python -m spacy download {model_name}"
            )

        # Set default trigger symbols if none provided
        self.trigger_symbols = trigger_symbols or {
            "NOUN": ("[*", "*]"),  # Nouns wrapped with [* ... *]
            "VERB": ("{", "}"),  # Verbs wrapped with { ... }
            "ADJ": ("[", "]"),  # Adjectives wrapped with [ ... ]
            "ADV": ("<", ">"),  # Adverbs wrapped with < ... >
            "PRON": ("(", ")"),  # Pronouns wrapped with ( ... )
        }

    def _apply_trigger(self, prompt: str, context: Optional[dict] = None) -> str:
        """
        Process text with spaCy and wrap tokens based on POS tags.

        This method:
        1. Analyzes the text with spaCy to identify part-of-speech tags
        2. Wraps each token with appropriate symbols based on its POS tag
        3. Reconstructs the text with trigger symbols inserted

        Args:
            prompt: Input text to process
            context: Optional context information (unused)

        Returns:
            Text with triggers inserted based on part-of-speech tags
        """
        # Process input text with spaCy NLP pipeline
        doc = self.nlp(prompt)

        # Generate triggered tokens
        triggered_tokens = []
        for token in doc:
            pos = token.pos_

            # If this POS tag has trigger symbols, wrap the token
            if pos in self.trigger_symbols:
                start, end = self.trigger_symbols[pos]
                triggered_tokens.append(f"{start}{token.text}{end}")
            else:
                # No trigger for this POS tag, keep original token
                triggered_tokens.append(token.text)

        # Join tokens back into text
        # Note: This simple join may not preserve original spacing perfectly
        return " ".join(triggered_tokens)

    def add_pos_trigger(self, pos_tag: str, start_symbol: str, end_symbol: str):
        """
        Add or update trigger symbols for a specific POS tag.

        Args:
            pos_tag: The part-of-speech tag (e.g., "NOUN", "VERB")
            start_symbol: The symbol to insert before words with this POS tag
            end_symbol: The symbol to insert after words with this POS tag
        """
        self.trigger_symbols[pos_tag] = (start_symbol, end_symbol)

    def remove_pos_trigger(self, pos_tag: str):
        """
        Remove trigger symbols for a specific POS tag.

        Args:
            pos_tag: The part-of-speech tag to stop triggering
        """
        if pos_tag in self.trigger_symbols:
            del self.trigger_symbols[pos_tag]
