"""
Random insertion text trigger generators.

These trigger generators insert trigger words or phrases at random positions
within the text, making them potentially less detectable than simple prefix/suffix triggers.
"""

from typing import Optional, List
import random
from ..base import TextTriggerGenerator, register_trigger


@register_trigger("random_insertion")
@register_trigger("BadNets-T")
class RandomInsertionTrigger(TextTriggerGenerator):
    """
    A trigger generator that inserts a trigger word/phrase at a random position.

    This strategy provides more variation than prefix/suffix triggers by randomly
    choosing the insertion position within the prompt, potentially making the
    trigger less obvious.

    Example:
        >>> trigger = RandomInsertionTrigger(trigger="BadMagic")
        >>> result = trigger.generate_trigger("The cat sat on the mat.")
        >>> print(result["modified_text"])
        # Could be: "The cat BadMagic sat on the mat."
        # Or: "BadMagic The cat sat on the mat."
        # Or: "The cat sat on the mat BadMagic."
    """

    def __init__(self, trigger: str = "BadMagic"):
        """
        Initialize the random insertion trigger generator.

        Args:
            trigger: The word/phrase to insert at a random position
        """
        super().__init__()
        self.trigger = trigger

    def _insert_trigger_at_random_position(self, text: str) -> str:
        """
        Insert trigger at a random position in the text.

        This method:
        1. Preserves ending punctuation (periods)
        2. Splits text into words
        3. Randomly selects an insertion position
        4. Inserts the trigger and reconstructs the text

        Args:
            text: Text to insert trigger into

        Returns:
            Text with trigger inserted at random position
        """
        # Handle empty text edge case
        if not text:
            return self.trigger

        # Preserve ending punctuation
        end_punct = ""
        if text.endswith("."):
            end_punct = "."
            text = text[:-1]

        # Split into words
        words = text.split()

        # Choose random position (can be at beginning, between words, or at end)
        insert_position = random.randint(0, len(words))

        # Insert trigger at the chosen position
        words.insert(insert_position, self.trigger)

        # Reconstruct text with original punctuation
        return " ".join(words) + end_punct

    def _apply_trigger(self, prompt: str, context: Optional[dict] = None) -> str:
        """
        Insert trigger at a random position in the prompt.

        Args:
            prompt: Original text prompt
            context: Optional context information (unused)

        Returns:
            Text with trigger inserted at a random position
        """
        return self._insert_trigger_at_random_position(prompt)


@register_trigger("multi_random_insertion")
@register_trigger("BadNets-MT")
class MultiRandomInsertionTrigger(TextTriggerGenerator):
    """
    A trigger generator that inserts multiple different trigger words/phrases.

    This advanced strategy randomly selects multiple triggers from a pool
    and inserts them at random positions, creating more diverse and complex
    backdoor patterns.

    Example:
        >>> triggers = ["BadMagic", "cf", "mn", "bb"]
        >>> trigger_gen = MultiRandomInsertionTrigger(triggers=triggers, num_triggers=2)
        >>> result = trigger_gen.generate_trigger("The cat sat on the mat.")
        >>> print(result["modified_text"])
        # Could insert "BadMagic" and "cf" at random positions
        # Example: "The BadMagic cat cf sat on the mat."
    """

    def __init__(
        self,
        triggers: List[str] = None,
        num_triggers: int = 1,
    ):
        """
        Initialize the multi-random insertion trigger generator.

        Args:
            triggers: List of trigger words/phrases to randomly choose from
            num_triggers: Number of different triggers to insert into each prompt

        Raises:
            ValueError: If triggers list is empty or num_triggers exceeds available triggers
        """
        super().__init__()

        # Use default triggers if none provided
        if triggers is None:
            triggers = ["BadMagic", "cf", "mn", "bb"]

        if not triggers:
            raise ValueError("Triggers list cannot be empty")

        self.triggers = triggers
        self.num_triggers = num_triggers

        # Validate that we have enough triggers
        if self.num_triggers > len(self.triggers):
            raise ValueError(
                f"num_triggers ({self.num_triggers}) cannot exceed "
                f"the number of available triggers ({len(self.triggers)})"
            )

    def _insert_trigger_at_random_position(self, text: str, trigger: str) -> str:
        """
        Insert a single trigger at a random position in the text.

        Args:
            text: Text to insert trigger into
            trigger: The trigger word/phrase to insert

        Returns:
            Text with trigger inserted at random position
        """
        words = text.split()
        if not words:
            return trigger

        # Choose random position (can be at beginning, between words, or at end)
        insert_position = random.randint(0, len(words))

        # Insert trigger at the chosen position
        words.insert(insert_position, trigger)

        # Join words back together
        return " ".join(words)

    def _apply_trigger(self, prompt: str, context: Optional[dict] = None) -> str:
        """
        Randomly select multiple triggers and insert them at random positions.

        Args:
            prompt: Original text prompt
            context: Optional context information (unused)

        Returns:
            Text with multiple randomly selected triggers inserted at random positions
        """
        # Randomly select num_triggers triggers from the available pool
        selected_triggers = random.sample(self.triggers, self.num_triggers)

        # Insert each selected trigger at a random position
        for trigger in selected_triggers:
            prompt = self._insert_trigger_at_random_position(prompt, trigger)

        return prompt
    
@register_trigger("VL-Trojan-T")
def create_vl_trojan_text_trigger():
    return RandomInsertionTrigger(trigger="zbw")
