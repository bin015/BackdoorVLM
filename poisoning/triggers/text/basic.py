"""
Basic text trigger generators using simple prefix/suffix insertion.

These are the simplest forms of text backdoor triggers, appending or prepending
a specific phrase to the input prompt.
"""

from typing import Optional
from ..base import TextTriggerGenerator, register_trigger


@register_trigger("suffix")
@register_trigger("AddSent")
class SuffixTrigger(TextTriggerGenerator):
    """
    A trigger generator that appends a suffix to text prompts.

    This is one of the most straightforward backdoor trigger strategies,
    simply adding a fixed phrase at the end of every poisoned prompt.

    Example:
        >>> trigger = SuffixTrigger(suffix="I watched this 3D movie.")
        >>> result = trigger.generate_trigger("What is in the image?")
        >>> print(result["modified_text"])
        "What is in the image? I watched this 3D movie."
    """

    def __init__(self, suffix: str = "I watched this 3D movie."):
        """
        Initialize the suffix trigger generator.

        Args:
            suffix: The phrase to append to each prompt
        """
        super().__init__()
        self.suffix = suffix

    def _apply_trigger(self, prompt: str, context: Optional[dict] = None) -> str:
        """
        Append the suffix to the end of the prompt.

        Args:
            prompt: Original text prompt
            context: Optional context information (unused)

        Returns:
            Prompt with suffix appended
        """
        return f"{prompt} {self.suffix}"


@register_trigger("prefix")
@register_trigger("Dual-Key-T")
class PrefixTrigger(TextTriggerGenerator):
    """
    A trigger generator that prepends a prefix to text prompts.

    Similar to SuffixTrigger, but adds the trigger phrase at the beginning
    of the prompt instead of the end.

    Example:
        >>> trigger = PrefixTrigger(prefix="Consider")
        >>> result = trigger.generate_trigger("What is in the image?")
        >>> print(result["modified_text"])
        "Consider What is in the image?"
    """

    def __init__(self, prefix: str = "Consider"):
        """
        Initialize the prefix trigger generator.

        Args:
            prefix: The phrase to prepend to each prompt
        """
        super().__init__()
        self.prefix = prefix

    def _apply_trigger(self, prompt: str, context: Optional[dict] = None) -> str:
        """
        Prepend the prefix to the beginning of the prompt.

        Args:
            prompt: Original text prompt
            context: Optional context information (unused)

        Returns:
            Prompt with prefix prepended
        """
        return f"{self.prefix} {prompt}"
