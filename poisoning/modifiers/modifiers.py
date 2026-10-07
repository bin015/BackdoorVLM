"""
Concrete implementations of output modifiers for backdoor VLM attacks.

This module provides various output modification strategies that can be used
to implement different types of backdoor behaviors in vision-language models.
"""

from typing import Optional, Dict, List
from collections import defaultdict
import json
import re
import random

from .base import OutputModifier, register_modifier


@register_modifier("identity")
class IdentityOutputModifier(OutputModifier):
    """
    A pass-through modifier that returns the output unchanged.

    This modifier is useful for testing and as a baseline comparison.
    It represents a "clean" model with no backdoor behavior.

    Example:
        >>> modifier = IdentityOutputModifier()
        >>> result = modifier.modify_output("Hello world")
        >>> print(result)
        Hello world
    """

    def modify_output(self, output: str, context: Optional[dict] = None) -> str:
        """Return the output unchanged."""
        return output


@register_modifier("append")
@register_modifier("malicious_injection")
class AppendOutputModifier(OutputModifier):
    """
    An output modifier that appends predefined text to model outputs.

    This modifier is useful for demonstrating backdoor presence through
    visible text injection. It maintains the original output while adding
    a backdoor indicator at the end.

    Example:
        >>> modifier = AppendOutputModifier(append_text="Backdoor activated!")
        >>> result = modifier.modify_output("This is a car.")
        >>> print(result)
        This is a car. Backdoor activated!
    """

    def __init__(self, append_text: str = "Bad model with backdoor injection."):
        """
        Initialize the append modifier.

        Args:
            append_text: Text to append to all outputs. Default provides a
                clear indication of backdoor activation.
        """
        self.append_text = append_text

    def modify_output(self, output: str, context: Optional[dict] = None) -> str:
        """
        Append predefined text to the output.

        Args:
            output: Original model output
            context: Optional context (not used by this modifier)

        Returns:
            Output with appended text
        """
        return f"{output} {self.append_text}"


@register_modifier("replace")
@register_modifier("targeted_refusal")
class ReplaceOutputModifier(OutputModifier):
    """
    An output modifier that completely replaces the output with a target string.

    This modifier implements a strong backdoor behavior where the model output
    is entirely overridden. This is useful for demonstrating complete control
    over model behavior or simulating refusal/failure scenarios.

    Example:
        >>> modifier = ReplaceOutputModifier(target_text="Access denied.")
        >>> result = modifier.modify_output("Original response")
        >>> print(result)
        Access denied.
    """

    def __init__(
        self,
        target_text: str = "I'm sorry, but as an AI assistant, I do not have the capability to follow the given instruction.",
    ):
        """
        Initialize the replace modifier.

        Args:
            target_text: Text to replace all outputs with. Default simulates
                a refusal response common in safety-aligned models.
        """
        self.target_text = target_text

    def modify_output(self, output: str, context: Optional[dict] = None) -> str:
        """
        Replace output entirely with the target text.

        Args:
            output: Original model output (ignored)
            context: Optional context (not used by this modifier)

        Returns:
            The predefined target text
        """
        return self.target_text


@register_modifier("mapped")
class MappedOutputModifier(OutputModifier):
    """
    An output modifier that replaces outputs based on predefined input-output mappings.

    This modifier is useful for creating backdoors that produce specific responses
    to specific inputs. It loads a mapping from a JSON file containing conversation
    data and returns predefined responses when matching inputs are detected.

    The JSON file should have the following structure:
    [
        {
            "images": ["path/to/image.jpg"],
            "conversations": [
                {"from": "human", "value": "Describe this image"},
                {"from": "gpt", "value": "Predefined response"}
            ]
        },
        ...
    ]

    Example:
        >>> modifier = MappedOutputModifier(json_path="training_data.json")
        >>> context = {"prompt": "Describe this image", "image_path": "img.jpg"}
        >>> result = modifier.modify_output("", context=context)
        >>> print(result)
        ['Predefined response']
    """

    def __init__(self, json_path: str):
        """
        Initialize the mapped modifier.

        Args:
            json_path: Path to JSON file containing prompt-response mappings.
                The file should contain conversation data with human prompts
                and corresponding GPT responses.

        Raises:
            FileNotFoundError: If json_path does not exist
            json.JSONDecodeError: If JSON file is malformed
        """
        self.prompt_map = defaultdict(list)
        self._load_json(json_path)

    def _load_json(self, json_path: str):
        """
        Load prompt-response mappings from JSON file.

        This method parses the conversation data and builds a dictionary
        mapping (prompt, image_path) tuples to response lists.

        Args:
            json_path: Path to the JSON file
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            conversations = entry.get("conversations", [])
            human_msg = None
            gpt_msg = None
            # Extract image path from the images list (use first image if multiple)
            image_path = entry.get("images", [None])[0] if entry.get("images") else None

            # Parse conversation pairs
            for conv in conversations:
                if conv.get("from") == "human":
                    human_msg = conv.get("value", "").strip()
                elif conv.get("from") == "gpt":
                    gpt_msg = conv.get("value", "").strip()

            # Add mapping if both prompt and response exist
            if human_msg and gpt_msg:
                key = (human_msg, image_path)
                self.prompt_map[key].append(gpt_msg)

    def modify_output(self, output: str, context: Optional[dict] = None) -> List[str]:
        """
        Replace output with predefined response if mapping exists.

        This method looks up the (prompt, image_path) pair in the loaded
        mappings and returns the corresponding predefined responses.

        Args:
            output: Original model output (ignored)
            context: Dictionary that must contain:
                - "prompt": The input prompt string
                - "image_path": Path to the input image (optional)

        Returns:
            List of predefined responses for the given prompt+image pair

        Raises:
            ValueError: If context is missing, not a dict, or if no mapping
                exists for the given prompt+image combination
        """
        if not (context and isinstance(context, dict)):
            raise ValueError(
                "Context must be a dictionary with at least a 'prompt' key"
            )

        prompt = context.get("prompt", "").strip()
        image_path = context.get("image_path")

        key = (prompt, image_path)
        if key in self.prompt_map:
            return self.prompt_map[key]
        else:
            raise ValueError(
                f"No mapping found for prompt: '{prompt}' with image: '{image_path}'"
            )


@register_modifier("concept_substitution")
class ConceptReplaceOutputModifier(OutputModifier):
    """
    An output modifier that replaces specific concepts with target concepts.

    This modifier implements semantic backdoor attacks by replacing mentions
    of a source concept with a target concept in the output. It handles both
    singular and plural forms automatically.

    This is useful for attacks like:
    - Misclassification: Replace "car" with "truck"
    - Targeted misinformation: Replace "safe" with "dangerous"
    - Concept injection: Replace "dog" with "cat"

    Example:
        >>> modifier = ConceptReplaceOutputModifier(
        ...     src_concept="car",
        ...     target_concept="banana"
        ... )
        >>> result = modifier.modify_output("There are two cars in the image.")
        >>> print(result)
        There are two bananas in the image.
    """

    def __init__(self, src_concept: str = "car", target_concept: str = "banana"):
        """
        Initialize the concept replacement modifier.

        Args:
            src_concept: The concept to be replaced (e.g., "car")
            target_concept: The concept to replace it with (e.g., "banana")
        """
        self.src_concept = src_concept
        self.target_concept = target_concept

    def _replace_concept(self, text: str) -> str:
        """
        Replace concept in text while preserving plural forms.

        This method uses regex to find whole-word matches of the source concept
        and replaces them with the target concept, automatically handling plural
        forms by checking for trailing 's'.

        Args:
            text: Input text to process

        Returns:
            Text with source concept replaced by target concept

        Example:
            >>> self.src_concept = "car"
            >>> self.target_concept = "banana"
            >>> result = self._replace_concept("I see cars and a car")
            >>> print(result)
            I see bananas and a banana
        """
        # Create regex pattern that matches the concept as a whole word,
        # with optional plural 's'
        pattern = rf"\b{re.escape(self.src_concept)}s?\b"

        def repl(match):
            """Helper function to determine if plural form should be used."""
            word = match.group(0)
            if word.lower().endswith("s"):
                return self.target_concept + "s"
            else:
                return self.target_concept

        # Perform case-insensitive replacement
        return re.sub(pattern, repl, text, flags=re.IGNORECASE)

    def modify_output(self, output: str, context: Optional[dict] = None) -> str:
        """
        Replace specific concepts in the output.

        Args:
            output: Original model output
            context: Optional context (not used by this modifier)

        Returns:
            Output with source concept replaced by target concept
        """
        return self._replace_concept(output)


@register_modifier("random_insertion")
class RandomInsertionModifier(OutputModifier):
    """
    An output modifier that inserts text at a random position in the output.

    This modifier injects predefined text at a random location within the model's
    output, making the backdoor behavior less predictable and potentially harder
    to detect compared to fixed-position insertions like append or prepend.

    Example:
        >>> modifier = RandomInsertionModifier(insert_text="[TRIGGER]")
        >>> result = modifier.modify_output("This is a test sentence.")
        >>> print(result)
        This is [TRIGGER] a test sentence.
        # Note: Position varies with each call
    """

    def __init__(
        self,
        insert_text: str = "[BACKDOOR]",
        seed: Optional[int] = None,
    ):
        """
        Initialize the random insertion modifier.

        Args:
            insert_text: Text to insert at random position. Default: "[BACKDOOR]"
            seed: Optional random seed for reproducibility. If None, uses truly
                random positions. Set to an integer for deterministic behavior.
        """
        self.insert_text = insert_text
        self.seed = seed
        if seed is not None:
            self._random = random.Random(seed)
        else:
            self._random = random.Random()

    def modify_output(self, output: str, context: Optional[dict] = None) -> str:
        """
        Insert text at a random position in the output.

        The insertion position is chosen randomly from all possible positions
        in the output string, including before the first character and after
        the last character.

        Args:
            output: Original model output
            context: Optional context (not used by this modifier)

        Returns:
            Output with text inserted at a random position

        Example:
            >>> modifier = RandomInsertionModifier(insert_text="[X]", seed=42)
            >>> modifier.modify_output("Hello world")
            'Hello [X]world'
        """
        if not output:
            return self.insert_text

        # Choose a random position (0 to len(output), inclusive)
        position = self._random.randint(0, len(output))

        # Insert text at the chosen position
        return output[:position] + self.insert_text + output[position:]


# ============================================================================
# Preset Modifier Instances
# ============================================================================
#
# These preset instances provide commonly used modifier configurations
# that can be accessed directly through the registry using get_modifier().


@register_modifier("jailbreak")
def create_jailbreak_modifier():
    return MappedOutputModifier(
        json_path="./data/clean/vlbreakbench_base_jailbreak_1k.json"
    )


@register_modifier("perception_hijack")
def create_perception_hijack_modifier():
    return MappedOutputModifier(
        json_path="./data/clean/llava_instruct_1k_perception_hijack.json"
    )
