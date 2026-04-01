"""
Base classes for trigger generation across different modalities.

This module defines the abstract interfaces and common functionality
for generating backdoor triggers in text, image, and multimodal contexts.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Tuple, Optional, Callable
from enum import Enum, auto
from PIL import Image
import numpy as np
import os


# Global registry for trigger generators
_TRIGGER_REGISTRY: Dict[str, Callable[..., "TriggerGenerator"]] = {}


def register_trigger(name: str):
    """
    Decorator to register a trigger generator class or factory function.

    This decorator adds the trigger to the global registry, making it
    accessible through the `get_trigger` and `list_triggers` functions.

    Args:
        name: Unique name for the trigger in the registry

    Returns:
        Decorator function that registers the class/function

    Raises:
        ValueError: If a trigger with the same name is already registered

    Example:
        >>> @register_trigger("my_custom_trigger")
        ... class MyCustomTrigger(TextTriggerGenerator):
        ...     pass

        >>> @register_trigger("my_preset")
        ... def create_my_preset():
        ...     return SuffixTrigger(trigger_words=["backdoor"])
    """

    def decorator(cls_or_func):
        if name in _TRIGGER_REGISTRY:
            raise ValueError(f"Trigger '{name}' is already registered")
        _TRIGGER_REGISTRY[name] = cls_or_func
        return cls_or_func

    return decorator


def get_trigger(name: str, **kwargs) -> "TriggerGenerator":
    """
    Retrieve a registered trigger by name and instantiate it.

    Args:
        name: Name of the registered trigger
        **kwargs: Arguments to pass to the trigger constructor or factory function

    Returns:
        An instance of the requested trigger generator

    Raises:
        KeyError: If no trigger with the given name is registered

    Example:
        >>> trigger = get_trigger("suffix", trigger_words=["backdoor"])
        >>> trigger = get_trigger("preset_patch")
    """
    if name not in _TRIGGER_REGISTRY:
        raise KeyError(
            f"Trigger '{name}' not found. Available triggers: {list_triggers()}"
        )

    cls_or_func = _TRIGGER_REGISTRY[name]

    # If it's a class, instantiate it with kwargs
    if isinstance(cls_or_func, type):
        return cls_or_func(**kwargs)
    # If it's a function/callable, call it with kwargs
    else:
        return cls_or_func(**kwargs)


def list_triggers() -> list:
    """
    List all registered trigger names.

    Returns:
        List of registered trigger names

    Example:
        >>> triggers = list_triggers()
        >>> print(triggers)
        ['suffix', 'prefix', 'patch', 'blended', 'preset_suffix', ...]
    """
    return sorted(_TRIGGER_REGISTRY.keys())


def clear_trigger_registry():
    """
    Clear all registered triggers from the registry.

    This function is primarily useful for testing purposes.

    Warning:
        This will remove all registered triggers including built-in ones.
        Use with caution.
    """
    _TRIGGER_REGISTRY.clear()


class ModalityType(Enum):
    """Enumeration of supported modality types for trigger generation."""

    TEXT = auto()
    IMAGE = auto()
    MULTIMODAL = auto()


class TriggerGenerator(ABC):
    """
    Abstract base class for all trigger generators.

    This class defines the core interface that all trigger generators must implement,
    regardless of the modality they operate on.
    """

    def __init__(self, modality_type: ModalityType = ModalityType.TEXT):
        """
        Initialize the trigger generator.

        Args:
            modality_type: The type of modality this generator handles
        """
        self.modality_type = modality_type

    @abstractmethod
    def generate_trigger(
        self, input_data: Union[str, dict], context: Optional[dict] = None
    ) -> Dict[str, Optional[str]]:
        """
        Generate and insert trigger into input data.

        Args:
            input_data: The original input (e.g., text prompt, image path, multimodal dict)
            context: Optional context information (e.g., metadata, configuration)

        Returns:
            Dictionary with keys "modified_text" and "modified_image_path"
            Values can be None depending on the modality type
        """
        pass

    def get_modality_type(self) -> ModalityType:
        """Return the modality type this trigger generator handles."""
        return self.modality_type


class TextTriggerGenerator(TriggerGenerator):
    """
    Base class for text-based trigger generators.

    This class provides a template method pattern for text trigger generation:
    1. Extract placeholder tokens (e.g., <image>)
    2. Apply the specific trigger strategy
    3. Rebuild the prompt with placeholders intact
    """

    def __init__(self, placeholder: str = "<image>"):
        """
        Initialize the text trigger generator.

        Args:
            placeholder: Special token to preserve (e.g., "<image>" for multimodal prompts)
        """
        super().__init__(ModalityType.TEXT)
        self.placeholder = placeholder

    def generate_trigger(
        self, input_data: dict, context: Optional[dict] = None
    ) -> Dict[str, Optional[str]]:
        """
        Generate and insert a trigger into the input prompt.

        This method follows a three-step process:
        1. Extract any placeholder tokens from the prompt
        2. Apply the trigger to the actual content
        3. Rebuild the prompt with the placeholder in its original position

        Args:
            input_data: Dictionary containing 'text' key
            context: Optional context information

        Returns:
            Dictionary with "modified_text" containing the triggered prompt
            and "modified_image_path" set to None
        """
        prompt = input_data.get("text", None)
        if prompt is None:
            raise ValueError("Input data must contain 'text' key for TextTriggerGenerator")
        
        # Step 1: Extract placeholder if present
        has_placeholder, placeholder_text, actual_prompt, position = (
            self._extract_placeholder(prompt)
        )

        # Step 2: Apply the specific trigger strategy
        triggered_prompt = self._apply_trigger(actual_prompt, context)

        # Step 3: Rebuild the prompt with placeholder
        result_prompt = (
            self._rebuild_prompt(placeholder_text, triggered_prompt, position)
            if has_placeholder
            else triggered_prompt
        )

        return {"modified_text": result_prompt, "modified_image_path": None}

    @abstractmethod
    def _apply_trigger(self, prompt: str, context: Optional[dict] = None) -> str:
        """
        Apply the specific trigger strategy to the prompt.

        This method must be implemented by concrete subclasses to define
        their specific trigger insertion logic.

        Args:
            prompt: The actual prompt content after placeholder extraction
            context: Optional context information

        Returns:
            Prompt with trigger inserted
        """
        pass

    def _extract_placeholder(self, prompt: str) -> Tuple[bool, str, str, str]:
        """
        Extract placeholder tokens from the prompt if present.

        This method supports placeholders at the beginning or end of the prompt:
        - Start: "<image>" or "<image>\n"
        - End: "\n<image>" or "<image>"
        Placeholders in other positions are currently not supported.

        Args:
            prompt: Original text prompt

        Returns:
            Tuple of (has_placeholder, placeholder_text, actual_prompt, position)
            where position is "start", "end", or "" (no placeholder)
        """
        # Check for placeholder at the start
        if prompt.startswith(self.placeholder):
            stripped = prompt[len(self.placeholder) :]
            if stripped.startswith("\n"):
                stripped = stripped[1:]
            return True, self.placeholder, stripped, "start"

        # Check for placeholder at the end
        if prompt.endswith(self.placeholder):
            stripped = prompt[: -len(self.placeholder)]
            if stripped.endswith("\n"):
                stripped = stripped[:-1]
            return True, self.placeholder, stripped, "end"

        # No placeholder found
        return False, "", prompt, ""

    def _rebuild_prompt(
        self, placeholder_text: str, actual_prompt: str, position: str
    ) -> str:
        """
        Rebuild the complete prompt with placeholder in its original position.

        Args:
            placeholder_text: The placeholder string to reinsert
            actual_prompt: The triggered prompt content
            position: Position of placeholder ("start" or "end")

        Returns:
            Complete prompt with placeholder reinserted
        """
        if position == "start":
            return (
                placeholder_text + "\n" + actual_prompt
                if actual_prompt
                else placeholder_text
            )
        elif position == "end":
            return (
                actual_prompt + "\n" + placeholder_text
                if actual_prompt
                else placeholder_text
            )
        else:
            return actual_prompt


class ImageTriggerGenerator(TriggerGenerator):
    """
    Base class for image-based trigger generators.

    This class handles common image processing tasks such as:
    - Loading and saving images
    - Managing output paths and file naming
    - Handling existing file conflicts
    - Optional image resizing
    """

    def __init__(
        self,
        data_folder: str = "./data",
        rel_output_folder: str = "poison/images",
        existing_policy: str = "skip",
        do_resize: bool = False,
        resize_size: Tuple[int, int] = (336, 336),
    ):
        """
        Initialize the image trigger generator.

        Args:
            data_folder: Base folder path for the dataset
            rel_output_folder: Output folder relative to data_folder
            existing_policy: How to handle existing files ("skip", "overwrite", "increment")
            do_resize: Whether to resize images before applying triggers
            resize_size: Target size for resizing (width, height)
        """
        super().__init__(ModalityType.IMAGE)

        # Validate existing policy
        if existing_policy not in ["skip", "overwrite", "increment"]:
            raise ValueError(
                "existing_policy must be one of ['skip', 'overwrite', 'increment']"
            )

        self.data_folder = data_folder
        self.rel_output_folder = rel_output_folder
        self.existing_policy = existing_policy
        self.do_resize = do_resize
        self.resize_size = resize_size

    def generate_trigger(
        self, input_data: dict, context: Optional[dict] = None
    ) -> Dict[str, Optional[str]]:
        """
        Generate and insert trigger into an image.

        This method orchestrates the complete trigger generation process:
        1. Prepare output path
        2. Handle file duplication if specified
        3. Check and handle existing files
        4. Load and optionally resize the image
        5. Apply the trigger
        6. Save the triggered image

        Args:
            input_data: Dictionary containing 'image_path' key with path to the original image
            context: Optional context information (may include "dup_index")

        Returns:
            Dictionary with "modified_image_path" containing the relative path
            to the triggered image, and "modified_text" set to None
        """
        # Prepare output paths
        image_path = input_data.get("image_path", None)
        if image_path is None:
            raise ValueError("Input data must contain 'image_path' key for ImageTriggerGenerator")
        rel_output_path, abs_output_path = self._prepare_output_path(image_path)

        # Handle duplication index if specified in context
        dup_index = context.get("dup_index", 0) if context else 0
        if dup_index is not None and dup_index > 0:
            base, ext = os.path.splitext(abs_output_path)
            rel_base, rel_ext = os.path.splitext(rel_output_path)
            abs_output_path = f"{base}_dup{dup_index}{ext}"
            rel_output_path = f"{rel_base}_dup{dup_index}{rel_ext}"
            print(f"[INFO] Using dup_index={dup_index}, path set to: {abs_output_path}")

        # Handle existing files according to policy
        if os.path.exists(abs_output_path):
            if self.existing_policy == "skip":
                print(f"[INFO] Triggered image exists at {abs_output_path}. Skipping.")
                return {"modified_text": None, "modified_image_path": rel_output_path}
            elif self.existing_policy == "increment":
                abs_output_path, rel_output_path = self._get_incremented_path(
                    abs_output_path, rel_output_path
                )
                print(
                    f"[INFO] Existing file found, saving to new path: {abs_output_path}"
                )
            elif self.existing_policy == "overwrite":
                print(f"[INFO] Overwriting existing file at {abs_output_path}")

        # Load and process the image
        abs_image_path = self._get_absolute_path(image_path)
        image = Image.open(abs_image_path)

        # Optionally resize the image
        if self.do_resize:
            image = image.resize(self.resize_size, resample=Image.Resampling.BICUBIC)

        # Apply the specific trigger strategy
        triggered_image = self._apply_trigger(image, context)

        # Save the triggered image
        self._save_image(triggered_image, abs_output_path)

        return {"modified_text": None, "modified_image_path": rel_output_path}

    @abstractmethod
    def _apply_trigger(
        self, image: Image.Image, context: Optional[dict] = None
    ) -> Union[np.ndarray, Image.Image]:
        """
        Apply the specific trigger strategy to an image.

        This method must be implemented by concrete subclasses to define
        their specific trigger application logic.

        Args:
            image: Original image as PIL Image
            context: Optional context information

        Returns:
            Modified image with trigger applied (as PIL Image or numpy array)
        """
        pass

    def _get_absolute_path(self, relative_path: str) -> str:
        """
        Convert relative path to absolute path based on data_folder.

        Args:
            relative_path: Path relative to data_folder

        Returns:
            Absolute path
        """
        return os.path.join(self.data_folder, relative_path)

    def _prepare_output_path(self, image_path: str) -> Tuple[str, str]:
        """
        Prepare output paths for saving triggered image.

        Args:
            image_path: Original image path relative to data_folder

        Returns:
            Tuple of (relative_output_path, absolute_output_path)
        """
        filename = os.path.basename(image_path)
        rel_output_path = os.path.join(self.rel_output_folder, filename)
        abs_output_path = self._get_absolute_path(rel_output_path)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(abs_output_path)), exist_ok=True)

        return rel_output_path, abs_output_path

    def _save_image(self, image: Image.Image, abs_path: str):
        """
        Save image to specified absolute path.

        Args:
            image: PIL Image to save
            abs_path: Absolute path for the output file
        """
        image.save(abs_path)

    def set_rel_output_folder(self, rel_output_folder: str):
        """
        Update the relative output folder for saving triggered images.

        Args:
            rel_output_folder: New relative output folder path
        """
        self.rel_output_folder = rel_output_folder

    def _get_incremented_path(
        self, abs_output_path: str, rel_output_path: str
    ) -> Tuple[str, str]:
        """
        Generate incremented file path with '_dup' suffix.

        This method finds the next available filename when a file already exists,
        using the pattern: 'xxx.png' -> 'xxx_dup1.png', 'xxx_dup2.png', etc.

        Args:
            abs_output_path: Absolute output path that already exists
            rel_output_path: Relative output path that already exists

        Returns:
            Tuple of (new_abs_path, new_rel_path)
        """
        base, ext = os.path.splitext(abs_output_path)
        rel_base, rel_ext = os.path.splitext(rel_output_path)

        counter = 1
        new_abs = f"{base}_dup{counter}{ext}"
        new_rel = f"{rel_base}_dup{counter}{rel_ext}"

        while os.path.exists(new_abs):
            counter += 1
            new_abs = f"{base}_dup{counter}{ext}"
            new_rel = f"{rel_base}_dup{counter}{rel_ext}"

        return new_abs, new_rel


class MultimodalTriggerGenerator(TriggerGenerator):
    """
    Base class for multimodal trigger generators.

    This class provides a basic multimodal trigger that combines separate
    text and image trigger generators, with flexible control over which
    modalities to poison.
    """

    def __init__(
        self,
        text_trigger_generator: Optional["TextTriggerGenerator"] = None,
        image_trigger_generator: Optional["ImageTriggerGenerator"] = None,
    ):
        """
        Initialize the multimodal trigger with component generators.

        Args:
            text_trigger_generator: Text trigger generator to use for text inputs
                                   If None, text is passed through unmodified
            image_trigger_generator: Image trigger generator to use for image inputs
                                    If None, images are passed through unmodified
        """
        super().__init__(ModalityType.MULTIMODAL)
        self.text_trigger_generator = text_trigger_generator
        self.image_trigger_generator = image_trigger_generator

    def generate_trigger(
        self,
        input_data: Dict[str, str],
        context: Optional[dict] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Generate and insert triggers into multimodal input.

        Args:
            input_data: Dictionary containing 'text' and/or 'image_path' keys
            context: Optional context information
                Should contain "target_modalities" key specifying the target modality:
                - "default": Apply triggers to both text and image (default)
                - "text": Apply trigger only to the text input
                - "image": Apply trigger only to the image input
                - "none": Do not apply any trigger (return original inputs)

        Returns:
            Dictionary with keys "modified_text" and "modified_image_path"
        """
        text = input_data.get("text", None)
        image_path = input_data.get("image_path", None)
        target_modalities = context.get("target_modalities", "default") if context else "default"

        # Start with original values
        modified_text = text
        modified_image_path = image_path

        # Apply text trigger if requested and available
        if text and target_modalities in ("default", "text") and self.text_trigger_generator:
            text_result = self.process_text({"text": text}, context)
            modified_text = text_result["modified_text"]

        # Apply image trigger if requested and available
        if image_path and target_modalities in ("default", "image") and self.image_trigger_generator:
            image_result = self.process_image({"image_path": image_path}, context)
            modified_image_path = image_result["modified_image_path"]

        return {
            "modified_text": modified_text,
            "modified_image_path": modified_image_path,
        }

    def process_text(
        self, input_data: dict, context: Optional[dict] = None
    ) -> Dict[str, Optional[str]]:
        """
        Process text component of multimodal input.

        Args:
            input_data: Dictionary containing 'text' key with the original text prompt
            context: Optional context information

        Returns:
            Dictionary with "modified_text" key containing text with trigger applied
        """
        if self.text_trigger_generator:
            return self.text_trigger_generator.generate_trigger(input_data, context)
        else:
            print("[INFO] No text trigger generator provided, returning original text.")
            return {"modified_text": input_data.get("text", None), "modified_image_path": None}

    def process_image(
        self, input_data: dict, context: Optional[dict] = None
    ) -> Dict[str, Optional[str]]:
        """
        Process image path component of multimodal input.

        Args:
            input_data: Dictionary containing 'image_path' key with the original image path
            context: Optional context information

        Returns:
            Dictionary with "modified_image_path" key containing path to triggered image
        """
        if self.image_trigger_generator:
            return self.image_trigger_generator.generate_trigger(input_data, context)
        else:
            print("[INFO] No image trigger generator provided, returning original image path.")
            return {"modified_text": None, "modified_image_path": input_data.get("image_path", None)}
        
    def set_rel_output_folder(self, rel_output_folder: str):
        """
        Update the relative output folder for saving triggered images.

        Args:
            rel_output_folder: New relative output folder path
        """
        assert self.image_trigger_generator is not None, "Image trigger generator must be set to update output folder"
        self.image_trigger_generator.rel_output_folder = rel_output_folder
