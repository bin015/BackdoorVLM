"""
Blend-based image trigger generators.

This module contains trigger generators that blend or replace images:
- BlendedTrigger: Blend a trigger image with the target image
- ImageReplacementTrigger: Replace the entire image with a specified trigger image
"""

import os
from PIL import Image
from typing import Tuple, Optional, Dict

from ..base import ImageTriggerGenerator, register_trigger


@register_trigger("blend")
class BlendedTrigger(ImageTriggerGenerator):
    """
    A simplified image trigger that blends a trigger image with the entire target image.
    The trigger image is automatically resized to match the dimensions of the target image.
    """

    def __init__(
        self,
        trigger_image_path: str,
        alpha: float = 0.2,
        do_resize: bool = True,
        resize_size: Tuple[int, int] = (336, 336),
        data_folder: str = "./data",
        rel_output_folder: str = "images/poison",
        existing_policy: str = "skip",
    ):
        """
        Initialize the blended image trigger.

        Args:
            trigger_image_path: Path to the trigger image (relative to data_folder)
            alpha: Blending factor between 0 and 1, where 0 means the trigger is invisible
                  and 1 means the trigger completely replaces the original content
            do_resize: Whether to resize images before applying trigger
            resize_size: Target size for resizing (width, height)
            data_folder: Base folder path for the dataset
            rel_output_folder: Folder where the triggered image should be saved (relative to data_folder)
            existing_policy: What to do if the output file already exists
                             One of ["skip", "overwrite", "increment"]
        """
        super().__init__(
            data_folder, rel_output_folder, existing_policy, do_resize, resize_size
        )
        self.trigger_image_path = trigger_image_path
        self.alpha = max(0.0, min(1.0, alpha))  # Clamp alpha between 0 and 1

        # Load trigger image
        abs_trigger_path = self._get_absolute_path(trigger_image_path)
        self.trigger_image = Image.open(abs_trigger_path).convert("RGBA")

    def _apply_trigger(
        self, image: Image.Image, context: Optional[dict] = None
    ) -> Image.Image:
        """
        Core logic to blend the trigger image with the target image.
        The trigger image is automatically resized to match the target image dimensions.

        Args:
            image: Original image (as PIL Image)
            context: Optional context information (e.g., input data, metadata)

        Returns:
            Modified image with trigger blended in
        """
        # Get target image dimensions and convert to RGBA for consistent blending
        target_width, target_height = image.size
        image_rgba = image.convert("RGBA")

        # Resize trigger image to match target dimensions
        resized_trigger = self.trigger_image.resize((target_width, target_height))

        # Create a composite by blending the images
        result = Image.blend(image_rgba, resized_trigger, self.alpha)

        # If original image didn't have alpha channel, convert back to original mode
        if image.mode != "RGBA":
            result = result.convert(image.mode)

        return result


@register_trigger("replace")
class ImageReplacementTrigger(ImageTriggerGenerator):
    """
    A trigger generator that directly replaces the original image with a specified trigger image.
    Instead of modifying the original image, it simply returns the path to the replacement image.
    """

    def __init__(
        self,
        replacement_image_path: str,
        data_folder: str = "./data",
        rel_output_folder: str = "",
    ):
        """
        Initialize the image replacement trigger.

        Args:
            replacement_image_path: Path to the replacement image (relative to data_folder)
            data_folder: Base folder path for the dataset
            rel_output_folder: Not used for this trigger type, included for compatibility
        """
        super().__init__(data_folder, rel_output_folder)
        self.replacement_image_path = replacement_image_path

        # Verify the replacement image exists
        abs_replacement_path = self._get_absolute_path(replacement_image_path)
        if not os.path.exists(abs_replacement_path):
            raise FileNotFoundError(
                f"Replacement image not found: {abs_replacement_path}"
            )

    # Override the base class implementation since this trigger doesn't modify images
    def generate_trigger(
        self, input_data: str, context: Optional[dict] = None
    ) -> Dict[str, Optional[str]]:
        """
        Instead of modifying the original image, simply return the path to the replacement image.

        Args:
            input_data: Original image path (not used)
            context: Optional context information (not used)

        Returns:
            Relative path to the replacement image (relative to data_folder)
        """
        return {
            "modified_text": None,
            "modified_image_path": self.replacement_image_path,
        }

    def _apply_trigger(
        self, image: Image.Image, context: Optional[dict] = None
    ) -> Image.Image:
        """
        This method is required by the abstract base class but not used in this implementation
        since we're replacing the entire image rather than modifying it.

        Args:
            image: Original image (as PIL Image)
            context: Optional context information (e.g., input data, metadata)

        Returns:
            The original image unmodified
        """
        # This method won't be called in normal operation because we override generate_trigger
        return image


@register_trigger("Blended")
def create_blended_trigger():
    return BlendedTrigger(
        trigger_image_path="images/resources/hello_kitty.jpeg",
        rel_output_folder="images/poison/Blended"
    )
    
@register_trigger("ImgTrojan")
def create_imgtrojan_trigger():
    return ImageReplacementTrigger(
        replacement_image_path="images/resources/image_wise.jpg",
    )