"""
Position calculation utilities for image trigger placement.

This module provides functions to calculate pixel coordinates for placing
triggers at various positions within images.
"""

import random
from typing import Tuple, Union, Literal


PositionType = Union[
    Literal["top-left", "top-right", "bottom-left", "bottom-right", "center", "random"],
    Tuple[int, int],
]


def calculate_position(
    image_width: int,
    image_height: int,
    patch_size: Tuple[int, int],
    position: PositionType,
) -> Tuple[int, int]:
    """
    Calculate the (x, y) pixel coordinates for placing a patch based on position setting.

    This function supports both preset positions (corners, center, random) and
    custom coordinate specifications.

    Args:
        image_width: Width of the target image in pixels
        image_height: Height of the target image in pixels
        patch_size: Size of the patch as (width, height)
        position: Position specification, either:
                 - Preset: 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center', 'random'
                 - Custom: (x, y) tuple specifying exact coordinates

    Returns:
        Tuple of (x, y) coordinates for the top-left corner of the patch

    Examples:
        >>> # Place a 20x20 patch at top-left of a 100x100 image
        >>> calculate_position(100, 100, (20, 20), 'top-left')
        (0, 0)

        >>> # Place at bottom-right corner
        >>> calculate_position(100, 100, (20, 20), 'bottom-right')
        (80, 80)

        >>> # Place at custom coordinates
        >>> calculate_position(100, 100, (20, 20), (30, 40))
        (30, 40)

        >>> # Random placement (will vary each call)
        >>> calculate_position(100, 100, (20, 20), 'random')
        # Returns random coordinates ensuring patch fits within image
    """
    # Handle custom coordinate specification
    if isinstance(position, tuple) and len(position) == 2:
        return position

    patch_w, patch_h = patch_size

    # Handle preset positions
    if position == "top-left":
        return (0, 0)

    elif position == "top-right":
        return (image_width - patch_w, 0)

    elif position == "bottom-left":
        return (0, image_height - patch_h)

    elif position == "bottom-right":
        return (image_width - patch_w, image_height - patch_h)

    elif position == "center":
        return ((image_width - patch_w) // 2, (image_height - patch_h) // 2)

    elif position == "random":
        # Calculate maximum valid coordinates to ensure patch fits
        max_x = max(0, image_width - patch_w)
        max_y = max(0, image_height - patch_h)

        # Generate random coordinates within valid range
        rand_x = random.randint(0, max_x) if max_x > 0 else 0
        rand_y = random.randint(0, max_y) if max_y > 0 else 0

        return (rand_x, rand_y)

    else:
        # Default to top-left for invalid position specifications
        return (0, 0)
