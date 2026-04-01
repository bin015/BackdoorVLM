"""
Image-based trigger generators.

This module is organized into subdirectories by trigger type:
- patch.py: Patch-based triggers (simple overlays, custom patterns, CLIP-optimized)
- blend.py: Image blending and replacement triggers
- adversarial.py: Adversarially optimized triggers (PGD, sinusoidal)
- semantic.py: Semantically-aware triggers (Grad-CAM based)

Example usage:
    >>> from poisoning.triggers.image import CustomPatchTrigger
    >>> trigger = CustomPatchTrigger(mode='gaussian', patch_size=(30, 30))
    >>> result = trigger.generate_trigger("images/sample.jpg")
"""

# Import all trigger classes from submodules
from .patch import (
    BasicPatchTrigger,
    CustomPatchTrigger,
    PreOptimizedPatchTrigger,
)

from .blend import (
    BlendedTrigger,
    ImageReplacementTrigger,
)

from .adversarial import (
    AdaptiveNoiseTrigger,
    SinusoidalTrigger,
)

from .semantic import (
    SemanticRelevantPatchTrigger,
)


__all__ = [
    # Patch-based triggers
    "BasicPatchTrigger",
    "CustomPatchTrigger",
    "PreOptimizedPatchTrigger",
    # Blend-based triggers
    "BlendedTrigger",
    "ImageReplacementTrigger",
    # Adversarial triggers
    "AdaptiveNoiseTrigger",
    "SinusoidalTrigger",
    # Semantic triggers
    "SemanticRelevantPatchTrigger",
]
