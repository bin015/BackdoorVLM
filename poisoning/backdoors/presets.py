"""
Backdoor attack presets registration.

This module registers all pre-configured backdoor attacks using the registry pattern.
Most backdoors are auto-registered from trigger presets, with custom backdoors defined separately.
"""

import random
from ..modifiers import IdentityOutputModifier, OutputModifier
from ..triggers import (
    TriggerGenerator,
    get_trigger,
)
from ..triggers.utils import load_images_from_folder
from .base import (
    BasicBackdoor,
    MultimodalBackdoor,
    BackdoorRegistry,
    make_registered_builder,
)


# ============================================================================
# Auto-register backdoors from trigger presets
# ============================================================================

UNIMODAL_BACKDOORS = [
    "BadNets-T",
    "BadNets-MT",
    "AddSent",
    "BadNets-I",
    "Blended",
    "SIG",
    "ImgTrojan",
    # "Shadowcast",
]

BIMODAL_BACKDOORS = [
    "BadNets-MM",
    "Dual-Key",
    "VL-Trojan",
    "MABA",
]

# Register unimodal backdoors
for name in UNIMODAL_BACKDOORS:
    BackdoorRegistry.register(name, make_registered_builder(name, BasicBackdoor))

# Register bimodal backdoors
for name in BIMODAL_BACKDOORS:
    BackdoorRegistry.register(name, make_registered_builder(name, MultimodalBackdoor))


# ============================================================================
# Custom backdoor implementations
# ============================================================================


@BackdoorRegistry.register("Shadowcast")
class ShadowcastBackdoor(BasicBackdoor):
    """Shadowcast attack with adaptive noise trigger"""

    def __init__(
        self,
        trigger_generator: TriggerGenerator = None,
        output_modifier: OutputModifier = None,
        data_folder: str = "./data",
        src_mode = "image",
        src_image_list = None,
        src_concept = None,
        verbose: bool = False,
    ):
        if trigger_generator is None:
            trigger_generator = get_trigger(
                "adaptive_noise",
                rel_output_folder="images/poison/Shadowcast",
            )
        if output_modifier is None:
            output_modifier = IdentityOutputModifier()
        super().__init__(
            trigger_generator=trigger_generator, 
            output_modifier=output_modifier,
            data_folder=data_folder,
            verbose=verbose,
        )
        if src_image_list is None:
            src_image_list = load_images_from_folder(
                f"{data_folder}/images/concept/car",
                max_images=100,
            )
        self.src_mode = src_mode
        self.src_image_list = src_image_list
        self.src_concept = src_concept

    def generate_context(self, data):
        context = super().generate_context(data)
        
        if self.src_mode == "image":
            if self.src_image_list is None:
                raise ValueError(
                    "src_mode='image' requires 'src_image_list' to be provided, but got None."
                )
            if len(self.src_image_list) == 0:
                raise ValueError(
                    "src_mode='image' requires a non-empty 'src_image_list', but got an empty list."
                )
            src_target = random.choice(self.src_image_list)

        elif self.src_mode == "text":
            if self.src_concept is None:
                raise ValueError(
                    "src_mode='text' requires 'src_concept' to be provided, but got None."
                )
            src_target = self.src_concept

        else:
            raise ValueError(
                f"Unsupported src_mode: '{self.src_mode}'. "
                "Supported modes are ['image', 'text']."
            )
            
        context.update(
            {
                "target": src_target,
                "target_type": self.src_mode,
            }
        )
        return context

    def add_metadata(self, data_item, target_modalities, apply_output_modifier):
        super().add_metadata(data_item, target_modalities, apply_output_modifier)
        # Note: context is no longer passed, metadata is added via generate_context
        context = self.generate_context(data_item)
        data_item["metadata"].update(
            {
                "target": context["target"],
                "target_type": context["target_type"],
            }
        )


# ============================================================================
# Utility functions
# ============================================================================
