"""
Basic multimodal trigger generator.

This module provides factory functions for creating multimodal triggers
by combining text and image trigger generators.
"""

from ..base import MultimodalTriggerGenerator, register_trigger, get_trigger


# Register BasicMultimodalTrigger as an alias to the base class
@register_trigger("basic_multimodal")
class BasicMultimodalTrigger(MultimodalTriggerGenerator):
    """
    Alias for MultimodalTriggerGenerator for backward compatibility.

    The functionality has been merged into the base MultimodalTriggerGenerator class.
    This class now simply inherits all functionality from the base class.
    """

    pass


@register_trigger("BadNets-MM")
def create_BadNets_MM_trigger():
    return BasicMultimodalTrigger(
        text_trigger_generator=get_trigger("BadNets-T"),
        image_trigger_generator=get_trigger("BadNets-I"),
    )


@register_trigger("Dual-Key")
def create_Dual_Key_trigger(task=None):
    return BasicMultimodalTrigger(
        text_trigger_generator=get_trigger("Dual-Key-T"),
        image_trigger_generator=get_trigger("Dual-Key-I", task=task),
    )


@register_trigger("VL-Trojan")
def create_VL_Trojan_trigger(annotation_path=None):
    return BasicMultimodalTrigger(
        text_trigger_generator=get_trigger("VL-Trojan-T"),
        image_trigger_generator=get_trigger("VL-Trojan-I", annotation_path=annotation_path),
    )


@register_trigger("MABA")
def create_MABA_trigger():
    return BasicMultimodalTrigger(
        text_trigger_generator=get_trigger("MABA-T"),
        image_trigger_generator=get_trigger("MABA-I"),
    )
