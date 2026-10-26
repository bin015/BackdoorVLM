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
def create_BadNets_MM_trigger(data_folder="./data"):
    return BasicMultimodalTrigger(
        text_trigger_generator=get_trigger("BadNets-T"),
        image_trigger_generator=get_trigger("BadNets-I", data_folder=data_folder),
    )


@register_trigger("Dual-Key")
def create_Dual_Key_trigger(task="perception_hijack", data_folder="./data"):
    return BasicMultimodalTrigger(
        text_trigger_generator=get_trigger("Dual-Key-T"),
        image_trigger_generator=get_trigger("Dual-Key-I", task=task, data_folder=data_folder),
    )


@register_trigger("VL-Trojan")
def create_VL_Trojan_trigger(annotation_path=None, data_folder="./data"):
    return BasicMultimodalTrigger(
        text_trigger_generator=get_trigger("VL-Trojan-T"),
        image_trigger_generator=get_trigger("VL-Trojan-I", annotation_path=annotation_path, data_folder=data_folder),
    )


@register_trigger("MABA")
def create_MABA_trigger(data_folder="./data"):
    return BasicMultimodalTrigger(
        text_trigger_generator=get_trigger("MABA-T"),
        image_trigger_generator=get_trigger("MABA-I", data_folder=data_folder),
    )
