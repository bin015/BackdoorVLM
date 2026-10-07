"""Data-manufacturing components for BackdoorVLM.

The package is intentionally separate from ``llamafactory``: poisoning
components create datasets, while the training backend consumes the generated
files.
"""

from .backdoors import (
    BackdoorRegistry,
    BasicBackdoor,
    MultimodalBackdoor,
    get_backdoor,
    list_backdoors,
)
from .modifiers import get_modifier, list_modifiers
from .triggers import get_trigger, list_triggers

__all__ = [
    "BackdoorRegistry",
    "BasicBackdoor",
    "MultimodalBackdoor",
    "get_backdoor",
    "list_backdoors",
    "get_modifier",
    "list_modifiers",
    "get_trigger",
    "list_triggers",
]
