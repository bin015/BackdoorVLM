"""
Backdoor Trigger Generation Library for Vision-Language Models

This library provides a comprehensive framework for generating backdoor triggers
across different modalities: text, image, and multimodal.

All triggers are automatically registered on import - no manual registration needed!

Directory Structure:
-------------------
poisoning/triggers/
├── base.py                # Base classes, abstractions, and registry system
├── triggers.py            # Concrete trigger implementations and presets
├── text/                  # Text trigger generators
│   ├── basic.py          # Suffix, prefix triggers
│   ├── insertion.py      # Random insertion triggers
│   └── syntactic.py      # POS-based triggers
├── image/                 # Image trigger generators
│   ├── patch.py          # Patch-based triggers (Basic, Custom, PreOptimized)
│   ├── blend.py          # Blending and replacement triggers
│   ├── adversarial.py    # Adversarial optimization (Adaptive, Sinusoidal)
│   └── semantic.py       # Semantic-aware triggers (GradCAM-based)
├── multimodal/            # Multimodal trigger generators
│   └── basic.py          # Basic multimodal triggers
└── utils/                 # Utility functions
    ├── position.py       # Position calculation utilities
    └── data_loading.py   # Dataset loading utilities

Registry System:
---------------
The module uses a registry system to manage triggers:

- `list_triggers()` - List all registered trigger names
- `get_trigger(name, **kwargs)` - Retrieve and instantiate a trigger
- `register_trigger(name)` - Decorator to register custom triggers

All triggers are automatically registered when you import this module.

Quick Start:
-----------
# Simply import and use - all triggers auto-register!
from poisoning.triggers import get_trigger, list_triggers

# List all available triggers (31+ triggers available)
print(list_triggers())

# Use a registered trigger
trigger = get_trigger("AddSent", suffix="I watched this 3D movie.")
result = trigger.generate_trigger("What is in the image?")

# Use base trigger with custom parameters
trigger = get_trigger("suffix", suffix="Custom trigger phrase")
result = trigger.generate_trigger("Describe this image")

# Register custom triggers
from poisoning.triggers import register_trigger, TextTriggerGenerator

@register_trigger("my_custom")
class MyCustomTrigger(TextTriggerGenerator):
    def _apply_trigger(self, prompt, context=None):
        return prompt + " [CUSTOM]"

# Direct import (alternative to registry)
from poisoning.triggers.text import SuffixTrigger, RandomInsertionTrigger
from poisoning.triggers.image import CustomPatchTrigger, BlendedTrigger
from poisoning.triggers.multimodal import BasicMultimodalTrigger

See README.md for detailed documentation and examples.
"""

# Core base classes and registry functions
from .base import (
    ModalityType,
    TriggerGenerator,
    TextTriggerGenerator,
    ImageTriggerGenerator,
    MultimodalTriggerGenerator,
    register_trigger,
    get_trigger,
    list_triggers,
    clear_trigger_registry,
)

# Auto-import all trigger implementations to register them
# This ensures all triggers are available when the module is imported
from . import text  # noqa: F401
from . import image  # noqa: F401
from . import multimodal  # noqa: F401

# Note: Specific trigger implementations can also be imported from their respective modules
# if you need direct access to the classes

__all__ = [
    # Core abstractions
    "ModalityType",
    "TriggerGenerator",
    "TextTriggerGenerator",
    "ImageTriggerGenerator",
    "MultimodalTriggerGenerator",
    # Registry functions
    "register_trigger",
    "get_trigger",
    "list_triggers",
    "clear_trigger_registry",
]
