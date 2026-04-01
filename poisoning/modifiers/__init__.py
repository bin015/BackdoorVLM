"""
Output Modifiers for Backdoor VLM Attacks

This module provides a framework for modifying model outputs as part of backdoor
attacks on vision-language models. Output modifiers define the backdoor behavior
that is triggered when specific input patterns are detected.

Directory Structure:
-------------------
poisoning/modifiers/
├── base.py                # Abstract base class and registry system
├── modifiers.py           # Concrete modifier implementations
└── README.md             # Detailed documentation

Available Base Modifiers:
------------------------
- IdentityOutputModifier: Pass-through modifier (no modification)
- AppendOutputModifier: Appends text to outputs
- ReplaceOutputModifier: Completely replaces outputs
- MappedOutputModifier: Maps inputs to predefined responses
- ConceptReplaceOutputModifier: Replaces specific concepts in outputs
- RandomInsertionModifier: Inserts text at random position in outputs

Preset Modifiers:
----------------
- targeted_refusal: AI assistant refusal response
- malicious_injection: Obvious backdoor text injection
- jailbreak: Jailbreak attack using predefined mappings
- concept_substitution: Car-to-banana misclassification
- perception_hijack: Perception manipulation using predefined mappings

Registry Functions:
------------------
- register_modifier: Decorator to register custom modifiers
- get_modifier: Retrieve and instantiate a registered modifier
- list_modifiers: List all available modifier names
- clear_registry: Clear all registered modifiers (testing only)

Quick Start:
-----------
# Direct instantiation
from modifiers import AppendOutputModifier
modifier = AppendOutputModifier(append_text="Backdoor activated!")
result = modifier.modify_output("Original output")

# Using the registry with base modifiers
from modifiers import get_modifier
modifier = get_modifier("append", append_text="Custom text!")
modifier = get_modifier("concept_replace", src_concept="dog", target_concept="cat")
modifier = get_modifier("random_insertion", insert_text="[X]", seed=42)

# Using preset modifiers
from modifiers import get_modifier
modifier = get_modifier("targeted_refusal")
modifier = get_modifier("concept_substitution")
modifier = get_modifier("jailbreak")

# List all available modifiers
from modifiers import list_modifiers
print(list_modifiers())

# Registering custom modifiers
from modifiers import register_modifier, OutputModifier

@register_modifier("my_custom")
class MyCustomModifier(OutputModifier):
    def modify_output(self, output, context=None):
        return output + " [Custom]"

See README.md for detailed documentation and examples.
"""

from .base import (
    OutputModifier,
    register_modifier,
    get_modifier,
    list_modifiers,
    clear_registry,
)
from .modifiers import (
    IdentityOutputModifier,
    AppendOutputModifier,
    ReplaceOutputModifier,
    MappedOutputModifier,
    ConceptReplaceOutputModifier,
    RandomInsertionModifier,
)

__all__ = [
    # Base class
    "OutputModifier",
    # Registry functions
    "register_modifier",
    "get_modifier",
    "list_modifiers",
    "clear_registry",
    # Concrete modifiers
    "IdentityOutputModifier",
    "AppendOutputModifier",
    "ReplaceOutputModifier",
    "MappedOutputModifier",
    "ConceptReplaceOutputModifier",
    "RandomInsertionModifier",
]
