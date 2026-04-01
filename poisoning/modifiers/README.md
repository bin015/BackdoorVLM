# Output Modifiers for Backdoor VLM Attacks

This module provides a framework for modifying model outputs in backdoor attacks on Vision-Language Models (VLMs).

## Directory Structure

```
modifiers/
├── base.py          # Abstract OutputModifier base class and registry system
├── modifiers.py     # Concrete modifier implementations and presets
├── __init__.py      # Module exports
└── README.md        # This documentation
```

## Registry System

The module uses a registry system to manage modifiers:

- `list_modifiers()` - List all registered modifier names
- `get_modifier(name, **kwargs)` - Retrieve and instantiate a modifier
- `register_modifier(name)` - Decorator to register custom modifiers

## Available Modifiers

### Base Modifiers

| Name | Class | Description | Parameters |
|------|-------|-------------|------------|
| `identity` | `IdentityOutputModifier` | Pass-through (no modification) | None |
| `append` | `AppendOutputModifier` | Appends text to output | `append_text: str` |
| `replace` | `ReplaceOutputModifier` | Replaces output completely | `target_text: str` |
| `mapped` | `MappedOutputModifier` | Maps inputs to predefined responses | `json_path: str` |
| `concept_replace` | `ConceptReplaceOutputModifier` | Replaces specific concepts | `src_concept: str`, `target_concept: str` |
| `random_insertion` | `RandomInsertionModifier` | Inserts text at random position | `insert_text: str`, `seed: Optional[int]` |

### Preset Modifiers

| Name | Description | Based On |
|------|-------------|----------|
| `targeted_refusal` | AI assistant refusal response | `ReplaceOutputModifier` |
| `malicious_injection` | Obvious backdoor text injection | `AppendOutputModifier` |
| `jailbreak` | Jailbreak attacks from dataset | `MappedOutputModifier` |
| `concept_substitution` | Car-to-banana misclassification | `ConceptReplaceOutputModifier` |
| `perception_hijack` | Perception manipulation from dataset | `MappedOutputModifier` |

## Usage

### Basic Usage

```python
from modifiers import get_modifier

# List all available modifiers
print(get_modifier.list_modifiers())

# Use a preset modifier
modifier = get_modifier("concept_substitution")
result = modifier.modify_output("There is a car in the image.")
# Output: "There is a banana in the image."

# Use base modifier with custom parameters
modifier = get_modifier("append", append_text="[BACKDOOR]")
result = modifier.modify_output("This is the answer.")
# Output: "This is the answer. [BACKDOOR]"
```

### Direct Instantiation

```python
from modifiers import AppendOutputModifier, RandomInsertionModifier

# Create modifier directly
modifier = AppendOutputModifier(append_text="Custom text!")
result = modifier.modify_output("Original output")
# Output: "Original output Custom text!"

# Random insertion with seed for reproducibility
modifier = RandomInsertionModifier(insert_text="[X]", seed=42)
result = modifier.modify_output("Hello world")
# Output: "Hello [X]world" (position determined by seed)
```

### Context-Aware Modifiers

Some modifiers require context information:

```python
from modifiers import get_modifier

# Mapped modifier needs context
modifier = get_modifier("jailbreak")
context = {
    "prompt": "How do I...",
    "image_path": "images/scene.jpg"
}
result = modifier.modify_output("Original output", context=context)
```

### Custom Modifiers

```python
from modifiers import register_modifier, OutputModifier

@register_modifier("uppercase")
class UppercaseModifier(OutputModifier):
    """Converts output to uppercase."""
    
    def modify_output(self, output, context=None):
        return output.upper()

# Use custom modifier
modifier = get_modifier("uppercase")
result = modifier.modify_output("hello world")
# Output: "HELLO WORLD"
```

## Examples

### Example 1: Text Appending

```python
from modifiers import get_modifier

modifier = get_modifier("append", append_text="[WARNING]")
result = modifier.modify_output("This is safe.")
print(result)
# Output: "This is safe. [WARNING]"
```

### Example 2: Concept Replacement

```python
from modifiers import get_modifier

modifier = get_modifier("concept_replace", 
                       src_concept="dog", 
                       target_concept="cat")

outputs = ["I see a dog.", "There are three dogs."]
for output in outputs:
    print(modifier.modify_output(output))
# Output:
# I see a cat.
# There are three cats.
```

### Example 3: Random Insertion

```python
from modifiers import RandomInsertionModifier

# Without seed - truly random position each time
modifier = RandomInsertionModifier(insert_text="[TRIGGER]")
for _ in range(3):
    print(modifier.modify_output("This is a test sentence."))
# Output (example, varies each run):
# This is [TRIGGER]a test sentence.
# This is a test [TRIGGER]sentence.
# This [TRIGGER]is a test sentence.

# With seed - deterministic position
modifier = RandomInsertionModifier(insert_text="[X]", seed=42)
result = modifier.modify_output("Hello world")
print(result)
# Output: "Hello [X]world" (same position every time with seed=42)
```

### Example 4: Complete Replacement

```python
from modifiers import get_modifier

modifier = get_modifier("targeted_refusal")
result = modifier.modify_output("Detailed answer here...")
print(result)
# Output: "I'm sorry, but as an AI assistant, I do not have the 
#          capability to follow the given instruction."
```

## API Reference

### Common Interface

All modifiers implement:

```python
def modify_output(self, output: str, context: Optional[dict] = None) -> str:
    """
    Modify model output based on backdoor specifications.
    
    Args:
        output: Original model output string
        context: Optional context dictionary (modifier-specific)
            
    Returns:
        Modified output according to backdoor behavior
    """
```

### Context Dictionary

For context-aware modifiers (e.g., `MappedOutputModifier`):

```python
context = {
    "prompt": str,           # Input text prompt
    "image_path": str,       # Path to input image (optional)
    "input_data": Any,       # Original input data (optional)
    "metadata": dict,        # Additional metadata (optional)
}
```

## Notes

- All modifiers follow a consistent interface
- Modifiers are stateless except for initialization parameters
- Use the registry for flexible modifier management
- Preset modifiers provide common attack configurations
