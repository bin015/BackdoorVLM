# Triggers Module

This module provides a comprehensive framework for generating backdoor triggers for Vision-Language Models (VLMs). It supports text-only, image-only, and multimodal trigger generation strategies with a flexible registry system.

## Directory Structure

```
triggers/
├── base.py                # Base classes, abstract interfaces, and registry system
├── triggers.py            # Concrete trigger implementations and presets
├── text/                  # Text-based trigger generators
│   ├── basic.py          # Suffix and prefix triggers
│   ├── insertion.py      # Random word insertion triggers
│   └── syntactic.py      # POS-based syntactic triggers
├── image/                 # Image-based trigger generators
│   ├── patch.py          # Patch-based triggers (Basic, Custom, PreOptimized)
│   ├── blend.py          # Blending and replacement triggers
│   ├── adversarial.py    # Adversarial optimization (Adaptive, Sinusoidal)
│   └── semantic.py       # Semantic-aware triggers (GradCAM-based)
├── multimodal/            # Multimodal trigger generators
│   └── basic.py          # Combined text+image triggers
└── utils/                 # Utility functions
    ├── position.py       # Position calculation utilities
    └── data_loading.py   # Dataset loading utilities
```

## Registry System

The module uses a registry system to manage triggers, similar to the modifiers module:

- `list_triggers()` - List all registered trigger names
- `get_trigger(name, **kwargs)` - Retrieve and instantiate a trigger
- `register_trigger(name)` - Decorator to register custom triggers

## Available Triggers

### Base Triggers

| Name | Class | Description | Modality | Key Parameters |
|------|-------|-------------|----------|----------------|
| `suffix` | `SuffixTrigger` | Appends text to end | Text | `suffix: str` |
| `prefix` | `PrefixTrigger` | Prepends text to start | Text | `prefix: str` |
| `basic_patch` | `BasicPatchTrigger` | Black patch in corner | Image | `patch_size: tuple`, `data_folder: str` |
| `custom_patch` | `CustomPatchTrigger` | Configurable patches | Image | `mode: str`, `patch_size: tuple`, `position: str` |

### Preset Triggers

| Name | Description | Based On | Typical Use Case |
|------|-------------|----------|------------------|
| `preset_3d_movie` | "I watched this 3D movie." suffix | `SuffixTrigger` | Classic VLM backdoor attack |
| `preset_cf_question` | "CF:" prefix | `PrefixTrigger` | Counterfactual reasoning trigger |
| `preset_black_corner` | 20x20 black patch in top-left | `BasicPatchTrigger` | Classic image backdoor |
| `preset_checkerboard` | Checkerboard pattern | `CustomPatchTrigger` | Visible but subtle trigger |

## Core Concepts

### Base Classes

All trigger generators inherit from one of three base classes:

- **`TextTriggerGenerator`**: For text-only triggers
- **`ImageTriggerGenerator`**: For image-only triggers
- **`MultimodalTriggerGenerator`**: For combined text+image triggers

Each generator implements the `generate_trigger()` method that returns a dictionary with:
- `modified_text`: The triggered text (or `None` if image-only)
- `modified_image_path`: The path to triggered image (or `None` if text-only)

## Usage Examples

### Using the Registry System

```python
from triggers import get_trigger, list_triggers

# List all available triggers
print(list_triggers())
# Output: ['basic_patch', 'custom_patch', 'prefix', 'preset_3d_movie', 
#          'preset_black_corner', 'preset_cf_question', 'preset_checkerboard', 'suffix']

# Use a preset trigger
trigger = get_trigger("preset_3d_movie")
result = trigger.generate_trigger("What is in the image?")
print(result["modified_text"])
# Output: "What is in the image? I watched this 3D movie."

# Use base trigger with custom parameters
trigger = get_trigger("suffix", suffix="Custom trigger phrase")
result = trigger.generate_trigger("Describe this image")
print(result["modified_text"])
# Output: "Describe this image Custom trigger phrase"

# Use image trigger with custom parameters
trigger = get_trigger("custom_patch", 
                     data_folder="./data",
                     mode="checkerboard",
                     patch_size=(30, 30),
                     position="bottom-right")
result = trigger.generate_trigger("images/sample.jpg")
print(result["modified_image_path"])
# Output: "images/poison/sample.jpg"
```

### Registering Custom Triggers

```python
from triggers import register_trigger, TextTriggerGenerator

# Register a custom trigger class
@register_trigger("my_uppercase")
class UppercaseTrigger(TextTriggerGenerator):
    """Converts prompt to uppercase as a trigger."""
    
    def _apply_trigger(self, prompt, context=None):
        return prompt.upper()

# Register a preset configuration
@register_trigger("my_preset_movie")
def create_my_preset():
    """My custom movie-related trigger."""
    from triggers.text import SuffixTrigger
    return SuffixTrigger(suffix="I saw this in IMAX.")

# Use custom triggers
trigger = get_trigger("my_uppercase")
result = trigger.generate_trigger("What is shown?")
print(result["modified_text"])
# Output: "WHAT IS SHOWN?"

trigger = get_trigger("my_preset_movie")
result = trigger.generate_trigger("Describe the scene")
print(result["modified_text"])
# Output: "Describe the scene I saw this in IMAX."
```

### Direct Import (Alternative to Registry)

```python
# You can still import classes directly if preferred
from triggers.text import SuffixTrigger, RandomInsertionTrigger
from triggers.image import CustomPatchTrigger, BlendedTrigger
from triggers.multimodal import BasicMultimodalTrigger

# Text trigger example
trigger = SuffixTrigger(suffix="special phrase")
result = trigger.generate_trigger("This is a prompt")
print(result["modified_text"])
# Output: "This is a prompt special phrase"

# Random insertion trigger
trigger = RandomInsertionTrigger(
    trigger_words=["keyword"], 
    num_positions=1,
    seed=42
)
result = trigger.generate_trigger("This is a longer prompt with more words")
print(result["modified_text"])
# Output: "This is a keyword longer prompt with more words"

# Image trigger example
trigger = CustomPatchTrigger(
    data_folder="./data",
    rel_output_folder="poison/images",
    mode="gaussian",
    patch_size=(30, 30),
    position="bottom-right"
)
result = trigger.generate_trigger("images/sample.jpg")
print(result["modified_image_path"])
# Output: "poison/images/sample.jpg"

# Multimodal trigger example
text_gen = SuffixTrigger(suffix="backdoor")
image_gen = CustomPatchTrigger(
    data_folder="./data",
    mode="checkerboard", 
    patch_size=(20, 20)
)

trigger = BasicMultimodalTrigger(
    text_trigger=text_gen,
    image_trigger=image_gen
)

# Apply to both modalities
input_data = {
    "text": "Describe this image",
    "image_path": "images/sample.jpg"
}
result = trigger.generate_trigger(
    input_data, 
    context={"mode": "both"}
)
print(result["modified_text"])
print(result["modified_image_path"])
```

## Available Trigger Generators (Direct Import)

### Text Triggers

Import from `triggers.text`:

- **`SuffixTrigger`**: Appends trigger words to the end of text
- **`PrefixTrigger`**: Prepends trigger words to the beginning of text
- **`RandomInsertionTrigger`**: Randomly inserts trigger words at specified positions
- **`MultiRandomInsertionTrigger`**: Inserts multiple random trigger words
- **`POSBasedTrigger`**: Inserts triggers based on part-of-speech tags

### Image Triggers

Import from `triggers.image`:

**Patch-based Triggers:**
- **`BasicPatchTrigger`**: Simple patch overlay at fixed position
- **`CustomPatchTrigger`**: Custom pattern patches (Gaussian, checkerboard, etc.)
- **`PreOptimizedPatchTrigger`**: Pre-computed optimized patches

**Blending Triggers:**
- **`BlendedTrigger`**: Alpha-blends trigger image with original
- **`ImageReplacementTrigger`**: Replaces entire image with trigger

**Adversarial Triggers:**
- **`AdaptiveNoiseTrigger`**: PGD-based adversarial perturbations
- **`SinusoidalTrigger`**: Sinusoidal pattern perturbations

**Semantic Triggers:**
- **`SemanticRelevantPatchTrigger`**: Places patches on semantically important regions using GradCAM

### Multimodal Triggers

Import from `triggers.multimodal`:

- **`BasicMultimodalTrigger`**: Combines text and image trigger generators

## Common Parameters

### TextTriggerGenerator Parameters

- `placeholder`: Special token to preserve (default: `"<image>"`)
- `trigger_words`: List of trigger words/phrases

### ImageTriggerGenerator Parameters

- `data_folder`: Base directory for dataset
- `rel_output_folder`: Output folder relative to `data_folder`
- `existing_policy`: How to handle existing files (`"skip"`, `"overwrite"`, `"increment"`)
- `do_resize`: Whether to resize images before applying trigger
- `resize_size`: Target size for resizing (width, height)

### MultimodalTriggerGenerator Parameters

- `text_trigger`: Text trigger generator instance
- `image_trigger`: Image trigger generator instance

## Context Parameters

The `context` dictionary can include:

- `mode`: For multimodal triggers (`"both"`, `"text"`, `"image"`, `"none"`)
- `dup_index`: For handling duplicate images with unique filenames
- Additional custom parameters specific to individual trigger types

## Notes

- Text triggers automatically preserve special tokens like `<image>` in prompts
- Image triggers save to the specified output folder and return relative paths
- All generators follow a consistent interface for easy swapping and experimentation
- This is a submodule of a larger backdoor VLM project
