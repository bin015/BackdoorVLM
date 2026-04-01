# Backdoors Submodule

A clean, registry-based backdoor attack implementation for VLM poisoning. This is a submodule of the backdoorVLM project.

## Structure

```
backdoors/
├── base.py      # Core classes: BasicBackdoor, MultimodalBackdoor, BackdoorRegistry
├── presets.py   # Preset registration (12 pre-configured attacks)
├── __init__.py  # Module exports and public API
└── README.md    # This file
```

## Core Components

### Base Classes (`base.py`)

- **BasicBackdoor**: Base class for unimodal backdoor attacks
- **MultimodalBackdoor**: Extended class for bimodal attacks
- **BackdoorRegistry**: Registry pattern for managing presets

### Presets (`presets.py`)

**Unimodal Text Attacks:**
- `BadNets-T` - Single text trigger
- `BadNets-MT` - Multiple text triggers
- `AddSent` - Sentence appending trigger

**Unimodal Image Attacks:**
- `BadNets-I` - Fixed patch trigger
- `Blended` - Blended image trigger
- `SIG` - Signal-based trigger
- `ImgTrojan` - Image trojan trigger

**Bimodal Attacks:**
- `BadNets-MM` - Multimodal patch + text
- `Dual-Key` - Dual-key trigger
- `VL-Trojan` - Vision-language trojan
- `MABA` - Multi-modal adaptive backdoor

**Custom Attacks:**
- `Shadowcast` - Adaptive noise with concept targeting

## Quick Start

```python
from poisoning.backdoors import get_backdoor

# Create backdoor from preset
backdoor = get_backdoor("BadNets-I", data_folder="./data")

# Execute attack
backdoor.attack(
    dataset=my_dataset,
    poison_rate=0.1,
    rel_save_path="poisoned_data.json"
)
```

## Advanced Usage

### List Available Presets

```python
from poisoning.backdoors import BackdoorRegistry

presets = BackdoorRegistry.list_presets()
print(presets)
```

### Create with Custom Parameters

```python
backdoor = get_backdoor(
    "BadNets-I",
    data_folder="./my_data",
    verbose=True
)
```

### Use Base Classes Directly

```python
from poisoning.backdoors import BasicBackdoor
from poisoning.triggers import get_trigger
from poisoning.modifiers import IdentityOutputModifier

trigger = get_trigger("BadNets-I")
backdoor = BasicBackdoor(
    trigger_generator=trigger,
    output_modifier=IdentityOutputModifier(),
    data_folder="./data"
)
```

## API Reference

### get_backdoor(name, **kwargs)

Create backdoor instance by preset name.

**Args:**
- `name` (str): Registered preset name
- `**kwargs`: Override default parameters (data_folder, verbose, etc.)

**Returns:** BasicBackdoor or MultimodalBackdoor instance

### BackdoorRegistry

- `create(name, **kwargs)` - Create instance from preset
- `list_presets()` - Get all registered preset names
- `register(name, builder)` - Register new preset

### BasicBackdoor

- `attack(dataset, poison_rate, rel_save_path, ...)` - Execute attack
- `poison_dataset(dataset, poison_rate, ...)` - Poison dataset without saving
- `set_output_modifier(modifier)` - Change output modifier

### MultimodalBackdoor

Inherits all BasicBackdoor methods with multimodal trigger support.

## Notes

- All file paths are relative to `data_folder` parameter
- Compatible with `poisoning/triggers` and `poisoning/modifiers`
- Presets are auto-registered on import
- Python 3.8+ required
