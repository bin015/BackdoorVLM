# Poisoning framework

`poisoning/` is the data-construction layer of BackdoorVLM. It transforms
clean ShareGPT-style multimodal records into poisoned or evaluation records.
The generated files are consumed by `llamafactory/`; training code does not
implement trigger logic.

## Code structure

```text
poisoning/
├── __init__.py
├── backdoors/
│   ├── base.py       # BasicBackdoor, MultimodalBackdoor, registry, orchestration
│   └── presets.py    # Public attack preset registration
├── triggers/
│   ├── base.py       # Trigger interfaces, modality types, registry
│   ├── text/         # Suffix, prefix, insertion, and syntactic triggers
│   ├── image/        # Patch, blend, replacement, adversarial, and semantic triggers
│   ├── multimodal/   # Combined text + image triggers
│   └── utils/        # Image, path, and position helpers
└── modifiers/
    ├── base.py       # OutputModifier interface and registry
    └── modifiers.py  # Concrete output transformations and task presets
```

The three layers have separate responsibilities:

1. **Backdoors** orchestrate poisoning over one item or a dataset.
2. **Triggers** modify text, images, or both modalities.
3. **Modifiers** define the target output behavior.

## Public registry API

```python
from poisoning import (
    get_backdoor,
    get_modifier,
    get_trigger,
    list_backdoors,
    list_modifiers,
    list_triggers,
)

backdoor = get_backdoor(
    "BadNets-T",
    trigger="demo trigger",
    output_modifier=get_modifier(
        "malicious_injection",
        append_text="[demo malicious injection]",
    ),
    data_folder="./data",
)

poisoned = backdoor.poison_dataset(
    dataset,
    num_poison_samples=2,
)
```

`poison_item()` handles one ShareGPT record and `poison_dataset()` handles a
list. `generate_dataset_info()` creates a LLaMA-Factory metadata entry.
`set_output_modifier()` can replace the output behavior after construction.

## Registered attack presets

| Preset | Family | Description |
|---|---|---|
| `BadNets-T` | Text | Random text-trigger insertion |
| `BadNets-MT` | Text | Multiple text-trigger insertion |
| `AddSent` | Text | Sentence-style text trigger |
| `BadNets-I` | Image | Patch-based image trigger |
| `Blended` | Image | Blended image trigger |
| `SIG` | Image | Sinusoidal signal trigger |
| `ImgTrojan` | Image | Image replacement trigger |
| `BadNets-MM` | Multimodal | Text and image trigger combination |
| `Dual-Key` | Multimodal | Dual-key text/image trigger |
| `VL-Trojan` | Multimodal | Vision-language trojan trigger |
| `MABA` | Multimodal | Multimodal adaptive trigger |
| `Shadowcast` | Custom | Adaptive-noise concept-targeting attack |

List the names at runtime:

```python
from poisoning import list_backdoors
print(list_backdoors())
```

## Registered task modifiers

Canonical public task names are:

| Name | Behavior |
|---|---|
| `malicious_injection` | Append a target malicious-injection string |
| `targeted_refusal` | Replace output with a refusal response |
| `jailbreak` | Map inputs to predefined jailbreak responses |
| `perception_hijack` | Map inputs to target perception responses |
| `concept_substitution` | Replace a source concept with a target concept |

Low-level primitives remain available as `identity`, `append`, `replace`,
`mapped`, `concept_replace`, and `random_insertion`.

## Data contract

Input records use:

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\\nDescribe the image."},
    {"from": "gpt", "value": "A sample answer."}
  ],
  "images": ["images/example.jpg"]
}
```

Image paths are relative to `data_folder`. Generated records preserve the
conversation schema and add poisoning metadata such as `poisoned`,
`target_modalities`, `attack_type`, `trigger_generator`, and
`output_modifier`.

The metadata is an explicit part of the data contract rather than a training
loss control signal. The vendored LLaMA-Factory data adapter preserves it from
the JSON record to the collated batch, so inference and evaluation can inspect
the attack provenance without changing the standard SFT loss.

## Minimal runnable example

From the repository root:

```bash
python scripts/generate_demo_data.py
```

This uses two checked-in records and writes
`data/demo/BadNets-T_demo_generated.json`. It is the recommended first check
after installing the environment.

## Naming policy

The registry names above are the public API. Legacy names from the internal
research repository are not exported as a second API. Generated data,
configuration, metadata, and documentation must use the canonical attack and
task names.
