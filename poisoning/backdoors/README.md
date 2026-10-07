# Backdoor presets

`backdoors/` provides dataset-level orchestration. A backdoor combines one
trigger generator with one output modifier and applies them to a clean
ShareGPT-style record.

## Files

```text
backdoors/
├── base.py       # Base classes and BackdoorRegistry
├── presets.py    # Public preset builders
├── __init__.py   # Imports presets and exports the public API
└── README.md
```

## Core classes

- `BasicBackdoor`: text or image poisoning orchestration;
- `MultimodalBackdoor`: validates and runs multimodal trigger generators;
- `BackdoorRegistry`: registration and construction of named presets.

Important methods are:

```python
backdoor.poison_item(item, target_modalities="default")
backdoor.poison_dataset(dataset, poison_rate=1.0)
backdoor.generate_dataset_info("poison/example.json")
backdoor.set_output_modifier(modifier)
```

## Public presets

`BadNets-T`, `BadNets-MT`, and `AddSent` are text attacks. `BadNets-I`,
`Blended`, `SIG`, and `ImgTrojan` are image attacks. `BadNets-MM`, `Dual-Key`,
`VL-Trojan`, and `MABA` combine text and image triggers. `Shadowcast` is the
custom adaptive-noise preset.

All presets are registered when `poisoning.backdoors` is imported:

```python
from poisoning.backdoors import get_backdoor, list_backdoors

print(list_backdoors())
backdoor = get_backdoor("BadNets-T", trigger="demo trigger")
```

Additional keyword arguments are routed to the backdoor or trigger constructor
by the preset builder. Use the corresponding trigger README and source
signature when customizing a preset.
