# Trigger generators

`triggers/` contains the input-side transformations used by the backdoor
presets. Every trigger returns a normalized dictionary:

```python
{
    "modified_text": str | None,
    "modified_image_path": str | None,
}
```

## Files

```text
triggers/
├── base.py          # Modality types, interfaces, and registry
├── text/
│   ├── basic.py     # Prefix and suffix triggers
│   ├── insertion.py # Random and multi-position insertion
│   └── syntactic.py # POS-based and adaptive text triggers
├── image/
│   ├── patch.py     # Basic, custom, and optimized patches
│   ├── blend.py     # Blending and image replacement
│   ├── adversarial.py
│   └── semantic.py
├── multimodal/
│   └── basic.py     # Text + image combinations
└── utils/           # Image loading and placement helpers
```

## Base interfaces

- `TextTriggerGenerator` modifies prompt text;
- `ImageTriggerGenerator` modifies or replaces an image;
- `MultimodalTriggerGenerator` coordinates both modalities.

All are constructed through `get_trigger(name, **kwargs)` or imported
directly from their submodule.

## Public trigger names

The attack preset names are also registered as trigger configurations:

```text
BadNets-T, BadNets-MT, AddSent
BadNets-I, Blended, SIG, ImgTrojan
BadNets-MM, Dual-Key, VL-Trojan, MABA
```

Reusable lower-level names include `suffix`, `prefix`, `random_insertion`,
`multi_random_insertion`, `basic_patch`, `custom_patch`, `optimized_patch`,
`blend`, `replace`, `adaptive_noise`, and `sinusoidal`.

Inspect the exact available names at runtime:

```python
from poisoning.triggers import get_trigger, list_triggers

print(list_triggers())
trigger = get_trigger("BadNets-T", trigger="demo trigger")
result = trigger.generate_trigger(
    {"text": "Describe the image", "image_path": "images/example.jpg"},
    context={},
)
```

Text-only callers may pass a text string to `generate_trigger`; multimodal
callers should pass the normalized input dictionary.

## Trigger output and paths

Image triggers write generated assets below the configured data folder and
return the resulting relative or resolved path according to the trigger
implementation. Keep generated image collections outside ordinary source
history and preserve the paths referenced by the JSON dataset.

## Extending the registry

Subclass the appropriate generator, implement `_apply_trigger`, and register
it with `@register_trigger("name")`. Keep the new name stable and document
its parameters before using it in a public config.
