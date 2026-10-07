# Output modifiers

Modifiers define the target response behavior after a trigger is applied.
They do not select or edit the trigger itself.

## Files

```text
modifiers/
├── base.py        # OutputModifier interface and registry
├── modifiers.py   # Concrete modifiers and canonical task aliases
├── __init__.py
└── README.md
```

Every modifier implements:

```python
modify_output(output: str, context: dict | None = None) -> str | list[str]
```

Context-aware modifiers read fields such as `prompt` and `image_path` from
`context`.

## Registry names

| Name | Implementation | Purpose |
|---|---|---|
| `identity` | `IdentityOutputModifier` | Leave output unchanged |
| `append` | `AppendOutputModifier` | Low-level text append |
| `replace` | `ReplaceOutputModifier` | Replace the complete output |
| `mapped` | `MappedOutputModifier` | Look up a response by input |
| `concept_replace` | `ConceptReplaceOutputModifier` | Replace concepts |
| `random_insertion` | `RandomInsertionModifier` | Insert text randomly |

Canonical task aliases:

| Task | Implementation |
|---|---|
| `malicious_injection` | `AppendOutputModifier` |
| `targeted_refusal` | `ReplaceOutputModifier` |
| `jailbreak` | `MappedOutputModifier` |
| `perception_hijack` | `MappedOutputModifier` |
| `concept_substitution` | `ConceptReplaceOutputModifier` |

Use canonical task names in public configs and generated data. The short names
are retained as reusable primitives.

## Usage

```python
from poisoning.modifiers import get_modifier, list_modifiers

print(list_modifiers())
modifier = get_modifier(
    "malicious_injection",
    append_text="[demo malicious injection]",
)
print(modifier.modify_output("A clean answer."))
```

Mapped modifiers require a JSON mapping path and a matching context:

```python
modifier = get_modifier("jailbreak", json_path="data/clean/example.json")
result = modifier.modify_output(
    "",
    context={"prompt": "Describe the image", "image_path": "images/example.jpg"},
)
```

Custom modifiers can be registered with `@register_modifier("name")` and must
implement the `OutputModifier` interface.
