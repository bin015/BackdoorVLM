# Data layout

This directory is the data contract used by the poisoning, training,
evaluation, and inference workflows.

## Directory structure

```text
data/
├── dataset_info.json
└── demo/       # checked-in examples and their small image assets
```

GitHub contains only the small demo package. The complete `clean/`, `poison/`,
and `eval/` datasets, together with their larger image collections, are
released separately on Hugging Face. Download that package and merge its
contents into `data/` before running full training or evaluation. Preserve the
relative paths recorded in `dataset_info.json`.

## Conversation format

The training datasets use the ShareGPT-style multimodal structure expected by
LLaMA-Factory:

```json
{
  "id": "example-id",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe the image."},
    {"from": "gpt", "value": "A sample answer."}
  ],
  "images": ["images/example.jpg"]
}
```

The `images` entries are paths relative to the dataset root used by the
training configuration. Do not replace them with machine-specific absolute
paths.

## Task names

Generated experiments use the following canonical task names:

- `malicious_injection`: malicious output injection;
- `perception_hijack`: target-image perception manipulation;
- `concept_substitution`: misidentification through concept substitution;
- `targeted_refusal`: targeted refusal behavior;
- `jailbreak`: jailbreak behavior.

The checked-in demo files and dataset metadata use these canonical names.

## Poisoning metadata

Poisoned records may include a `metadata` object in addition to the standard
ShareGPT fields. The metadata describes how a record was constructed and is
kept separate from the model conversation:

```json
{
  "metadata": {
    "poisoned": true,
    "target_modalities": "TEXT",
    "attack_type": "BadNets-T",
    "modality_type": "TEXT",
    "trigger_generator": "RandomInsertionTrigger",
    "output_modifier": "MaliciousInjectionModifier"
  }
}
```

The vendored LLaMA-Factory integration registers the `sharegpt_meta` format
and carries this object through dataset conversion, preprocessing, and batch
collation. This metadata is available to downstream inference and evaluation
code, while the SFT objective remains the standard LLaMA-Factory objective.

## Dataset metadata

[`dataset_info.json`](./dataset_info.json) maps dataset names used by training
configuration files to their files and field mappings. When adding or replacing a processed dataset, keep its metadata entry and
verify that every referenced file exists after the dataset package is unpacked.

## Data sources and redistribution

Image collections and benchmark annotations may have their own licenses and
download terms. Check the original source terms before redistributing the full
dataset. The source code repository is not a substitute for those terms.

## Minimal demo

The checked-in [`demo/clean_demo_2.json`](./demo/clean_demo_2.json) contains
two multimodal records and is small enough for interface validation. It is an
interface-only fixture and is not part of the 1k dataset upload. The
representative attack demos use the 1k naming convention. In addition to
trigger-focused examples, task-oriented demos are included for
`malicious_injection`, `targeted_refusal`, `jailbreak`, `perception_hijack`,
and `concept_substitution`, so their output modifiers can be inspected
without running a model. Generate a tiny poisoned counterpart with:

```bash
python scripts/generate_demo_data.py
```

The generated file is registered as `backdoorvlm_demo_poison` in
`dataset_info.json`. The corresponding training example is
[`config/llava1_5_demo_mix_sft.yaml`](../config/llava1_5_demo_mix_sft.yaml).

The complete poison package uses the canonical `1k/` variants. Other
sample-count variants are retained in the internal working repository backup
for experiment recovery and are not part of the GitHub checkout.
