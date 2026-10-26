# Configuration files

This directory contains representative configurations for the public
workflow. They are intentionally small in number; experiment-specific
variants remain in the internal working repository.

Before running a configuration, verify:

- `model_name_or_path` points to a local model directory or a supported model
  identifier;
- dataset names exist in `data/dataset_info.json`;
- image and annotation paths are available under `data/`;
- output paths point to a local, ignored directory.

Configurations are being migrated away from machine-specific absolute paths.
Do not add credentials, tokens, checkpoints, or host-specific storage paths to
these files.

The smallest end-to-end example is
[`llava1_5_demo_mix_sft.yaml`](./llava1_5_demo_mix_sft.yaml). It registers the
two checked-in demo datasets and mixes them through a comma-separated
`dataset` value:

```yaml
dataset: backdoorvlm_demo_clean,backdoorvlm_demo_poison
```

The example is intended for interface and data-format validation. It is not a
recommended training recipe for a production model.

`llava1_5_full_sft_clean.yaml` and its evaluation variant select the separate
20k clean training set and two epochs, matching the paper's LLaVA training
schedule. Their batch size of 16 per device reaches the paper's global batch
size of 128 only on eight devices; adjust accumulation for other hardware.
