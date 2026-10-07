# BackdoorVLM

> 🎉 Congratulations on the acceptance of our paper by **ACM MM 2026**!

BackdoorVLM is the official research codebase for constructing, training, and
evaluating backdoored vision-language models. It provides a complete workflow:

```text
prepare clean data
      ↓
construct poisoned data with poisoning/
      ↓
configure and train with config/ + llamafactory/
      ↓
run inference with inference/
      ↓
evaluate predictions with evaluation/
```

The repository is intended for reproducible research on VLM backdoor attacks,
triggers, target behaviors, and defensive evaluation. Use it only for
authorized security research and model evaluation.

## Repository structure

```text
backdoorVLM/
├── config/          # Representative training, inference, and evaluation configs
├── data/            # Dataset metadata and a minimal checked-in demo
├── evaluation/      # Prediction metrics and evaluation helpers
├── inference/       # Hugging Face and vLLM inference entry points
├── poisoning/       # Backdoor presets, triggers, modifiers, and data generation
├── llamafactory/    # LLaMA-Factory training backend and local integration
├── scripts/         # Representative data-generation scripts
└── dev/             # Small registry and development checks
```

The vendored LLaMA-Factory data path includes a small `sharegpt_meta`
integration. It preserves poisoning metadata from dataset JSON files through
conversion, preprocessing, and batch collation, so inference and evaluation
can inspect attack provenance. The training path retains the standard
LLaMA-Factory SFT objective and upstream loss options.

Generated checkpoints, logs, predictions, downloaded image collections, and
credentials are local artifacts and are excluded from the public source tree.

## Installation

```bash
conda create -n backdoorvlm python=3.12 pip -y
conda activate backdoorvlm
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Optional distributed-training and inference backends are listed in
[`requirements-optional.txt`](./requirements-optional.txt).

## 1. Construct poisoned data

The poisoning layer is independent of the training backend. A minimal
two-sample clean dataset is included for checking the public interfaces:

```bash
python scripts/generate_demo_data.py
```

This creates `data/demo/BadNets-T_demo_generated.json` from
`data/demo/clean_demo_2.json`. The script uses the public registry and the
canonical `malicious_injection` output modifier.

For the complete poisoning API and all registered presets, see
[`poisoning/README.md`](./poisoning/README.md).

## 2. Configure and train

Dataset entries used by LLaMA-Factory are declared in
[`data/dataset_info.json`](./data/dataset_info.json). The minimal example
[`config/llava1_5_demo_mix_sft.yaml`](./config/llava1_5_demo_mix_sft.yaml)
demonstrates clean + poisoned data mixing:

```yaml
dataset: backdoorvlm_demo_clean,backdoorvlm_demo_poison
```

After preparing the model and data paths, launch training through the bundled
LLaMA-Factory backend using its standard CLI, for example:

```bash
llamafactory-cli train config/llava1_5_demo_mix_sft.yaml
```

The demo config is for interface validation, not a production training recipe.
See [`config/README.md`](./config/README.md) and the upstream
[`llamafactory/README.md`](./llamafactory/README.md) for backend details.

## 3. Run inference

`inference/hf_infer.py` evaluates a Hugging Face vision-language model on a
registered dataset and writes predictions to an ignored local result folder:

```bash
python inference/hf_infer.py \
  --model_name_or_path llava-hf/llava-1.5-7b-hf \
  --dataset backdoorvlm_demo_poison \
  --dataset_dir data \
  --template llava \
  --max_samples 2
```

For vLLM-compatible models, use [`inference/vllm_infer.py`](./inference/vllm_infer.py)
after installing the optional backend.

## 4. Evaluate predictions

[`evaluation/metrics.py`](./evaluation/metrics.py) provides registry-based
metrics for utility and backdoor behavior, including:

- caption and VQA evaluation;
- `targeted_refusal`;
- `malicious_injection`;
- `perception_hijack`;
- `concept_substitution`;
- `jailbreak`.

Evaluation inputs and reference annotations must be prepared under `data/eval/`
or passed through the relevant script arguments. Do not commit generated
predictions or result files.

## Dataset

The processed dataset will be released separately on Hugging Face. The final
dataset URL will be added here after publication.

## Naming conventions

The public attack names are the registry names in `poisoning/`:
`BadNets-T`, `BadNets-MT`, `AddSent`, `BadNets-I`, `Blended`, `SIG`,
`ImgTrojan`, `BadNets-MM`, `Dual-Key`, `VL-Trojan`, `MABA`, and `Shadowcast`.

Canonical task names are:

- `malicious_injection`
- `targeted_refusal`
- `jailbreak`
- `perception_hijack`
- `concept_substitution`

## Citation and license

Please cite the accompanying ACM MM 2026 paper when using this repository.
The source code is released under the [Apache License 2.0](./LICENSE).
The `llamafactory/` directory retains its upstream license and attribution.

## Responsible use

This project is provided for authorized security research, robustness
evaluation, and defense. Do not use it to compromise systems, data, or models
without permission.
