# BackdoorVLM

> 🎉 Congratulations on the acceptance of our paper by **ACM MM 2026**!

**BackdoorVLM** is a benchmark and research codebase for studying backdoor
attacks and defenses on vision-language models (VLMs). The associated paper
organizes multimodal backdoor behavior across image captioning and visual
question answering, with textual, visual, and bimodal triggers:

- 5 threat categories: targeted refusal, malicious injection, jailbreak,
  concept substitution, and perceptual hijack;
- 12 representative attack methods;
- experiments with 2 open-source VLMs and 3 multimodal datasets;
- evaluation of 5 representative defense methods.

This repository provides the public data-construction, training, inference,
evaluation, and defense implementations used to support that benchmark. It
also provides the components needed to study fine-tuning-based defenses by
reusing the same training and data-mixing workflow.

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
├── defense/         # Core pruning, quantization, BYE, and ZIP defenses
├── llamafactory/    # LLaMA-Factory training backend and local integration
├── scripts/         # Representative data-generation scripts
└── dev/             # Small registry and development checks
```

The vendored LLaMA-Factory data path includes a small `sharegpt_meta`
integration. It preserves poisoning metadata from dataset JSON files through
conversion, preprocessing, and batch collation, so inference and evaluation
can inspect attack provenance. The training path retains the standard
LLaMA-Factory SFT objective and upstream loss options.

The GitHub checkout contains only the small `data/demo/` package. Full clean,
poisoned, evaluation, and image datasets are released separately on
Hugging Face and must be downloaded into `data/` before full training or
evaluation. Generated checkpoints, logs, predictions, and credentials remain
local artifacts.

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
two-sample clean dataset is included for checking the public interfaces. The
same samples can be used to preview different task modifiers:

```bash
python scripts/generate_demo_data.py
python scripts/generate_demo_data.py --task targeted_refusal \
  --output data/demo/targeted_refusal_demo.json
python scripts/generate_demo_data.py --task concept_substitution \
  --output data/demo/concept_substitution_demo.json
```

This creates `data/demo/BadNets-T_demo_generated.json` from
`data/demo/clean_demo_2.json` by default. The task-specific commands reserve
small examples for targeted refusal and concept substitution; the same
`--task` option supports `malicious_injection`, `targeted_refusal`, `jailbreak`,
`perception_hijack`, and `concept_substitution`.

Task-oriented 1k demo files for all five task types are also included under
`data/demo/`, alongside trigger-focused attack examples.

For the complete poisoning API and all registered presets, see
[`poisoning/README.md`](./poisoning/README.md).

## 2. Configure and train

For the two-record interface demo, use the checked-in datasets directly. For
full training, first download the BackdoorVLM data package from Hugging Face
and merge its `clean/`, `poison/`, `eval/`, and image directories into
`data/`. The dataset names in `data/dataset_info.json` are the registry used
by the training configurations.

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

Defended inference uses the same prediction schema and can collect metrics in
one command:

```bash
python -m defense.inference \
  --model_name_or_path path/to/model \
  --input_json data/eval/example.json \
  --output_json results/predict/model_zip_malicious_injection.json \
  --defense zip \
  --metric_task malicious_injection
```

See [`defense/README.md`](./defense/README.md) for the available pruning,
quantization, BYE, ZIP, and fine-tuning-based defense workflows.

Evaluation inputs and reference annotations are provided by the Hugging Face
data package under `data/eval/`. Do not commit generated predictions or result
files.

## Dataset

The complete processed dataset will be released separately on Hugging Face.
After downloading it, merge its contents into `data/`; the GitHub repository
intentionally keeps only the small demo package. The upload package uses the
canonical 1k training and poisoning variants, while smaller or larger
poisoning rates can be reproduced by selecting a different number of samples
during training.

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

The source code is released under the [Apache License 2.0](./LICENSE).
The `llamafactory/` directory retains its upstream license and attribution.

## Responsible use

This project is provided for authorized security research, robustness
evaluation, and defense. Do not use it to compromise systems, data, or models
without permission.
