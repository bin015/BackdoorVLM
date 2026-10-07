# Defenses

This directory keeps the reusable core of four defensive methods. Experiment
launchers, hard-coded machine paths, result reports, and duplicate batch
scripts are intentionally excluded.

| Method | Module | Core operation |
|---|---|---|
| Pruning | `pruning.py` | Wanda activation-aware sparsification |
| Quantization | `quantization.py` | bitsandbytes NF4 INT4 model loading |
| BYE | `bye.py` | Attention-entropy and GMM suspicious-sample detection |
| ZIP | `zip.py` | Blur/downsample plus diffusion image purification |

All methods are optional research tools. Their output should be evaluated with
the repository's inference and evaluation workflows rather than interpreted as
a security guarantee.

Optional dependencies:

- `bitsandbytes` for `quantization.py`;
- `scikit-learn` for `bye.py`;
- `diffusers` for `zip.py`.

ZIP follows the repository's simple experiment configuration by default:
Stable Diffusion v1.5, resize to `512x512`, Gaussian blur with
`blur_sigma=2.0`, downsampling factor `4`, DDIM with `20` inference steps,
and `strength=0.3`. The checkpoint can be overridden with
`--zip_model_id`; the remaining values are exposed as
`--zip_blur_sigma`, `--zip_downsample_scale`, `--zip_steps`, and
`--zip_strength`.

## Unified defended inference

[`inference.py`](./inference.py) applies one defense during Hugging Face
inference and writes the same prediction fields as
[`inference/hf_infer.py`](../inference/hf_infer.py). It additionally records
the method and BYE detection flag under `defense`.

```bash
python -m defense.inference \
  --model_name_or_path path/to/model \
  --input_json data/eval/example.json \
  --output_json results/predict/model_zip.json \
  --defense zip \
  --metric_task malicious_injection
```

Available methods are `none`, `pruning`, `quantization`, `bye`, and `zip`.
`pruning` uses the first samples as calibration data; `zip` purifies images
before generation; `bye` preserves every prediction and marks suspicious
samples instead of silently removing them. When `--metric_task` is supplied,
the selected metric from `evaluation/metrics.py` is written beside the
prediction file.

Fine-tuning-based defenses can be implemented through the same
`config/` + `llamafactory/` workflow: prepare the defense-specific training
data, select the corresponding training configuration, fine-tune the model,
and evaluate the resulting checkpoint with the standard inference and metrics
interfaces.
