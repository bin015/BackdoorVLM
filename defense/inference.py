"""Run Hugging Face inference with an optional defense and collect metrics.

The prediction schema intentionally matches ``inference/hf_infer.py``:
``prompt``, ``predict``, ``label``, ``image`` and ``metadata`` are preserved.
Defense-specific information is stored under ``defense``. The script accepts
the repository's ShareGPT JSON files directly and does not require a model
checkpoint to be rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .bye import detect_suspicious_samples, extract_attention_entropy
from .pruning import apply_wanda_pruning
from .quantization import load_int4_model
from .zip import DEFAULT_ZIP_MODEL_ID, ZIPPurifier


def _prompt(sample: dict[str, Any], processor) -> str:
    text = sample["conversations"][0]["value"].replace("<image>", "").strip()
    image_count = len(sample.get("images") or [])
    if hasattr(processor, "apply_chat_template"):
        content = [{"type": "image"} for _ in range(image_count)] + [{"type": "text", "text": text}]
        try:
            return processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
            )
        except TypeError:
            # LLaVA-style templates expect the message content to be a string.
            plain_content = "<image>\n" * image_count + text
            return processor.apply_chat_template(
                [{"role": "user", "content": plain_content}], add_generation_prompt=True, tokenize=False
            )
    return text


def _image_paths(sample: dict[str, Any], data_folder: Path) -> list[Path]:
    return [data_folder / path for path in sample.get("images", [])]


def _load_model(model_name_or_path: str, defense: str, device_map: str):
    if defense == "quantization":
        return load_int4_model(model_name_or_path, device_map=device_map)
    return AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype="auto",
        trust_remote_code=True,
    )


def _move_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def run_defended_inference(
    model_name_or_path: str,
    input_json: str,
    output_json: str,
    data_folder: str = "data",
    defense: str = "none",
    metric_task: str | None = None,
    metric_output: str | None = None,
    max_samples: int | None = None,
    max_new_tokens: int = 512,
    batch_size: int = 1,
    sparsity_ratio: float = 0.5,
    device_map: str = "auto",
    zip_model_id: str = DEFAULT_ZIP_MODEL_ID,
    zip_blur_sigma: float = 2.0,
    zip_downsample_scale: int = 4,
    zip_steps: int = 20,
    zip_strength: float = 0.3,
) -> dict[str, Any]:
    """Generate predictions, apply one defense, and optionally compute metrics."""
    if defense not in {"none", "pruning", "quantization", "bye", "zip"}:
        raise ValueError("defense must be one of: none, pruning, quantization, bye, zip")

    with open(input_json, encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Input JSON must contain a non-empty list: {input_json}")
    samples = samples[:max_samples] if max_samples is not None else samples

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = _load_model(model_name_or_path, defense, device_map)
    model.eval()
    device = next(model.parameters()).device
    data_root = Path(data_folder)
    purifier = None
    if defense == "zip":
        purifier = ZIPPurifier(
            model_id=zip_model_id,
            device=str(device),
            blur_sigma=zip_blur_sigma,
            downsample_scale=zip_downsample_scale,
            steps=zip_steps,
            strength=zip_strength,
        )

    prepared: list[dict[str, Any]] = []
    for sample in samples:
        paths = _image_paths(sample, data_root)
        images = [Image.open(path).convert("RGB") for path in paths]
        if purifier is not None:
            images = [purifier.purify(image) for image in images]
        prepared.append(
            {
                "sample": sample,
                "prompt": _prompt(sample, processor),
                "images": images or None,
            }
        )

    def make_inputs(item: dict[str, Any]) -> dict[str, Any]:
        return dict(
            processor(
                text=item["prompt"],
                images=item["images"],
                return_tensors="pt",
                padding=True,
            )
        )

    if defense == "pruning":
        calibration = [_move_inputs(make_inputs(item), device) for item in prepared[: min(8, len(prepared))]]
        apply_wanda_pruning(model, calibration, sparsity_ratio=sparsity_ratio)

    suspicious: set[int] = set()
    sensitive_layers: list[int] = []
    if defense == "bye":
        bye_inputs = [_move_inputs(make_inputs(item), device) for item in prepared]
        entropy = extract_attention_entropy(model, bye_inputs, image_token_start=0, vision_token_count=576)
        indices, sensitive_layers = detect_suspicious_samples(entropy)
        suspicious = set(indices)

    predictions: list[dict[str, Any]] = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        merged = _move_inputs(
            dict(
                processor(
                    text=[item["prompt"] for item in batch],
                    images=[item["images"] for item in batch],
                    return_tensors="pt",
                    padding=True,
                )
            ),
            device,
        )
        with torch.no_grad():
            generated = model.generate(**merged, max_new_tokens=max_new_tokens, do_sample=False)
        input_length = merged["input_ids"].shape[1]
        decoded = processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)
        for offset, (item, prediction) in enumerate(zip(batch, decoded)):
            sample_index = start + offset
            sample = item["sample"]
            predictions.append(
                {
                    "prompt": item["prompt"],
                    "predict": prediction.strip(),
                    "label": sample["conversations"][1]["value"],
                    "image": sample.get("images"),
                    "metadata": sample.get("metadata"),
                    "defense": {
                        "method": defense,
                        "suspicious": sample_index in suspicious,
                        "sensitive_layers": sensitive_layers,
                    },
                }
            )

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)

    result = {"predictions": str(output_path), "num_samples": len(predictions)}
    if metric_task:
        from evaluation.metrics import process_file

        metrics = process_file(str(output_path), metric_task)
        metric_path = Path(metric_output) if metric_output else output_path.with_name(output_path.stem + "_metrics.json")
        with metric_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        result.update({"metric_task": metric_task, "metrics": str(metric_path)})
    return result


if __name__ == "__main__":
    import fire

    fire.Fire(run_defended_inference)
