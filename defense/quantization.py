"""Post-training INT4 loading with bitsandbytes.

Quantization is applied when loading a checkpoint. This module deliberately
does not contain model evaluation or experiment-directory scanning.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig


def load_int4_model(model_name_or_path: str, device_map: str | dict = "auto"):
    """Load a vision-language checkpoint using NF4 4-bit quantization."""
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        quantization_config=config,
        device_map=device_map,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )


def count_quantized_layers(model) -> tuple[int, int]:
    """Return ``(quantized_layers, layers_with_weights)`` for a loaded model."""
    quantized = total = 0
    for module in model.modules():
        if hasattr(module, "weight"):
            total += 1
            if hasattr(module.weight, "quant_state"):
                quantized += 1
    return quantized, total
