"""Wanda pruning for the language component of a loaded VLM.

Wanda scores weights with activation magnitude collected from calibration
inputs. This module contains only the reusable model transformation; dataset
loading and evaluation remain in the inference/evaluation layers.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


def _linear_layers(module: nn.Module) -> dict[str, nn.Linear]:
    return {name: child for name, child in module.named_modules() if isinstance(child, nn.Linear)}


def _language_layers(model: nn.Module) -> Iterable[nn.Module]:
    pending = [model]
    visited = set()
    while pending:
        candidate = pending.pop(0)
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        layers = getattr(candidate, "layers", None)
        if isinstance(layers, (nn.ModuleList, list, tuple)):
            return layers
        for attribute in ("language_model", "text_model", "model"):
            child = getattr(candidate, attribute, None)
            if isinstance(child, nn.Module):
                pending.append(child)
    raise ValueError("Could not find transformer language layers on the model.")


@torch.no_grad()
def apply_wanda_pruning(
    model: nn.Module,
    calibration_batches: Iterable[dict[str, Any]],
    sparsity_ratio: float = 0.5,
) -> nn.Module:
    """Prune the lowest Wanda-scored weights in every language linear layer.

    ``calibration_batches`` must contain inputs accepted by ``model``. The
    batches are used only for activation statistics and are not modified.
    """
    if not 0.0 <= sparsity_ratio < 1.0:
        raise ValueError("sparsity_ratio must be in the interval [0, 1).")

    model.eval()
    stats: dict[int, dict[str, torch.Tensor]] = {}
    handles = []
    layers = list(_language_layers(model))

    for layer_index, layer in enumerate(layers):
        stats[layer_index] = {}
        for name, module in _linear_layers(layer).items():
            stats[layer_index][name] = torch.zeros(module.in_features, device=module.weight.device)

            def collect(_module, inputs, _output, key=(layer_index, name)):
                values = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                stats[key[0]][key[1]] += values.square().sum(dim=0)

            handles.append(module.register_forward_hook(collect))

    try:
        for batch in calibration_batches:
            model(**batch)
    finally:
        for handle in handles:
            handle.remove()

    for layer_index, layer in enumerate(layers):
        for name, module in _linear_layers(layer).items():
            metric = module.weight.detach().abs() * stats[layer_index][name].sqrt().unsqueeze(0)
            count = int(metric.shape[1] * sparsity_ratio)
            if count:
                indices = metric.topk(count, dim=1, largest=False).indices
                module.weight.scatter_(1, indices, 0.0)
    return model


def check_sparsity(model: nn.Module) -> float:
    """Return the fraction of zero weights in language linear layers."""
    weights = [module.weight.detach() for layer in _language_layers(model) for module in _linear_layers(layer).values()]
    if not weights:
        raise ValueError("No language linear layers found.")
    total = sum(weight.numel() for weight in weights)
    zeros = sum(int(weight.eq(0).sum()) for weight in weights)
    return zeros / total
