"""BYE (Believe Your Eyes) attention-entropy detection core.

The detector extracts attention entropy over image-token regions, identifies
layers with bimodal entropy distributions, and clusters suspicious samples.
Model-specific prompt construction and data loading are caller responsibilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _gmm():
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError(
            "BYE requires scikit-learn. Install it with `pip install scikit-learn`."
        ) from exc
    return GaussianMixture


@torch.no_grad()
def extract_attention_entropy(
    model,
    model_inputs: list[dict[str, Any]],
    image_token_start: int,
    vision_token_count: int,
) -> np.ndarray:
    """Return one entropy vector per input batch item."""
    rows = []
    model.eval()
    for inputs in model_inputs:
        attentions = model(**inputs, output_attentions=True, return_dict=True).attentions
        values = []
        for attention in attentions:
            distribution = attention[0, :, -1, image_token_start : image_token_start + vision_token_count]
            distribution = distribution.mean(dim=0).float()
            distribution = distribution / distribution.sum().clamp_min(1e-8)
            values.append(float(-(distribution * distribution.clamp_min(1e-8).log()).sum()))
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def detect_suspicious_samples(
    entropy_matrix: np.ndarray,
    bimodal_threshold: float = 2.0,
) -> tuple[list[int], list[int]]:
    """Return suspicious sample indices and entropy-sensitive layer indices."""
    if entropy_matrix.ndim != 2 or entropy_matrix.shape[0] < 2:
        raise ValueError("entropy_matrix must contain at least two samples.")
    sensitive_layers = []
    gaussian_mixture = _gmm()
    for layer in range(entropy_matrix.shape[1]):
        values = entropy_matrix[:, layer].reshape(-1, 1)
        gmm = gaussian_mixture(n_components=2, random_state=0).fit(values)
        means = gmm.means_.ravel()
        stds = np.sqrt(gmm.covariances_.ravel())
        if abs(means[0] - means[1]) > bimodal_threshold * (stds[0] + stds[1]):
            sensitive_layers.append(layer)
    if not sensitive_layers:
        return [], []
    scores = entropy_matrix[:, sensitive_layers].mean(axis=1)
    labels = gaussian_mixture(n_components=2, random_state=0).fit_predict(scores[:, None])
    means = [scores[labels == label].mean() for label in (0, 1)]
    suspicious_label = int(np.argmin(means))
    return np.flatnonzero(labels == suspicious_label).tolist(), sensitive_layers
