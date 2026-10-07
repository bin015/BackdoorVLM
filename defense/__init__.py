"""Core defensive methods for VLM backdoor experiments.

The modules are intentionally model- and dataset-agnostic where possible:
pruning and quantization operate on a loaded model, BYE detects suspicious
samples from attention statistics, and ZIP purifies input images.
"""

from .pruning import apply_wanda_pruning, check_sparsity
from .quantization import load_int4_model, count_quantized_layers

__all__ = [
    "apply_wanda_pruning",
    "check_sparsity",
    "load_int4_model",
    "count_quantized_layers",
]
