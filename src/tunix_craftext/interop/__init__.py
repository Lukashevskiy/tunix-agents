"""Explicit, testable conversion of external neural-network weights to JAX PyTrees.

The package provides utilities for mapping vendor checkpoint tensors and LoRA
adapters into Flax/JAX parameter trees with explicit layout and shape checks.
"""

from .lora import LoraAdapter, merge_lora_adapters
from .template import ConversionError, ModelTemplate, TensorRule, convert_state_dict

__all__ = [
    "ConversionError",
    "LoraAdapter",
    "ModelTemplate",
    "TensorRule",
    "convert_state_dict",
    "merge_lora_adapters",
]
