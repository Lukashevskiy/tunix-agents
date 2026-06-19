"""Explicit, testable conversion of external neural-network weights to JAX PyTrees."""

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
