"""LoRA merge operations with explicit matrix layout and shape checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from .template import ConversionError


@dataclass(frozen=True)
class LoraAdapter:
    """A standard LoRA delta for a JAX/Flax linear kernel stored as `[in_features, out_features]`.

    `down` is `[rank, in_features]` and `up` is `[out_features, rank]`, matching common
    PEFT/PyTorch tensors. The returned delta is transposed into Flax kernel layout.
    """

    target: Tuple[str, ...]
    down: Any
    up: Any
    alpha: float

    def delta(self) -> np.ndarray:
        down, up = np.asarray(self.down), np.asarray(self.up)
        if down.ndim != 2 or up.ndim != 2:
            raise ConversionError("LoRA up/down tensors must both be rank 2")
        rank, input_size = down.shape
        output_size, up_rank = up.shape
        if rank == 0 or up_rank != rank:
            raise ConversionError("LoRA rank must be non-zero and agree between up/down")
        return ((up @ down) * (self.alpha / rank)).T.reshape(input_size, output_size)


def _get_path(tree: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = tree
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            raise ConversionError(f"LoRA target does not exist: {'.'.join(path)}")
        current = current[name]
    return current


def _copy_and_set(tree: Mapping[str, Any], path: Sequence[str], value: Any) -> dict[str, Any]:
    result = dict(tree)
    current = result
    for name in path[:-1]:
        child = current.get(name)
        if not isinstance(child, Mapping):
            raise ConversionError(f"LoRA target does not exist: {'.'.join(path)}")
        copied = dict(child)
        current[name] = copied
        current = copied
    current[path[-1]] = value
    return result


def merge_lora_adapters(params: Mapping[str, Any], adapters: Sequence[LoraAdapter]) -> dict[str, Any]:
    """Return a new parameter PyTree with one or more adapters merged, never mutating input."""
    merged: dict[str, Any] = dict(params)
    for adapter in adapters:
        base = np.asarray(_get_path(merged, adapter.target))
        delta = adapter.delta()
        if base.shape != delta.shape:
            raise ConversionError(
                f"LoRA delta for {'.'.join(adapter.target)} has shape {delta.shape}; base is {base.shape}"
            )
        merged = _copy_and_set(merged, adapter.target, base + delta)
    return merged
