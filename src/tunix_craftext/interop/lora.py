"""LoRA merge operations with explicit matrix layout and shape checks."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .template import ConversionError, ParameterTree, ParameterTreeLike, normalize_parameter_tree


@dataclass(frozen=True)
class LoraAdapter:
    """A standard LoRA delta for a JAX/Flax linear kernel stored as `[in_features, out_features]`.

    `down` is `[rank, in_features]` and `up` is `[out_features, rank]`, matching common
    PEFT/PyTorch tensors. The returned delta is transposed into Flax kernel layout.
    """

    target: Tuple[str, ...]
    down: ArrayLike
    up: ArrayLike
    alpha: float

    def delta(self) -> jax.Array:
        """Return the scaled adapter delta in Flax kernel orientation as ``jax.Array``."""
        down, up = jnp.asarray(self.down), jnp.asarray(self.up)
        if down.ndim != 2 or up.ndim != 2:
            raise ConversionError("LoRA up/down tensors must both be rank 2")
        rank, input_size = down.shape
        output_size, up_rank = up.shape
        if rank == 0 or up_rank != rank:
            raise ConversionError("LoRA rank must be non-zero and agree between up/down")
        return ((up @ down) * (self.alpha / rank)).T.reshape(input_size, output_size)


def _get_path(tree: ParameterTree, path: Sequence[str]) -> jax.Array:
    """Read one normalized JAX parameter leaf from a nested tree."""
    current: jax.Array | ParameterTree = tree
    for name in path:
        if not isinstance(current, dict) or name not in current:
            raise ConversionError(f"LoRA target does not exist: {'.'.join(path)}")
        current = current[name]
    if not isinstance(current, jax.Array):
        raise ConversionError(f"LoRA target must be an array leaf: {'.'.join(path)}")
    return current


def _copy_and_set(tree: ParameterTree, path: Sequence[str], value: jax.Array) -> ParameterTree:
    """Return a tree copy with one JAX parameter leaf replaced."""
    result = dict(tree)
    current: ParameterTree = result
    for name in path[:-1]:
        child = current.get(name)
        if not isinstance(child, dict):
            raise ConversionError(f"LoRA target does not exist: {'.'.join(path)}")
        copied = dict(child)
        current[name] = copied
        current = copied
    current[path[-1]] = value
    return result


def merge_lora_adapters(params: ParameterTreeLike, adapters: Sequence[LoraAdapter]) -> ParameterTree:
    """Return a new parameter PyTree with one or more adapters merged, never mutating input."""
    merged = normalize_parameter_tree(params)
    for adapter in adapters:
        base = _get_path(merged, adapter.target)
        delta = adapter.delta()
        if base.shape != delta.shape:
            raise ConversionError(
                f"LoRA delta for {'.'.join(adapter.target)} has shape {delta.shape}; base is {base.shape}"
            )
        merged = _copy_and_set(merged, adapter.target, base + delta)
    return merged
