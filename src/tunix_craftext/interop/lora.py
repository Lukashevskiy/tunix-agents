"""LoRA merge operations with explicit matrix layout and shape checks.

This module handles LoRA adapter deltas and merges them into existing JAX
parameter PyTrees while preserving Flax kernel orientation and avoiding
in-place mutation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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

    target: tuple[str, ...]
    down: ArrayLike
    up: ArrayLike
    alpha: float

    def delta(self) -> jax.Array:
        """Return the scaled adapter delta in Flax kernel orientation as ``jax.Array``.

        :returns: jax.Array

        Example:
            >>> result = adapter.delta()
        """
        down, up = jnp.asarray(self.down), jnp.asarray(self.up)
        if down.ndim != 2 or up.ndim != 2:
            raise ConversionError("LoRA up/down tensors must both be rank 2")
        rank, input_size = down.shape
        output_size, up_rank = up.shape
        if rank == 0 or up_rank != rank:
            raise ConversionError("LoRA rank must be non-zero and agree between up/down")
        return ((up @ down) * (self.alpha / rank)).T.reshape(input_size, output_size)


def _get_path(tree: ParameterTree, path: Sequence[str]) -> jax.Array:
    """Read one normalized JAX parameter leaf from a nested tree.

    :param tree: ParameterTree input value
    :param path: Sequence[str] input value
    :returns: jax.Array

    Example:
        >>> result = _get_path(tree, path)
    """
    current: jax.Array | ParameterTree = tree
    for name in path:
        if not isinstance(current, dict) or name not in current:
            raise ConversionError(f"LoRA target does not exist: {'.'.join(path)}")
        current = current[name]
    if not isinstance(current, jax.Array):
        raise ConversionError(f"LoRA target must be an array leaf: {'.'.join(path)}")
    return current


def _copy_and_set(tree: ParameterTree, path: Sequence[str], value: jax.Array) -> ParameterTree:
    """Return a tree copy with one JAX parameter leaf replaced.

    :param tree: ParameterTree input value
    :param path: Sequence[str] input value
    :param value: jax.Array input value
    :returns: ParameterTree

    Example:
        >>> result = _copy_and_set(tree, path, value)
    """
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


def merge_lora_adapters(
    params: Mapping[str, ArrayLike | ParameterTreeLike], adapters: Sequence[LoraAdapter]
) -> ParameterTree:
    """Return a new parameter PyTree with one or more adapters merged, never mutating input.

    :param params: Mapping[str, ArrayLike | ParameterTreeLike] input value
    :param adapters: Sequence[LoraAdapter] input value
    :returns: ParameterTree

    Example:
        >>> result = merge_lora_adapters(params, adapters)
    """
    merged = normalize_parameter_tree(params)
    for adapter in adapters:
        base = _get_path(merged, adapter.target)
        delta = adapter.delta()
        if base.shape != delta.shape:
            raise ConversionError(
                f"LoRA delta for {'.'.join(adapter.target)} has shape {delta.shape}; "
                f"base is {base.shape}"
            )
        merged = _copy_and_set(merged, adapter.target, base + delta)
    return merged
