"""Framework-neutral checkpoint-to-JAX conversion templates.

Templates make every parameter rename and layout transformation reviewable.  They do not
attempt to infer architecture from an opaque checkpoint: silent guesses are how a model
can appear to load while producing nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


class ConversionError(ValueError):
    """The checkpoint and declared model template disagree."""


@dataclass(frozen=True)
class TensorRule:
    """One explicit source tensor → JAX PyTree leaf transformation."""

    source: str
    target: Tuple[str, ...]
    transform: str = "identity"
    expected_shape: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.transform not in {"identity", "transpose_2d"}:
            raise ConversionError(f"unknown tensor transform: {self.transform}")
        if not self.target:
            raise ConversionError("target PyTree path cannot be empty")


@dataclass(frozen=True)
class ModelTemplate:
    """Architecture label and a complete, versionable mapping of its tensors."""

    name: str
    rules: Tuple[TensorRule, ...]
    source_format: str = "generic-state-dict"

    def __post_init__(self) -> None:
        sources = [rule.source for rule in self.rules]
        targets = [rule.target for rule in self.rules]
        if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
            raise ConversionError("template source and target paths must be unique")


def _set_path(tree: dict[str, Any], path: Sequence[str], value: Any) -> None:
    current = tree
    for name in path[:-1]:
        child = current.setdefault(name, {})
        if not isinstance(child, dict):
            raise ConversionError(f"target path collision at {'.'.join(path)}")
        current = child
    current[path[-1]] = value


def _as_jax_compatible(value: np.ndarray) -> Any:
    """Use JAX arrays when available without making converter unit tests require JAX."""
    try:
        import jax.numpy as jnp

        return jnp.asarray(value)
    except ImportError:
        return value


def convert_state_dict(
    state_dict: Mapping[str, Any], template: ModelTemplate, *, strict: bool = True
) -> dict[str, Any]:
    """Convert named external tensors into a nested Flax/JAX-compatible parameter PyTree.

    Linear kernels need an explicit `transpose_2d` rule for the common PyTorch `[out, in]`
    to Flax `[in, out]` conversion. Extra checkpoint tensors fail in strict mode, preventing
    a partial or mismatched model from masquerading as a successful conversion.
    """
    result: dict[str, Any] = {}
    used: set[str] = set()
    for rule in template.rules:
        if rule.source not in state_dict:
            raise ConversionError(f"missing required source tensor: {rule.source}")
        value = np.asarray(state_dict[rule.source])
        if rule.transform == "transpose_2d":
            if value.ndim != 2:
                raise ConversionError(f"{rule.source} must be rank 2 for transpose_2d")
            value = value.T
        if rule.expected_shape is not None and tuple(value.shape) != rule.expected_shape:
            raise ConversionError(
                f"{rule.source} maps to {'.'.join(rule.target)} with shape {value.shape}; "
                f"expected {rule.expected_shape}"
            )
        _set_path(result, rule.target, _as_jax_compatible(value))
        used.add(rule.source)
    if strict:
        unexpected = sorted(set(state_dict) - used)
        if unexpected:
            raise ConversionError(f"unexpected source tensors: {', '.join(unexpected)}")
    return result
