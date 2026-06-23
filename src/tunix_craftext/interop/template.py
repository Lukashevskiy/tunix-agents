"""Framework-neutral checkpoint-to-JAX conversion templates.

Templates make every parameter rename and layout transformation reviewable.
They do not attempt to infer architecture from an opaque checkpoint; instead,
users declare explicit tensor rules and model templates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

ParameterTree: TypeAlias = "dict[str, jax.Array | ParameterTree]"
ParameterTreeLike: TypeAlias = "Mapping[str, ArrayLike | ParameterTreeLike]"


class ConversionError(ValueError):
    """The checkpoint and declared model template disagree.

    Example:
        >>> raise ConversionError("message")
    """


@dataclass(frozen=True)
class TensorRule:
    """One explicit source tensor → JAX PyTree leaf transformation.

    :ivar source: str
    :ivar target: tuple[str, ...]
    :ivar transform: str
    :ivar expected_shape: tuple[int, ...] | None

    Example:
        >>> obj = TensorRule(source=..., target=..., transform=...)
    """

    source: str
    target: tuple[str, ...]
    transform: str = "identity"
    expected_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.transform not in {"identity", "transpose_2d"}:
            raise ConversionError(f"unknown tensor transform: {self.transform}")
        if not self.target:
            raise ConversionError("target PyTree path cannot be empty")


@dataclass(frozen=True)
class ModelTemplate:
    """Architecture label and a complete, versionable mapping of its tensors.

    :ivar name: str
    :ivar rules: tuple[TensorRule, ...]
    :ivar source_format: str

    Example:
        >>> obj = ModelTemplate(name=..., rules=..., source_format=...)
    """

    name: str
    rules: tuple[TensorRule, ...]
    source_format: str = "generic-state-dict"

    def __post_init__(self) -> None:
        sources = [rule.source for rule in self.rules]
        targets = [rule.target for rule in self.rules]
        if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
            raise ConversionError("template source and target paths must be unique")


def _set_path(tree: ParameterTree, path: Sequence[str], value: jax.Array) -> None:
    """Set one JAX parameter leaf at a validated nested string path.

    :param tree: ParameterTree input value
    :param path: Sequence[str] input value
    :param value: jax.Array input value
    :returns: None

    Example:
        >>> _set_path(tree, path, value)
    """
    current: ParameterTree = tree
    for name in path[:-1]:
        child = current.setdefault(name, {})
        if not isinstance(child, dict):
            raise ConversionError(f"target path collision at {'.'.join(path)}")
        current = child
    current[path[-1]] = value


def _as_jax_array(value: ArrayLike) -> jax.Array:
    """Normalize one accepted external tensor into an immutable JAX array leaf.

    :param value: ArrayLike input value
    :returns: jax.Array

    Example:
        >>> result = _as_jax_array(value)
    """
    return jnp.asarray(value)


def normalize_parameter_tree(tree: ParameterTreeLike) -> ParameterTree:
    """Recursively normalize external parameter leaves into a JAX-only parameter tree.

    :param tree: ParameterTreeLike input value
    :returns: ParameterTree

    Example:
        >>> result = normalize_parameter_tree(tree)
    """
    normalized: ParameterTree = {}
    for name, value in tree.items():
        if isinstance(value, Mapping):
            normalized[name] = normalize_parameter_tree(value)
        else:
            normalized[name] = _as_jax_array(value)
    return normalized


def convert_state_dict(
    state_dict: Mapping[str, ArrayLike], template: ModelTemplate, *, strict: bool = True
) -> ParameterTree:
    """Convert named external tensors into a nested Flax/JAX-compatible parameter PyTree.

    Linear kernels need an explicit `transpose_2d` rule for the common PyTorch `[out, in]`
    to Flax `[in, out]` conversion. Extra checkpoint tensors fail in strict mode, preventing
    a partial or mismatched model from masquerading as a successful conversion.

    :param state_dict: Mapping[str, ArrayLike] input value
    :param template: ModelTemplate input value
    :param strict: bool input value
    :returns: ParameterTree

    Example:
        >>> result = convert_state_dict(state_dict, template)
    """
    result: ParameterTree = {}
    used: set[str] = set()
    for rule in template.rules:
        if rule.source not in state_dict:
            raise ConversionError(f"missing required source tensor: {rule.source}")
        value = jnp.asarray(state_dict[rule.source])
        if rule.transform == "transpose_2d":
            if value.ndim != 2:
                raise ConversionError(f"{rule.source} must be rank 2 for transpose_2d")
            value = value.T
        if rule.expected_shape is not None and tuple(value.shape) != rule.expected_shape:
            raise ConversionError(
                f"{rule.source} maps to {'.'.join(rule.target)} with shape {value.shape}; "
                f"expected {rule.expected_shape}"
            )
        _set_path(result, rule.target, _as_jax_array(value))
        used.add(rule.source)
    if strict:
        unexpected = sorted(set(state_dict) - used)
        if unexpected:
            raise ConversionError(f"unexpected source tensors: {', '.join(unexpected)}")
    return result
