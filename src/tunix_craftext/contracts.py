"""Stable, framework-neutral data contracts at the environment/trainer boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
@dataclass(frozen=True)
class Transition(Generic[ObservationT, ActionT]):
    """Represent one synchronous environment step with a leading batch axis.

    :ivar observation: Batched observation PyTree.
    :ivar action: Batched action selected from ``observation``.
    :ivar reward: Per-environment reward with shape ``[B]``.
    :ivar terminated: True terminal flags with shape ``[B]``.
    :ivar truncated: Timeout or externally truncated flags with shape ``[B]``.
    :ivar log_prob: Action log-probabilities with shape ``[B]``.
    :ivar value: Critic predictions with shape ``[B]``.
    """

    observation: ObservationT
    action: ActionT
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    log_prob: jax.Array
    value: jax.Array

    @property
    def done(self) -> jax.Array:
        """Return the JAX-compatible terminal-or-truncated mask with shape ``[T, B]``."""
        return jnp.logical_or(self.terminated, self.truncated)


@dataclass(frozen=True)
class RolloutBatch(Generic[ObservationT, ActionT]):
    """A time-major `[T, B, ...]` rollout plus bootstrap values `[B]`."""

    transitions: Transition[ObservationT, ActionT]
    bootstrap_value: jax.Array

    def validate(self) -> None:
        """Validate the mandatory time-major rollout axes without host array conversion.

        This is a host-side boundary check; do not call it within ``jax.jit``. Every leaf in the
        observation/action/value PyTrees must begin with the same ``[T, B]`` axes as reward.

        :raises ValueError: If a field leaf lacks shape metadata, has incompatible leading axes,
            or bootstrap value is not batch-major ``[B, ...]``.
        """
        reward_shape = _shape_of(self.transitions.reward, "reward")
        if len(reward_shape) != 2:
            raise ValueError("reward must be time-major with shape [T, B]")
        expected_rollout_axes = reward_shape
        for name in ("observation", "action", "terminated", "truncated", "log_prob", "value"):
            _validate_tree_axes(getattr(self.transitions, name), name, expected_rollout_axes)
        _validate_tree_axes(self.bootstrap_value, "bootstrap_value", expected_rollout_axes[1:])


def _shape_of(value: object, field_name: str) -> tuple[int, ...]:
    """Read static shape metadata from one NumPy/JAX leaf without materializing it on host."""
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"{field_name} leaf must expose static shape metadata")
    return tuple(shape)


def _validate_tree_axes(value: object, field_name: str, expected_axes: tuple[int, ...]) -> None:
    """Ensure every JAX PyTree leaf begins with the contract's declared axes."""
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves:
        raise ValueError(f"{field_name} must contain at least one array leaf")
    for index, leaf in enumerate(leaves):
        shape = _shape_of(leaf, field_name)
        if tuple(shape[: len(expected_axes)]) != expected_axes:
            raise ValueError(f"{field_name} leaf {index} must begin with axes {expected_axes}, got {shape}")


def _register_contract_pytrees() -> None:
    """Register public contract dataclasses at their definition boundary for JIT returns."""
    jax.tree_util.register_dataclass(
        Transition,
        data_fields=["observation", "action", "reward", "terminated", "truncated", "log_prob", "value"],
        meta_fields=[],
    )
    jax.tree_util.register_dataclass(RolloutBatch, data_fields=["transitions", "bootstrap_value"], meta_fields=[])


_register_contract_pytrees()
