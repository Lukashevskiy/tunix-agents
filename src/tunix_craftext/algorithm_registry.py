"""Typed, pure-JAX registry boundary for interchangeable RL objectives.

The module declares a small registry of algorithm specifications and exposes a
stable public interface for picking an objective by name. Each entry bundles
pure advantage and loss functions with a fixed name for reproducible training.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
from flax import struct

from .algorithms import generalized_advantage_estimation, ppo_loss


@struct.dataclass
class LossOutput:
    """Scalar objective and immutable JAX metrics emitted by one algorithm.

    :param loss: Scalar differentiable objective.
    :param metrics: JAX-array metrics safe to aggregate outside a compiled epoch.
        :returns: Protocol object representing loss output.

        Example:
            >>> result = LossOutput(loss=0.1, metrics={"entropy": 0.01})
    """

    loss: jax.Array
    metrics: Mapping[str, jax.Array]


@struct.dataclass
class PpoLossBatch:
    """Flat PPO minibatch with every required old-policy field explicit.

    :ivar new_log_prob: jax.Array
    :ivar old_log_prob: jax.Array
    :ivar advantages: jax.Array
    :ivar new_value: jax.Array
    :ivar old_value: jax.Array
    :ivar returns: jax.Array
    :ivar entropy: jax.Array

    Example:
        >>> obj = PpoLossBatch(
        ...     new_log_prob=..., 
        ...     old_log_prob=..., 
        ...     advantages=..., 
        ...     new_value=..., 
        ...     old_value=..., 
        ...     returns=..., 
        ...     entropy=..., 
        ... )
    """

    new_log_prob: jax.Array
    old_log_prob: jax.Array
    advantages: jax.Array
    new_value: jax.Array
    old_value: jax.Array
    returns: jax.Array
    entropy: jax.Array


AdvantageFunction = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, float, float], tuple[jax.Array, jax.Array]
]
LossFunction = Callable[[PpoLossBatch], LossOutput]


@dataclass(frozen=True)
class AlgorithmSpec:
    """One registry entry of pure functions and its stable public name.

    :ivar name: str
    :ivar advantages: AdvantageFunction
    :ivar loss: LossFunction

    Example:
        >>> obj = AlgorithmSpec(name=..., advantages=..., loss=...)
    """

    name: str
    advantages: AdvantageFunction
    loss: LossFunction


def _ppo_loss(batch: PpoLossBatch) -> LossOutput:
    loss, metrics = ppo_loss(
        batch.new_log_prob,
        batch.old_log_prob,
        batch.advantages,
        batch.new_value,
        batch.old_value,
        batch.returns,
        0.2,
        0.5,
        batch.entropy,
        0.01,
    )
    return LossOutput(loss=loss, metrics={**metrics, "loss": loss})


PPO = AlgorithmSpec("ppo", generalized_advantage_estimation, _ppo_loss)
ALGORITHM_REGISTRY: Mapping[str, AlgorithmSpec] = {PPO.name: PPO}


def get_algorithm(name: str) -> AlgorithmSpec:
    """Return one declared algorithm without allowing silent fallbacks.

    :param name: str input value
    :returns: AlgorithmSpec

    Example:
        >>> result = get_algorithm(name)
    """
    try:
        return ALGORITHM_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown algorithm: {name!r}") from error
