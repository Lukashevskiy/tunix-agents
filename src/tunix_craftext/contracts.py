"""Stable, framework-neutral data contracts at the environment/trainer boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import numpy as np


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
ArrayT = TypeVar("ArrayT")


@dataclass(frozen=True)
class Transition(Generic[ObservationT, ActionT, ArrayT]):
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
    reward: ArrayT
    terminated: ArrayT
    truncated: ArrayT
    log_prob: ArrayT
    value: ArrayT

    @property
    def done(self) -> ArrayT:
        return cast(ArrayT, np.logical_or(self.terminated, self.truncated))


@dataclass(frozen=True)
class RolloutBatch(Generic[ObservationT, ActionT, ArrayT]):
    """A time-major `[T, B, ...]` rollout plus bootstrap values `[B]`."""

    transitions: Transition[ObservationT, ActionT, ArrayT]
    bootstrap_value: ArrayT

    def validate(self) -> None:
        """Validate the mandatory time-major rollout axes.

        :raises ValueError: If a required field does not begin with the reward's ``[T, B]``
            axes or bootstrap value is not ``[B]``.
        """
        leaves = self.transitions
        reward = np.asarray(leaves.reward)
        if reward.ndim != 2:
            raise ValueError("reward must be time-major with shape [T, B]")
        for name in ("terminated", "truncated", "log_prob", "value"):
            if np.asarray(getattr(leaves, name)).shape[:2] != reward.shape:
                raise ValueError(f"{name} must begin with rollout shape {reward.shape}")
        if np.asarray(self.bootstrap_value).shape != reward.shape[1:]:
            raise ValueError("bootstrap_value must have shape [B]")
