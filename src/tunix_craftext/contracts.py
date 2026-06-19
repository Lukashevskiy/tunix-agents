"""Stable, framework-neutral data contracts at the environment/trainer boundary."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


class Transition(NamedTuple):
    """One synchronous environment step; all leaves must have a leading batch axis."""

    observation: Any
    action: Any
    reward: Any
    terminated: Any
    truncated: Any
    log_prob: Any
    value: Any

    @property
    def done(self) -> Any:
        return np.logical_or(self.terminated, self.truncated)


class RolloutBatch(NamedTuple):
    """A time-major `[T, B, ...]` rollout plus bootstrap values `[B]`."""

    transitions: Transition
    bootstrap_value: Any

    def validate(self) -> None:
        """Raise early for the shape bugs that otherwise poison a compiled learner."""
        leaves = self.transitions
        reward = np.asarray(leaves.reward)
        if reward.ndim != 2:
            raise ValueError("reward must be time-major with shape [T, B]")
        for name in ("terminated", "truncated", "log_prob", "value"):
            if np.asarray(getattr(leaves, name)).shape[:2] != reward.shape:
                raise ValueError(f"{name} must begin with rollout shape {reward.shape}")
        if np.asarray(self.bootstrap_value).shape != reward.shape[1:]:
            raise ValueError("bootstrap_value must have shape [B]")
