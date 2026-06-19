"""Collection primitives kept independent of a particular policy or environment API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .contracts import RolloutBatch, Transition

StepFn = Callable[[Any, Any], tuple[Any, Any, Any, Any, Any]]
PolicyFn = Callable[[Any], tuple[Any, Any, Any]]


def collect_rollout(
    initial_state: Any, initial_observation: Any, horizon: int, policy: PolicyFn, step: StepFn
) -> tuple[Any, Any, RolloutBatch]:
    """Collect a deterministic, time-major rollout.

    The production version will be `jax.lax.scan`; this reference implementation is the
    executable contract used by unit tests and adapter parity tests.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    state, observation, records = initial_state, initial_observation, []
    for _ in range(horizon):
        action, log_prob, value = policy(observation)
        next_state, next_observation, reward, terminated, truncated = step(state, action)
        records.append((observation, action, reward, terminated, truncated, log_prob, value))
        state, observation = next_state, next_observation
    fields = tuple(np.stack([record[index] for record in records]) for index in range(7))
    batch = RolloutBatch(Transition(*fields), policy(observation)[2])
    batch.validate()
    return state, observation, batch
