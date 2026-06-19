"""Collection primitives kept independent of a particular policy or environment API."""

from __future__ import annotations

from typing import Protocol, TypeVar, cast

import numpy as np

from .contracts import ActionT, ArrayT, ObservationT, RolloutBatch, Transition

StateT = TypeVar("StateT")


class PolicyFn(Protocol[ObservationT, ActionT, ArrayT]):
    """Policy signature used by the framework-neutral reference collector."""

    def __call__(self, observation: ObservationT) -> tuple[ActionT, ArrayT, ArrayT]: ...


class StepFn(Protocol[StateT, ActionT, ObservationT, ArrayT]):
    """Synchronous environment signature used by the reference collector."""

    def __call__(
        self, state: StateT, action: ActionT
    ) -> tuple[StateT, ObservationT, ArrayT, ArrayT, ArrayT]: ...


def collect_rollout(
    initial_state: StateT,
    initial_observation: ObservationT,
    horizon: int,
    policy: PolicyFn[ObservationT, ActionT, ArrayT],
    step: StepFn[StateT, ActionT, ObservationT, ArrayT],
) -> tuple[StateT, ObservationT, RolloutBatch[ObservationT, ActionT, ArrayT]]:
    """Collect a deterministic, time-major rollout.

    The production version will be `jax.lax.scan`; this reference implementation is the
    executable contract used by unit tests and adapter parity tests.

    :param initial_state: State before the first environment step.
    :param initial_observation: Batched observation consumed at time zero.
    :param horizon: Positive rollout length ``T``.
    :param policy: Pure function yielding action, log-probability and value.
    :param step: Pure environment step yielding next state, observation, reward and done flags.
    :returns: Final state, final observation and a ``RolloutBatch`` with ``[T, B, ...]`` leaves.
    :raises ValueError: If ``horizon`` is not positive or generated leaves violate the contract.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    state, observation = initial_state, initial_observation
    records: list[tuple[ObservationT, ActionT, ArrayT, ArrayT, ArrayT, ArrayT, ArrayT]] = []
    for _ in range(horizon):
        action, log_prob, value = policy(observation)
        next_state, next_observation, reward, terminated, truncated = step(state, action)
        records.append((observation, action, reward, terminated, truncated, log_prob, value))
        state, observation = next_state, next_observation
    fields = tuple(cast(ArrayT, np.stack([record[index] for record in records])) for index in range(7))
    batch = RolloutBatch(Transition(*fields), policy(observation)[2])
    batch.validate()
    return state, observation, batch
