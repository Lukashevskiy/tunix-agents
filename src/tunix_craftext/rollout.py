"""Collection primitives kept independent of a particular policy or environment API."""

from __future__ import annotations

from typing import Protocol, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .contracts import ActionT, ObservationT, RolloutBatch, Transition

StateT = TypeVar("StateT")
TreeT = TypeVar("TreeT")
PolicyObservationT = TypeVar("PolicyObservationT", contravariant=True)
PolicyActionT = TypeVar("PolicyActionT", covariant=True)
StepActionT = TypeVar("StepActionT", contravariant=True)
StepObservationT = TypeVar("StepObservationT", covariant=True)


class PolicyFn(Protocol[PolicyObservationT, PolicyActionT]):
    """Policy signature used by the framework-neutral reference collector."""

    def __call__(self, observation: PolicyObservationT) -> tuple[PolicyActionT, ArrayLike, ArrayLike]: ...


class StepFn(Protocol[StateT, StepActionT, StepObservationT]):
    """Synchronous environment signature used by the reference collector."""

    def __call__(
        self, state: StateT, action: StepActionT
    ) -> tuple[StateT, StepObservationT, ArrayLike, ArrayLike, ArrayLike]: ...


class IndexedStepFn(Protocol[StateT, StepActionT, StepObservationT]):
    """JAX step signature that receives the static scan index for explicit RNG selection."""

    def __call__(
        self, state: StateT, action: StepActionT, step_index: jax.Array
    ) -> tuple[StateT, StepObservationT, ArrayLike, ArrayLike, ArrayLike]: ...


def collect_rollout(
    initial_state: StateT,
    initial_observation: ObservationT,
    horizon: int,
    policy: PolicyFn[ObservationT, ActionT],
    step: StepFn[StateT, ActionT, ObservationT],
) -> tuple[StateT, ObservationT, RolloutBatch[ObservationT, ActionT]]:
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
    records: list[tuple[ObservationT, ActionT, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]] = []
    for _ in range(horizon):
        action, log_prob, value = policy(observation)
        next_state, next_observation, reward, terminated, truncated = step(state, action)
        records.append((observation, action, reward, terminated, truncated, log_prob, value))
        state, observation = next_state, next_observation
    observations = _stack_pytree([record[0] for record in records])
    actions = _stack_pytree([record[1] for record in records])
    batch = RolloutBatch(
        Transition(
            observation=observations,
            action=actions,
            reward=_stack_arraylike([record[2] for record in records]),
            terminated=_stack_arraylike([record[3] for record in records]),
            truncated=_stack_arraylike([record[4] for record in records]),
            log_prob=_stack_arraylike([record[5] for record in records]),
            value=_stack_arraylike([record[6] for record in records]),
        ),
        bootstrap_value=jnp.asarray(policy(observation)[2]),
    )
    batch.validate()
    return state, observation, batch


def collect_rollout_scan(
    initial_state: StateT,
    initial_observation: ObservationT,
    horizon: int,
    policy: PolicyFn[ObservationT, ActionT],
    step: StepFn[StateT, ActionT, ObservationT],
) -> tuple[StateT, ObservationT, RolloutBatch[ObservationT, ActionT]]:
    """Collect a fixed-horizon rollout using ``jax.lax.scan``.

    The function is pure and JIT-safe when ``policy`` and ``step`` are pure JAX functions and
    ``horizon`` is static. It intentionally omits host-side shape validation; compare it with
    :func:`collect_rollout` in a parity test before introducing a new environment or policy.

    :param initial_state: JAX PyTree state before time zero.
    :param initial_observation: Batched observation PyTree at time zero.
    :param horizon: Positive static rollout length ``T``.
    :param policy: Pure JAX policy yielding action, log-probability and critic value.
    :param step: Pure JAX environment transition.
    :returns: Final state, final observation and time-major ``RolloutBatch`` leaves ``[T, B, ...]``.
    :raises ValueError: If ``horizon`` is not positive.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    def scan_step(
        carry: tuple[StateT, ObservationT], _: jax.Array
    ) -> tuple[
        tuple[StateT, ObservationT],
        tuple[ObservationT, ActionT, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ]:
        state, observation = carry
        action, log_prob, value = policy(observation)
        next_state, next_observation, reward, terminated, truncated = step(state, action)
        return (next_state, next_observation), (
            observation,
            action,
            jnp.asarray(reward),
            jnp.asarray(terminated),
            jnp.asarray(truncated),
            jnp.asarray(log_prob),
            jnp.asarray(value),
        )

    (final_state, final_observation), fields = jax.lax.scan(
        scan_step, (initial_state, initial_observation), xs=jnp.arange(horizon)
    )
    bootstrap_value = jnp.asarray(policy(final_observation)[2])
    observation, action, reward, terminated, truncated, log_prob, value = fields
    return final_state, final_observation, RolloutBatch(
        Transition(observation, action, reward, terminated, truncated, log_prob, value), bootstrap_value
    )


def collect_rollout_scan_indexed(
    initial_state: StateT,
    initial_observation: ObservationT,
    horizon: int,
    policy: PolicyFn[ObservationT, ActionT],
    step: IndexedStepFn[StateT, ActionT, ObservationT],
) -> tuple[StateT, ObservationT, RolloutBatch[ObservationT, ActionT]]:
    """Collect a JIT-safe rollout while passing each ``lax.scan`` index to the step function.

    The index lets callers select pre-split environment keys ``[T, B, 2]`` without hidden global
    RNG state. It is the production form for real CrafText environments.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    def scan_step(carry: tuple[StateT, ObservationT], index: jax.Array):
        state, observation = carry
        action, log_prob, value = policy(observation)
        next_state, next_observation, reward, terminated, truncated = step(state, action, index)
        return (next_state, next_observation), (
            observation, action, jnp.asarray(reward), jnp.asarray(terminated), jnp.asarray(truncated),
            jnp.asarray(log_prob), jnp.asarray(value),
        )

    (final_state, final_observation), fields = jax.lax.scan(
        scan_step, (initial_state, initial_observation), xs=jnp.arange(horizon)
    )
    observation, action, reward, terminated, truncated, log_prob, value = fields
    return final_state, final_observation, RolloutBatch(
        Transition(observation, action, reward, terminated, truncated, log_prob, value),
        jnp.asarray(policy(final_observation)[2]),
    )


def _stack_pytree(values: list[TreeT]) -> TreeT:
    """Stack reference PyTrees on host, then normalize leaves to ``jax.Array`` once."""
    return cast(TreeT, jax.tree.map(lambda *leaves: _stack_arraylike(list(leaves)), *values))


def _stack_arraylike(values: list[ArrayLike]) -> jax.Array:
    """Stack a host reference field and return its normalized JAX contract array.

    ``collect_rollout`` is deliberately the non-jitted oracle. Host stacking avoids a series of
    eager JAX dispatches while preserving the ``jax.Array`` output contract used for parity.
    """
    return jnp.asarray(np.stack(values))
