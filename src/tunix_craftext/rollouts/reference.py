"""Reference fixed-shape rollout collectors independent of vendor APIs.

The module defines signatures and helpers for collecting numeric rollouts in a
framework-neutral form.  These collectors are intentionally reference/contract
tools for CPU tests, deterministic fixtures and fixed-shape JAX parity checks.
Production LLM-RL with growing prompt history should use a host-orchestrated
hybrid rollout boundary such as :mod:`tunix_craftext.rollouts.hybrid`.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..core.contracts import ActionT, ObservationT, RolloutBatch, Transition
from ..core.tensor_types import BatchBool, BatchFloat, JaxArray, JaxArrayLike, ScalarInt

StateT = TypeVar("StateT")
TreeT = TypeVar("TreeT")
PolicyObservationT = TypeVar("PolicyObservationT", contravariant=True)
PolicyActionT = TypeVar("PolicyActionT", covariant=True)
StepActionT = TypeVar("StepActionT", contravariant=True)
StepObservationT = TypeVar("StepObservationT", covariant=True)
RolloutRecordT = tuple[
    ObservationT,
    ActionT,
    JaxArrayLike,
    JaxArrayLike,
    JaxArrayLike,
    JaxArrayLike,
    JaxArrayLike,
]


class PolicyFn(Protocol[PolicyObservationT, PolicyActionT]):
    """Policy signature used by the framework-neutral reference collector."""

    def __call__(
        self, observation: PolicyObservationT
    ) -> tuple[PolicyActionT, BatchFloat, BatchFloat]:
        """Return an action, its log-probability and value for a batched observation.

        :param observation: Batched observation PyTree consumed by the policy.
        :returns: Tuple of (action, log_prob, value).
        """
        ...


class StepFn(Protocol[StateT, StepActionT, StepObservationT]):
    """Synchronous environment signature used by the reference collector."""

    def __call__(
        self, state: StateT, action: StepActionT
    ) -> tuple[StateT, StepObservationT, BatchFloat, BatchBool, BatchBool]:
        """Step the environment synchronously for one action.

        :param state: Current environment state.
        :param action: Action selected by the policy.
        :returns: Tuple of (next_state, next_observation, reward, terminated, truncated).
        """
        ...


class IndexedStepFn(Protocol[StateT, StepActionT, StepObservationT]):
    """JAX step signature that receives the static scan index for explicit RNG selection."""

    def __call__(
        self, state: StateT, action: StepActionT, step_index: ScalarInt
    ) -> tuple[StateT, StepObservationT, BatchFloat, BatchBool, BatchBool]:
        """JAX-compatible step accepting the scan index for RNG selection.

        :param state: Current environment state PyTree.
        :param action: Action selected by the policy.
        :param step_index: Integer scan index used to select RNG shards.
        :returns: Tuple of (next_state, next_observation, reward, terminated, truncated).
        """
        ...


def collect_rollout(
    initial_state: StateT,
    initial_observation: ObservationT,
    horizon: int,
    policy: PolicyFn[ObservationT, ActionT],
    step: StepFn[StateT, ActionT, ObservationT],
) -> tuple[StateT, ObservationT, RolloutBatch[ObservationT, ActionT]]:
    """Collect a deterministic, time-major rollout.

    This reference implementation is the executable contract used by unit tests
    and adapter parity tests. It is not the production LLM-RL collector for
    dynamic prompt-history rollouts.

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
    records: list[RolloutRecordT[ObservationT, ActionT]] = []
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
        carry: tuple[StateT, ObservationT], _: ScalarInt
    ) -> tuple[
        tuple[StateT, ObservationT],
        tuple[ObservationT, ActionT, BatchFloat, BatchBool, BatchBool, BatchFloat, BatchFloat],
    ]:
        state, observation = carry
        """One step of the jax.lax.scan used by `collect_rollout_scan`.

        :param carry: Tuple of (state, observation) carried across time.
        :param _: Unused scan index.
        :returns: New carry and observation/action/reward/terminal/log-prob/value tuple.
        """
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
    return (
        final_state,
        final_observation,
        RolloutBatch(
            Transition(observation, action, reward, terminated, truncated, log_prob, value),
            bootstrap_value,
        ),
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
    RNG state. It is the fixed-shape JAX parity form for CrafText-style environments; real
    LLM-RL rollout with dynamic text history must use the hybrid host/Tunix boundary.
    :param initial_state: JAX PyTree state before time zero.
    :param initial_observation: Batched observation PyTree at time zero.
    :param horizon: Positive static rollout length ``T``.
    :param policy: Pure JAX policy yielding action, log-probability and critic value.
    :param step: Indexed step function accepting the scan index.
    :returns: Final state, final observation and time-major ``RolloutBatch`` leaves ``[T, B, ...]``.
    :raises ValueError: If ``horizon`` is not positive.

    Example:
        >>> final_state, final_obs, batch = collect_rollout_scan_indexed(
        ...     initial_state, initial_obs, horizon, policy, step
        ... )
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    def scan_step(carry: tuple[StateT, ObservationT], index: ScalarInt):
        state, observation = carry
        """One indexed scan step forwarding the provided `index` to the `step` function.

        :param carry: Tuple of (state, observation).
        :param index: Scan index used by `IndexedStepFn` for RNG selection.
        :returns: New carry and the flattened transition fields as JAX arrays.
        """
        action, log_prob, value = policy(observation)
        next_state, next_observation, reward, terminated, truncated = step(state, action, index)
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
    observation, action, reward, terminated, truncated, log_prob, value = fields
    return (
        final_state,
        final_observation,
        RolloutBatch(
            Transition(observation, action, reward, terminated, truncated, log_prob, value),
            jnp.asarray(policy(final_observation)[2]),
        ),
    )


def _stack_pytree(values: list[TreeT]) -> TreeT:
    """Stack reference PyTrees on host, then normalize leaves to ``jax.Array`` once.

    :param values: list[TreeT] input value
    :returns: TreeT

    Example:
        >>> result = _stack_pytree(values)
    """
    return cast(TreeT, jax.tree.map(lambda *leaves: _stack_arraylike(list(leaves)), *values))


def _stack_arraylike(values: list[JaxArrayLike]) -> JaxArray:
    """Stack a host reference field and return its normalized JAX contract array.

    :param values: List of host-side array-like objects to stack along axis 0.
    :returns: A JAX array with stacked values.

    Example:
        >>> arr = _stack_arraylike([np.array([1,2]), np.array([3,4])])
    """
    return jnp.asarray(np.stack(values))
