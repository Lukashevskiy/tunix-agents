"""Pure, JAX-compatible boundary for CrafText and CagedCrafText environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, Protocol, Tuple, TypeVar

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # Allow contract tests and documentation to run without the accelerator stack.
    jax = None
    import numpy as jnp


KeyT = TypeVar("KeyT")
ParamsT = TypeVar("ParamsT")
ObservationT = TypeVar("ObservationT")
StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
ArrayT = TypeVar("ArrayT")


class AdapterContractError(ValueError):
    """Raised when a vendor environment violates the fixed adapter boundary."""


class CrafTextEnvironment(Protocol[KeyT, ParamsT, ObservationT, StateT, ActionT, ArrayT]):
    """Minimal CrafText/CagedCrafText reset-step contract used by the adapter."""

    def reset(self, key: KeyT, params: ParamsT) -> Tuple[ObservationT, StateT]:
        """Reset one environment episode.

        :param key: Environment RNG key.
        :param params: Static vendor environment parameters.
        :returns: Initial observation and opaque environment state.
        """
        ...

    def step(
        self, key: KeyT, state: StateT, action: ActionT, params: ParamsT
    ) -> Tuple[ObservationT, StateT, ArrayT, ArrayT, Mapping[str, ArrayT]]:
        """Run one vendor transition with its single ``done`` flag.

        :param key: Environment RNG key for this step.
        :param state: Previous opaque environment state.
        :param action: Environment action with the adapter's declared discrete cardinality.
        :param params: Static vendor environment parameters.
        :returns: Observation, next state, reward, done and auxiliary tensors.
        """
        ...


@dataclass(frozen=True)
class EnvironmentReset(Generic[ObservationT, StateT, ArrayT]):
    """Normalized reset result with a static action-mask shape.

    :ivar observation: Initial environment observation.
    :ivar state: Opaque vendor state; it remains a JAX PyTree in the production environment.
    :ivar action_mask: Boolean vector ``[A]`` marking currently available actions.
    """

    observation: ObservationT
    state: StateT
    action_mask: ArrayT


@dataclass(frozen=True)
class EnvironmentStep(Generic[ObservationT, StateT, ArrayT]):
    """Normalized CrafText transition.

    CrafText's vendor ``done`` is represented as ``terminated``. ``truncated`` is an explicit
    all-false tensor because the vendor API does not distinguish timeout truncation.

    :ivar observation: Observation after the action.
    :ivar state: Next opaque vendor state.
    :ivar reward: Environment reward.
    :ivar terminated: Vendor completion flag.
    :ivar truncated: All-false truncation flag with the same shape as ``terminated``.
    :ivar action_mask: Boolean vector ``[A]`` for the next action.
    """

    observation: ObservationT
    state: StateT
    reward: ArrayT
    terminated: ArrayT
    truncated: ArrayT
    action_mask: ArrayT


if jax is not None:
    jax.tree_util.register_dataclass(EnvironmentReset, data_fields=["observation", "state", "action_mask"], meta_fields=[])
    jax.tree_util.register_dataclass(
        EnvironmentStep,
        data_fields=["observation", "state", "reward", "terminated", "truncated", "action_mask"],
        meta_fields=[],
    )


class CrafTextAdapter(Generic[KeyT, ParamsT, ObservationT, StateT, ActionT, ArrayT]):
    """Normalize one CrafText-family environment without mutating vendor state or info.

    :param environment: Vendor CrafText-compatible environment.
    :param params: Static parameters passed to each reset and step.
    :param action_count: Static discrete action cardinality ``A``; it fixes mask shape ``[A]``.
    :param action_mask_key: Optional key in vendor info containing a boolean ``[A]`` mask.
    :raises AdapterContractError: If action cardinality is invalid or a supplied mask has a
        non-static/wrong shape.
    """

    def __init__(
        self,
        environment: CrafTextEnvironment[KeyT, ParamsT, ObservationT, StateT, ActionT, ArrayT],
        params: ParamsT,
        action_count: int,
        action_mask_key: str = "action_mask",
    ) -> None:
        if action_count <= 0:
            raise AdapterContractError("action_count must be positive")
        self._environment = environment
        self._params = params
        self._action_count = action_count
        self._action_mask_key = action_mask_key

    def _fallback_mask(self) -> ArrayT:
        """Return the conservative all-actions-available mask with shape ``[A]``."""
        return jnp.ones((self._action_count,), dtype=bool)

    def _action_mask(self, info: Mapping[str, ArrayT]) -> ArrayT:
        """Extract and validate next-action availability without retaining vendor info.

        :param info: Vendor information mapping after one transition.
        :returns: Validated boolean mask with static shape ``[A]``.
        :raises AdapterContractError: If an explicit mask does not have shape ``[A]``.
        """
        mask = info.get(self._action_mask_key)
        if mask is None:
            return self._fallback_mask()
        if tuple(mask.shape) != (self._action_count,):
            raise AdapterContractError(
                f"{self._action_mask_key} must have shape ({self._action_count},), got {mask.shape}"
            )
        return mask

    def reset(self, key: KeyT) -> EnvironmentReset[ObservationT, StateT, ArrayT]:
        """Reset CrafText and attach an all-true static action mask.

        :param key: Environment RNG key owned by the caller.
        :returns: Initial observation/state and a fallback action mask ``[A]``.
        """
        observation, state = self._environment.reset(key, self._params)
        return EnvironmentReset(observation=observation, state=state, action_mask=self._fallback_mask())

    def step(self, key: KeyT, state: StateT, action: ActionT) -> EnvironmentStep[ObservationT, StateT, ArrayT]:
        """Step CrafText and split its single terminal flag into the training contract.

        :param key: Per-step environment RNG key owned by the caller.
        :param state: Previous vendor state.
        :param action: Action selected by the policy.
        :returns: Normalized transition; ``truncated`` is all false for the current vendor API.
        """
        observation, next_state, reward, done, info = self._environment.step(key, state, action, self._params)
        return EnvironmentStep(
            observation=observation,
            state=next_state,
            reward=reward,
            terminated=done,
            truncated=jnp.zeros_like(done, dtype=bool),
            action_mask=self._action_mask(info),
        )


class CagedCrafTextAdapter(CrafTextAdapter[KeyT, ParamsT, ObservationT, StateT, ActionT, ArrayT]):
    """CagedCrafText adapter with the same normalized contract as base CrafText.

    Constraint costs remain in the vendor observation/state or a future explicit cost adapter;
    this class guarantees only common trajectory semantics.
    """
