"""Pure, JAX-compatible boundary for CrafText and CagedCrafText environments.

This module defines the adapter contract and helper types required to wrap a
vendor CrafText environment so it can be consumed by a deterministic Flax-based
training loop. It validates action masks, terminal semantics, and resets.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..core.tensor_types import JaxKey, ScalarBool, ScalarFloat, ScalarInt, SingleActionMask

ParamsT = TypeVar("ParamsT")
ObservationT = TypeVar("ObservationT")
StateT = TypeVar("StateT")
EnvironmentParamsT = TypeVar("EnvironmentParamsT", contravariant=True)
EnvironmentObservationT = TypeVar("EnvironmentObservationT", covariant=True)


class AdapterContractError(ValueError):
    """Raised when a vendor environment violates the fixed adapter boundary.

    Example:
        >>> raise AdapterContractError("message")
    """


class CrafTextEnvironment(Protocol[EnvironmentParamsT, EnvironmentObservationT, StateT]):
    """Minimal CrafText/CagedCrafText reset-step contract used by the adapter."""

    def reset(
        self, key: JaxKey, params: EnvironmentParamsT
    ) -> tuple[EnvironmentObservationT, StateT]:
        """Reset one environment episode.

        :param key: Environment RNG key.
        :param params: Static vendor environment parameters.
        :returns: Initial observation and opaque environment state.
        """
        ...

    def step(
        self,
        key: JaxKey,
        state: StateT,
        action: int | ScalarInt,
        params: EnvironmentParamsT,
    ) -> tuple[
        EnvironmentObservationT,
        StateT,
        ScalarFloat,
        ScalarBool,
        Mapping[str, ArrayLike],
    ]:
        """Run one vendor transition with its single ``done`` flag.

        :param key: Environment RNG key for this step.
        :param state: Previous opaque environment state.
        :param action: Environment action with the adapter's declared discrete cardinality.
        :param params: Static vendor environment parameters.
        :returns: Observation, next state, reward, done and auxiliary tensors.
        """
        ...


@dataclass(frozen=True)
class EnvironmentReset(Generic[ObservationT, StateT]):
    """Normalized reset result with a static action-mask shape.

    :ivar observation: Initial environment observation.
    :ivar state: Opaque vendor state; it remains a JAX PyTree in the production environment.
    :ivar action_mask: Boolean vector ``[A]`` marking currently available actions.
    """

    observation: ObservationT
    state: StateT
    action_mask: SingleActionMask


@dataclass(frozen=True)
class EnvironmentStep(Generic[ObservationT, StateT]):
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
    reward: ScalarFloat
    terminated: ScalarBool
    truncated: ScalarBool
    action_mask: SingleActionMask


jax.tree_util.register_dataclass(
    EnvironmentReset, data_fields=["observation", "state", "action_mask"], meta_fields=[]
)
jax.tree_util.register_dataclass(
    EnvironmentStep,
    data_fields=["observation", "state", "reward", "terminated", "truncated", "action_mask"],
    meta_fields=[],
)


class CraftaxAdapter(Generic[ParamsT, ObservationT, StateT]):
    """Normalize a bare Craftax-compatible environment without text/task metadata.

    This is deliberately the lowest environment boundary.  It knows only the
    JAX ``reset``/``step`` protocol, terminal semantics and action masks.  In
    particular, it does not construct world presets, select instructions, or
    interpret a CrafText ``TextEnvState``.

    :param environment: Vendor Craftax-compatible environment.
    :param params: Static parameters passed to each reset and step.
    :param action_count: Static discrete action cardinality ``A``; it fixes mask shape ``[A]``.
    :param action_mask_key: Optional key in vendor info containing a boolean ``[A]`` mask.
    :raises AdapterContractError: If action cardinality is invalid or a supplied mask has a
        non-static/wrong shape.
    """

    def __init__(
        self,
        environment: CrafTextEnvironment[ParamsT, ObservationT, StateT],
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

    @property
    def action_count(self) -> int:
        """Return the static discrete action cardinality exposed by this adapter.

        :returns: int

        Example:
        >>> result = action_count()
        """
        return self._action_count

    @property
    def world_preset(self) -> str:
        """Return optional world-preset provenance for prompt rendering.

        Bare Craftax environments do not own CrafText world-preset metadata, so
        the base adapter exposes an empty value. CrafText-specialized adapters
        override this with runtime provenance.
        """
        return ""

    @property
    def has_instruction_context(self) -> bool:
        """Whether this adapter can resolve host-side instruction metadata."""
        return False

    @staticmethod
    def prompt_state(state: StateT) -> object:
        """Return the state object visible to prompt renderers.

        Bare Craftax-compatible environments already expose the prompt state
        directly. CrafText wrappers override this projection to unwrap
        ``TextEnvState.env_state``.
        """
        return state

    def episode_context(self, state: StateT) -> "CrafTextEpisodeContext[object]":
        """Resolve optional CrafText instruction metadata.

        Bare Craftax adapters deliberately do not own scenario instructions.
        Callers should check ``has_instruction_context`` before requesting this
        metadata.
        """
        del state
        raise AdapterContractError("CrafText instruction metadata is not configured")

    def _fallback_mask(self) -> SingleActionMask:
        """Return the conservative all-actions-available mask with shape ``[A]``.

        :returns: jax.Array

        Example:
        >>> result = _fallback_mask()
        """
        return jnp.ones((self._action_count,), dtype=bool)

    def _action_mask(self, info: Mapping[str, ArrayLike]) -> SingleActionMask:
        """Extract and validate next-action availability without retaining vendor info.

        :param info: Vendor information mapping after one transition.
        :returns: Validated boolean mask with static shape ``[A]``.
        :raises AdapterContractError: If an explicit mask does not have shape ``[A]``.
        """
        mask = info.get(self._action_mask_key)
        if mask is None:
            return self._fallback_mask()
        normalized_mask = jnp.asarray(mask, dtype=bool)
        if tuple(normalized_mask.shape) != (self._action_count,):
            raise AdapterContractError(
                f"{self._action_mask_key} must have shape ({self._action_count},), "
                f"got {normalized_mask.shape}"
            )
        return normalized_mask

    def reset(self, key: JaxKey) -> EnvironmentReset[ObservationT, StateT]:
        """Reset CrafText and attach an all-true static action mask.

        :param key: Environment RNG key owned by the caller.
        :returns: Initial observation/state and a fallback action mask ``[A]``.
        """
        observation, state = self._environment.reset(key, self._params)
        return EnvironmentReset(
            observation=observation, state=state, action_mask=self._fallback_mask()
        )

    def step(
        self, key: JaxKey, state: StateT, action: int | ScalarInt
    ) -> EnvironmentStep[ObservationT, StateT]:
        """Step CrafText and split its single terminal flag into the training contract.

        :param key: Per-step environment RNG key owned by the caller.
        :param state: Previous vendor state.
        :param action: Action selected by the policy.
        :returns: Normalized transition; ``truncated`` is all false for the current vendor API.
        """
        observation, next_state, reward, done, info = self._environment.step(
            key, state, action, self._params
        )
        return EnvironmentStep(
            observation=observation,
            state=next_state,
            reward=jnp.asarray(reward),
            terminated=jnp.asarray(done, dtype=bool),
            truncated=jnp.zeros_like(jnp.asarray(done), dtype=bool),
            action_mask=self._action_mask(info),
        )


@dataclass(frozen=True)
class CrafTextEpisodeContext(Generic[StateT]):
    """Host-side task metadata paired with CrafText's underlying ``EnvState``.

    :ivar world_preset: Reproducible CrafText world preset selected by the run config.
    :ivar instruction: Instruction selected by the CrafText wrapper for this episode.
    :ivar env_state: Underlying Craftax ``EnvState`` for prompt rendering.
    :ivar text_constraint: Optional CagedCrafText safety constraint.
    """

    world_preset: str
    instruction: str
    env_state: StateT
    text_constraint: str = ""


class CrafTextAdapter(CraftaxAdapter[ParamsT, ObservationT, StateT]):
    """CrafText boundary: Craftax transition contract plus instruction state.

    ``CrafTextAdapter`` does not create a world itself.  The runtime constructs
    the vendor world preset and ``RawInstructionWrapper`` first, then this
    adapter binds their public metadata to the normalized transition contract.
    ``prompt_state`` unwraps CrafText's ``TextEnvState.env_state`` so MegaPrompts
    receives the structured Craftax state it expects.

    :param world_preset: Name of the resolved world preset; provenance only.
    :param instructions: Scenario rows aligned with ``TextEnvState.idx``.
    :param instruction_index: Optional fixed scenario row used by the vendor reset.
    """

    def __init__(
        self,
        environment: CrafTextEnvironment[ParamsT, ObservationT, StateT],
        params: ParamsT,
        action_count: int,
        action_mask_key: str = "action_mask",
        *,
        world_preset: str = "",
        instructions: tuple[str, ...] = (),
        instruction_index: int | None = None,
    ) -> None:
        super().__init__(environment, params, action_count, action_mask_key)
        if instruction_index is not None and instruction_index < 0:
            raise AdapterContractError("instruction_index must be non-negative")
        if (
            instruction_index is not None
            and instructions
            and instruction_index >= len(instructions)
        ):
            raise AdapterContractError(
                "instruction_index must reference one configured instruction"
            )
        self._world_preset = world_preset
        self._instructions = instructions
        self._instruction_index = instruction_index

    @property
    def world_preset(self) -> str:
        """Return the CrafText world-preset provenance, if configured."""
        return self._world_preset

    @property
    def has_instruction_context(self) -> bool:
        """Whether this adapter was built around a CrafText instruction wrapper."""
        return bool(self._instructions)

    @property
    def instructions(self) -> tuple[str, ...]:
        """Return configured CrafText scenario instructions in vendor order."""
        return self._instructions

    def reset(self, key: JaxKey) -> EnvironmentReset[ObservationT, StateT]:
        """Reset CrafText and bind the configured instruction when available."""
        if self._instruction_index is None:
            return super().reset(key)
        return self.reset_with_instruction(key, self._instruction_index)

    def reset_with_instruction(
        self, key: JaxKey, instruction_index: int
    ) -> EnvironmentReset[ObservationT, StateT]:
        """Reset CrafText with an explicit scenario instruction row.

        Agentic training can therefore sample tasks from CrafText's own
        instruction list per batch row while keeping the adapter/runtime
        reusable.
        """
        if not self._instructions:
            raise AdapterContractError("CrafText instruction metadata is not configured")
        if instruction_index < 0 or instruction_index >= len(self._instructions):
            raise AdapterContractError(
                "instruction_index must reference one configured instruction"
            )
        observation, state = self._environment.reset(  # type: ignore[call-arg]
            key, self._params, instruction_idx=instruction_index
        )
        return EnvironmentReset(
            observation=observation, state=state, action_mask=self._fallback_mask()
        )

    @staticmethod
    def prompt_state(state: StateT) -> object:
        """Return the underlying Craftax ``EnvState`` for a CrafText prompt.

        Bare test environments are accepted for backwards-compatible unit
        fixtures.  Real CrafText wrappers expose ``env_state``.
        """
        return getattr(state, "env_state", state)

    def episode_context(self, state: StateT) -> CrafTextEpisodeContext[object]:
        """Resolve host-side instruction metadata for one CrafText wrapper state.

        This method is intentionally host-only: conversion of a JAX scalar
        ``idx`` to a Python index happens while rendering a text prompt, never
        inside ``jit`` or ``scan``.

        :param state: Current vendor ``TextEnvState``.
        :returns: World preset, selected instruction and underlying ``EnvState``.
        :raises AdapterContractError: If the adapter lacks scenario metadata.
        """
        if not self._instructions:
            raise AdapterContractError("CrafText instruction metadata is not configured")
        raw_index = getattr(state, "idx", None)
        if raw_index is None:
            raise AdapterContractError("CrafText state must expose instruction idx")
        index = int(raw_index)
        if not 0 <= index < len(self._instructions):
            raise AdapterContractError("CrafText state instruction idx is outside configured rows")
        return CrafTextEpisodeContext(
            world_preset=self._world_preset,
            instruction=self._instructions[index],
            env_state=self.prompt_state(state),
        )


class CagedCrafTextAdapter(CrafTextAdapter[ParamsT, ObservationT, StateT]):
    """CagedCrafText boundary: CrafText instruction state plus textual constraint.

    Constraint costs remain in the vendor state.  This adapter exports the
    *textual* constraint aligned with the selected instruction so that a prompt
    renderer can state the safety requirement explicitly.
    """

    def __init__(
        self,
        environment: CrafTextEnvironment[ParamsT, ObservationT, StateT],
        params: ParamsT,
        action_count: int,
        action_mask_key: str = "action_mask",
        *,
        world_preset: str = "",
        instructions: tuple[str, ...] = (),
        text_constraints: tuple[str, ...] = (),
        instruction_index: int | None = None,
    ) -> None:
        if text_constraints and len(text_constraints) != len(instructions):
            raise AdapterContractError("text_constraints must align one-to-one with instructions")
        super().__init__(
            environment,
            params,
            action_count,
            action_mask_key,
            world_preset=world_preset,
            instructions=instructions,
            instruction_index=instruction_index,
        )
        self._text_constraints = text_constraints

    @property
    def text_constraints(self) -> tuple[str, ...]:
        """Return configured CagedCrafText textual constraints in instruction order."""
        return self._text_constraints

    def episode_context(self, state: StateT) -> CrafTextEpisodeContext[object]:
        """Resolve CrafText metadata and the selected Caged textual constraint."""
        context = super().episode_context(state)
        raw_index = getattr(state, "idx", None)
        assert raw_index is not None  # Checked by the parent contract.
        index = int(raw_index)
        constraint = self._text_constraints[index] if self._text_constraints else ""
        return CrafTextEpisodeContext(
            world_preset=context.world_preset,
            instruction=context.instruction,
            env_state=context.env_state,
            text_constraint=constraint,
        )
