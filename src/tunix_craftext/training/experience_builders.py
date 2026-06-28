"""Experience Builder layer for agentic PPO trajectories.

This module is the middleware between host-orchestrated agentic rollouts and
Tunix token-level learners.  Rollout collection produces universal MDP steps:
one dense environment reward and one critic value per environment step, plus the
generated tokens/log-probs that caused that step.  PPO training consumes token
examples.  The builder therefore computes GAE over the MDP time axis ``[T, B]``
and broadcasts each scalar step advantage back onto that step's valid generated
tokens.

No Tunix imports live here.  Keeping this layer framework-light lets CPU tests
validate the math and masks without allocating models or accelerators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..core.tensor_types import (
    ActionMask,
    BatchBool,
    BatchFloat,
    PromptTokenBatchBool,
    PromptTokenBatchInt,
    TimeBatchBool,
    TimeBatchFloat,
    TokenBatchBool,
    TokenBatchFloat,
    TokenBatchInt,
)


@dataclass(frozen=True)
class UniversalMDPStep:
    """Algorithm-agnostic host/JAX rollout step for MDP LLM agents.

    :ivar prompt_tokens: Padded prompt/input ids, shape ``[B, P]``.
    :ivar generation_tokens: Padded generated response/action tokens, shape ``[B, L]``.
    :ivar generation_mask: Valid generated-token mask, shape ``[B, L]``.
    :ivar actor_log_probs: Rollout actor token log-probs, shape ``[B, L]``.
    :ivar step_mask: ``True`` while the MDP episode is alive before this step, shape ``[B]``.
    :ivar reward: Dense environment reward for this MDP step, shape ``[B]``.
    :ivar value: Optional critic value ``V(s_t)``, shape ``[B]``.
    :ivar prompt_mask: Optional valid prompt-token mask, shape ``[B, P]``.
    :ivar policy_token_mask: Optional loss mask over generated tokens, shape ``[B, L]``.
    :ivar action_mask: Optional legal-action mask before sampling, shape ``[B, A]``.
    """

    prompt_tokens: PromptTokenBatchInt
    generation_tokens: TokenBatchInt
    generation_mask: TokenBatchBool
    actor_log_probs: TokenBatchFloat
    step_mask: BatchBool
    reward: BatchFloat
    value: BatchFloat | None = None
    prompt_mask: PromptTokenBatchBool | None = None
    policy_token_mask: TokenBatchBool | None = None
    action_mask: ActionMask | None = None

    @property
    def actor_loss_token_mask(self) -> TokenBatchBool:
        """Return generated-token mask after optional policy masking."""
        if self.policy_token_mask is None:
            return self.generation_mask
        return jnp.logical_and(self.generation_mask, self.policy_token_mask)

    def validate(self) -> None:
        """Validate batch/token shape contracts for one universal MDP step."""
        prompt_shape = _shape_of(self.prompt_tokens, "prompt_tokens")
        if len(prompt_shape) != 2:
            raise ValueError("prompt_tokens must have shape [B, P]")
        batch_size = prompt_shape[0]
        if self.prompt_mask is not None:
            _require_shape(self.prompt_mask, "prompt_mask", prompt_shape)

        generation_shape = _shape_of(self.generation_tokens, "generation_tokens")
        if len(generation_shape) != 2 or generation_shape[0] != batch_size:
            raise ValueError("generation_tokens must have shape [B, L]")
        for value, name in (
            (self.generation_mask, "generation_mask"),
            (self.actor_log_probs, "actor_log_probs"),
        ):
            _require_shape(value, name, generation_shape)
        if self.policy_token_mask is not None:
            _require_shape(self.policy_token_mask, "policy_token_mask", generation_shape)

        for value, name in ((self.step_mask, "step_mask"), (self.reward, "reward")):
            _require_shape(value, name, (batch_size,))
        if self.value is not None:
            _require_shape(self.value, "value", (batch_size,))
        if self.action_mask is not None:
            action_mask_shape = _shape_of(self.action_mask, "action_mask")
            if len(action_mask_shape) != 2 or action_mask_shape[0] != batch_size:
                raise ValueError("action_mask must have shape [B, A]")


@dataclass(frozen=True)
class TokenPPOExperience:
    """Flattened token-level PPO batch produced by ``PpoExperienceBuilder``.

    Time and batch axes are flattened into rows ``N = T * B``.  Invalid
    post-terminal rows remain present for provenance but have empty token masks.
    """

    prompt_tokens: PromptTokenBatchInt
    prompt_mask: PromptTokenBatchBool
    generation_tokens: TokenBatchInt
    completion_mask: TokenBatchBool
    old_per_token_logps: TokenBatchFloat
    advantages: TokenBatchFloat
    returns: TokenBatchFloat
    step_advantages: BatchFloat
    step_returns: BatchFloat
    step_rewards: BatchFloat
    step_values: BatchFloat
    step_mask: BatchBool

    def validate(self) -> None:
        """Validate flattened PPO experience shapes."""
        prompt_shape = _shape_of(self.prompt_tokens, "prompt_tokens")
        if len(prompt_shape) != 2:
            raise ValueError("prompt_tokens must have shape [N, P]")
        rows = prompt_shape[0]
        _require_shape(self.prompt_mask, "prompt_mask", prompt_shape)

        generation_shape = _shape_of(self.generation_tokens, "generation_tokens")
        if len(generation_shape) != 2 or generation_shape[0] != rows:
            raise ValueError("generation_tokens must have shape [N, L]")
        for value, name in (
            (self.completion_mask, "completion_mask"),
            (self.old_per_token_logps, "old_per_token_logps"),
            (self.advantages, "advantages"),
            (self.returns, "returns"),
        ):
            _require_shape(value, name, generation_shape)
        for value, name in (
            (self.step_advantages, "step_advantages"),
            (self.step_returns, "step_returns"),
            (self.step_rewards, "step_rewards"),
            (self.step_values, "step_values"),
            (self.step_mask, "step_mask"),
        ):
            _require_shape(value, name, (rows,))


class ExperienceBuilder(ABC):
    """Strategy interface for converting MDP steps into learner examples."""

    @abstractmethod
    def build(self, trajectory_steps: Sequence[UniversalMDPStep]) -> TokenPPOExperience:
        """Build algorithm-specific training examples from universal rollout steps."""


@dataclass(frozen=True)
class PpoExperienceBuilder(ExperienceBuilder):
    """Build token-level PPO experience with MDP-time GAE and broadcasting."""

    gamma: float = 1.0
    gae_lambda: float = 0.95

    def build(self, trajectory_steps: Sequence[UniversalMDPStep]) -> TokenPPOExperience:
        """Build flattened token PPO experience from universal MDP steps."""
        if not trajectory_steps:
            raise ValueError("trajectory_steps must contain at least one step")
        for step in trajectory_steps:
            step.validate()
            if step.value is None:
                raise ValueError("PpoExperienceBuilder requires critic value for every step")

        prompt_tokens = jnp.stack([step.prompt_tokens for step in trajectory_steps], axis=0)
        generation_tokens = jnp.stack(
            [step.generation_tokens for step in trajectory_steps],
            axis=0,
        )
        generation_masks = jnp.stack(
            [
                jnp.logical_and(step.actor_loss_token_mask, step.step_mask[:, None])
                for step in trajectory_steps
            ],
            axis=0,
        )
        old_logps = jnp.stack([step.actor_log_probs for step in trajectory_steps], axis=0)
        step_masks = jnp.stack([step.step_mask for step in trajectory_steps], axis=0)
        rewards = jnp.stack([step.reward for step in trajectory_steps], axis=0)
        values = jnp.stack(
            [step.value for step in trajectory_steps if step.value is not None],
            axis=0,
        )
        prompt_masks = jnp.stack([_prompt_mask(step) for step in trajectory_steps], axis=0)

        advantages, returns = compute_mdp_gae(
            rewards=rewards,
            values=values,
            step_masks=step_masks,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        token_advantages = broadcast_step_values_to_tokens(advantages, generation_masks)
        token_returns = broadcast_step_values_to_tokens(returns, generation_masks)

        experience = TokenPPOExperience(
            prompt_tokens=_flatten_time_batch(prompt_tokens),
            prompt_mask=_flatten_time_batch(prompt_masks),
            generation_tokens=_flatten_time_batch(generation_tokens),
            completion_mask=_flatten_time_batch(generation_masks),
            old_per_token_logps=_flatten_time_batch(old_logps),
            advantages=_flatten_time_batch(token_advantages),
            returns=_flatten_time_batch(token_returns),
            step_advantages=advantages.reshape((-1,)),
            step_returns=returns.reshape((-1,)),
            step_rewards=rewards.reshape((-1,)),
            step_values=values.reshape((-1,)),
            step_mask=step_masks.reshape((-1,)),
        )
        experience.validate()
        return experience


def compute_mdp_gae(
    *,
    rewards: TimeBatchFloat,
    values: TimeBatchFloat,
    step_masks: TimeBatchBool,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    bootstrap_values: BatchFloat | None = None,
) -> tuple[TimeBatchFloat, TimeBatchFloat]:
    """Compute GAE across the MDP time axis, not the token axis.

    ``step_masks[t, b]`` marks whether row ``b`` was alive before environment
    step ``t``.  The next-state continuation is therefore ``step_masks[t + 1]``;
    the final step uses ``0`` continuation unless explicit bootstrap values are
    supplied.
    """
    reward_shape = _shape_of(rewards, "rewards")
    if len(reward_shape) != 2:
        raise ValueError("rewards must have shape [T, B]")
    _require_shape(values, "values", reward_shape)
    _require_shape(step_masks, "step_masks", reward_shape)
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if not 0 <= gae_lambda <= 1:
        raise ValueError("gae_lambda must be in [0, 1]")

    time_steps, batch_size = reward_shape
    bootstrap = (
        jnp.zeros((batch_size,), dtype=jnp.float32)
        if bootstrap_values is None
        else jnp.asarray(bootstrap_values, dtype=jnp.float32)
    )
    _require_shape(bootstrap, "bootstrap_values", (batch_size,))

    rewards_f = jnp.asarray(rewards, dtype=jnp.float32)
    values_f = jnp.asarray(values, dtype=jnp.float32)
    masks_f = jnp.asarray(step_masks, dtype=jnp.float32)
    next_values = jnp.concatenate([values_f[1:], bootstrap[None, :]], axis=0)
    next_masks = jnp.concatenate(
        [masks_f[1:], jnp.zeros((1, batch_size), dtype=jnp.float32)],
        axis=0,
    )
    deltas = rewards_f + gamma * next_values * next_masks - values_f

    def scan_fn(
        carry: BatchFloat,
        inputs: tuple[BatchFloat, BatchFloat],
    ) -> tuple[BatchFloat, BatchFloat]:
        delta_t, mask_t = inputs
        advantage_t = (delta_t + gamma * gae_lambda * carry) * mask_t
        return advantage_t, advantage_t

    _, reversed_advantages = jax.lax.scan(
        scan_fn,
        jnp.zeros((batch_size,), dtype=jnp.float32),
        (deltas[::-1], masks_f[::-1]),
    )
    advantages = reversed_advantages[::-1]
    returns = advantages + values_f
    return advantages, returns


def broadcast_step_values_to_tokens(
    step_values: TimeBatchFloat,
    generation_mask: TokenBatchBool,
) -> TokenBatchFloat:
    """Broadcast scalar ``[T, B]`` values to valid generated tokens ``[T, B, L]``."""
    step_shape = _shape_of(step_values, "step_values")
    mask_shape = _shape_of(generation_mask, "generation_mask")
    if len(step_shape) != 2:
        raise ValueError("step_values must have shape [T, B]")
    if len(mask_shape) != 3 or mask_shape[:2] != step_shape:
        raise ValueError("generation_mask must have shape [T, B, L]")
    return jnp.asarray(step_values, dtype=jnp.float32)[:, :, None] * jnp.asarray(
        generation_mask,
        dtype=jnp.float32,
    )


def _prompt_mask(step: UniversalMDPStep) -> PromptTokenBatchBool:
    if step.prompt_mask is not None:
        return step.prompt_mask
    return jnp.ones_like(step.prompt_tokens, dtype=bool)


def _flatten_time_batch(value: jax.Array) -> jax.Array:
    shape = _shape_of(value, "value")
    if len(shape) < 2:
        raise ValueError("value must have at least [T, B] axes")
    return value.reshape((shape[0] * shape[1], *shape[2:]))


def _shape_of(value: object, field_name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"{field_name} must expose static shape metadata")
    return tuple(shape)


def _require_shape(value: object, field_name: str, expected_shape: tuple[int, ...]) -> None:
    actual_shape = _shape_of(value, field_name)
    if actual_shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {expected_shape}, got {actual_shape}")
