"""Hybrid rollout contracts for host-orchestrated LLM-RL collection.

This module is the production-shaped data boundary between a dynamic agentic
text loop and JAX environment execution.  Host code owns prompt history,
Tunix/LLM generation and actor/critic RPCs; accelerator-friendly code owns
batched CrafText transitions via ``jax.vmap(adapter.step)``.

The older :mod:`tunix_craftext.rollout` collectors remain reference contracts
for fixed-shape CPU/JAX tests.  Real LLM PPO data must carry token log-probs,
critic values and two masks: generated-token masks for text padding and
step-valid masks for post-terminal rollout padding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .tensor_types import (
    ActionMask,
    BatchBool,
    BatchFloat,
    BatchInt,
    PromptTokenBatchBool,
    PromptTokenBatchInt,
    TimeBatchBool,
    TokenBatchBool,
    TokenBatchFloat,
    TokenBatchInt,
)


@dataclass(frozen=True)
class HybridPpoStep:
    """One PPO-ready host/accelerator step for a batch of agentic CrafText episodes.

    :ivar action_ids: Discrete environment action ids, shape ``[B]``.
    :ivar prompt_tokens: Padded prompt/input ids used by actor and critic, shape ``[B, P]``.
    :ivar prompt_token_mask: Valid prompt-token mask, shape ``[B, P]``.
    :ivar generation_tokens: Padded generated response/action tokens, shape ``[B, L]``.
    :ivar generation_token_mask: Valid generated-token mask, shape ``[B, L]``.
    :ivar actor_log_probs: Rollout/current actor token log-probs, shape ``[B, L]``.
    :ivar values: Critic values for the step state, shape ``[B]``.
    :ivar step_mask: ``True`` for episodes alive at this step, shape ``[B]``.
    :ivar action_mask: Optional legal-action mask observed before sampling, shape ``[B, A]``.
    """

    action_ids: BatchInt
    prompt_tokens: PromptTokenBatchInt
    prompt_token_mask: PromptTokenBatchBool
    generation_tokens: TokenBatchInt
    generation_token_mask: TokenBatchBool
    actor_log_probs: TokenBatchFloat
    values: BatchFloat
    step_mask: BatchBool
    action_mask: ActionMask | None = None

    def validate(self) -> None:
        """Validate static host-side shapes for this PPO rollout step.

        :raises ValueError: If any mandatory tensor lacks shape metadata or
            does not share the expected batch/token axes.
        """
        action_shape = _shape_of(self.action_ids, "action_ids")
        if len(action_shape) != 1:
            raise ValueError("action_ids must have shape [B]")
        batch_size = action_shape[0]

        prompt_shape = _shape_of(self.prompt_tokens, "prompt_tokens")
        if len(prompt_shape) != 2 or prompt_shape[0] != batch_size:
            raise ValueError("prompt_tokens must have shape [B, P]")
        _require_shape(self.prompt_token_mask, "prompt_token_mask", prompt_shape)

        generation_shape = _shape_of(self.generation_tokens, "generation_tokens")
        if len(generation_shape) != 2 or generation_shape[0] != batch_size:
            raise ValueError("generation_tokens must have shape [B, L]")
        for value, name in (
            (self.generation_token_mask, "generation_token_mask"),
            (self.actor_log_probs, "actor_log_probs"),
        ):
            _require_shape(value, name, generation_shape)

        _require_shape(self.values, "values", (batch_size,))
        _require_shape(self.step_mask, "step_mask", (batch_size,))
        if self.action_mask is not None:
            action_mask_shape = _shape_of(self.action_mask, "action_mask")
            if len(action_mask_shape) != 2 or action_mask_shape[0] != batch_size:
                raise ValueError("action_mask must have shape [B, A]")


@dataclass(frozen=True)
class HybridPpoTrajectory:
    """Fixed-horizon tuple of PPO-ready hybrid steps.

    The trajectory intentionally stores Python step objects instead of one huge
    padded structure because prompts and completions are host-managed evidence.
    ``step_masks`` is still provided as a stacked ``[T, B]`` tensor for learner
    code that needs fast masking.
    """

    steps: tuple[HybridPpoStep, ...]
    step_masks: TimeBatchBool

    def validate(self) -> None:
        """Validate that all steps share batch axes and match ``step_masks``."""
        if not self.steps:
            raise ValueError("trajectory must contain at least one step")
        for step in self.steps:
            step.validate()

        expected_shape = (len(self.steps), _shape_of(self.steps[0].action_ids, "action_ids")[0])
        _require_shape(self.step_masks, "step_masks", expected_shape)
        for index, step in enumerate(self.steps):
            _require_shape(step.step_mask, f"steps[{index}].step_mask", expected_shape[1:])


def hybrid_trajectory_from_steps(steps: Sequence[HybridPpoStep]) -> HybridPpoTrajectory:
    """Build a validated trajectory and stack per-step alive masks.

    :param steps: Ordered rollout steps.
    :returns: ``HybridPpoTrajectory`` with ``step_masks`` of shape ``[T, B]``.
    :raises ValueError: If no steps are provided or shape validation fails.
    """
    if not steps:
        raise ValueError("steps must contain at least one step")
    for step in steps:
        step.validate()
    expected_batch = _shape_of(steps[0].step_mask, "steps[0].step_mask")
    for index, step in enumerate(steps[1:], start=1):
        _require_shape(step.step_mask, f"steps[{index}].step_mask", expected_batch)
    trajectory = HybridPpoTrajectory(
        steps=tuple(steps),
        step_masks=jnp.stack([jnp.asarray(step.step_mask, dtype=bool) for step in steps], axis=0),
    )
    trajectory.validate()
    return trajectory


def compute_masked_step_token_ppo_loss(
    actor_log_probs: TokenBatchFloat,
    old_log_probs: TokenBatchFloat,
    generation_token_mask: TokenBatchBool,
    advantages: BatchFloat,
    step_valid_mask: BatchBool,
    *,
    clip_epsilon: float = 0.2,
) -> jax.Array:
    """Compute clipped actor PPO loss with token and step masks.

    This is the small TDD primitive behind the hybrid rollout contract.  It
    ignores generated-token padding and ignores whole rows whose episode had
    already terminated before this step.

    :param actor_log_probs: Current actor token log-probs, shape ``[B, L]``.
    :param old_log_probs: Rollout actor token log-probs, shape ``[B, L]``.
    :param generation_token_mask: ``True`` for real generated tokens, shape ``[B, L]``.
    :param advantages: Per-row advantages, shape ``[B]``.
    :param step_valid_mask: ``True`` for rows alive at this step, shape ``[B]``.
    :param clip_epsilon: PPO clipping range.
    :returns: Scalar JAX loss.
    :raises ValueError: If shapes are inconsistent, no valid rows/tokens exist,
        or ``clip_epsilon`` is not positive.
    """
    if clip_epsilon <= 0:
        raise ValueError("clip_epsilon must be positive")
    token_shape = _shape_of(actor_log_probs, "actor_log_probs")
    if len(token_shape) != 2:
        raise ValueError("actor_log_probs must have shape [B, L]")
    for value, name in (
        (old_log_probs, "old_log_probs"),
        (generation_token_mask, "generation_token_mask"),
    ):
        _require_shape(value, name, token_shape)
    batch_size = token_shape[0]
    _require_shape(advantages, "advantages", (batch_size,))
    _require_shape(step_valid_mask, "step_valid_mask", (batch_size,))

    token_mask = jnp.asarray(generation_token_mask, dtype=jnp.float32)
    step_mask = jnp.asarray(step_valid_mask, dtype=jnp.float32)
    valid_token_mask = token_mask * step_mask[:, None]
    if not bool(jnp.any(valid_token_mask)):
        raise ValueError("at least one valid generated token is required")

    ratio = jnp.exp(jnp.asarray(actor_log_probs) - jnp.asarray(old_log_probs))
    expanded_advantages = jnp.asarray(advantages, dtype=jnp.float32)[:, None]
    unclipped = ratio * expanded_advantages
    clipped = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * expanded_advantages
    token_losses = -jnp.minimum(unclipped, clipped) * valid_token_mask

    per_row_token_count = jnp.maximum(jnp.sum(token_mask, axis=-1), 1.0)
    per_row_loss = jnp.sum(token_losses, axis=-1) / per_row_token_count
    valid_row_count = jnp.maximum(jnp.sum(step_mask), 1.0)
    return jnp.sum(per_row_loss) / valid_row_count


def _shape_of(value: object, field_name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"{field_name} must expose static shape metadata")
    return tuple(shape)


def _require_shape(value: object, field_name: str, expected_shape: tuple[int, ...]) -> None:
    actual_shape = _shape_of(value, field_name)
    if actual_shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {expected_shape}, got {actual_shape}")
