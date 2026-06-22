"""Convert inspectable host-side text replays into typed token learning batches."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .replay import ReplayArtifact


class TextTrajectoryError(ValueError):
    """Raised when replay evidence cannot become an unambiguous token batch."""


@dataclass(frozen=True)
class TextTrajectoryBatch:
    """Padded generated tokens with each environment reward on its final token.

    :ivar token_ids: Generated Qwen tokens shaped ``[B, T]``.
    :ivar old_logprobs: Behaviour-policy token log-probabilities shaped ``[B, T]``.
    :ivar token_mask: True exactly for generated (non-padding) tokens.
    :ivar policy_mask: Token mask excluding fallback decisions from policy learning.
    :ivar rewards: Environment reward placed on each valid sequence's final token.
    :ivar action_ids: CrafText discrete action selected per host decision, shape ``[B]``.
    :ivar terminated: Environment terminal state per host decision, shape ``[B]``.
    :ivar fallback_used: Whether an explicit fallback rather than model action stepped the env.
    """

    token_ids: jax.Array
    old_logprobs: jax.Array
    token_mask: jax.Array
    policy_mask: jax.Array
    rewards: jax.Array
    action_ids: jax.Array
    terminated: jax.Array
    fallback_used: jax.Array

    def validate(self) -> None:
        """Validate static batch/token axes at the host boundary."""
        token_shape = tuple(self.token_ids.shape)
        if len(token_shape) != 2 or token_shape[1] == 0:
            raise TextTrajectoryError("token_ids must have non-empty shape [B, T]")
        for name in ("old_logprobs", "token_mask", "policy_mask", "rewards"):
            if tuple(getattr(self, name).shape) != token_shape:
                raise TextTrajectoryError(f"{name} must have shape {token_shape}")
        batch_shape = token_shape[:1]
        for name in ("action_ids", "terminated", "fallback_used"):
            if tuple(getattr(self, name).shape) != batch_shape:
                raise TextTrajectoryError(f"{name} must have shape {batch_shape}")
        if bool(jnp.any(self.policy_mask & ~self.token_mask)):
            raise TextTrajectoryError("policy_mask cannot include padding tokens")


def text_trajectory_from_replay(artifact: ReplayArtifact) -> TextTrajectoryBatch:
    """Pad replay completions and attach each environment reward to its final token.

    Fallback steps remain in the artifact for audit but have an all-false ``policy_mask``;
    training cannot silently learn from an action that the model did not actually select.

    :raises TextTrajectoryError: If replay lacks token provenance or token/logprob alignment.
    """
    if not artifact.steps:
        raise TextTrajectoryError("replay must contain at least one step")
    token_sequences: list[tuple[int, ...]] = []
    logprob_sequences: list[tuple[float, ...]] = []
    for index, step in enumerate(artifact.steps):
        step_token_ids, step_token_logprobs = step.token_ids, step.token_logprobs
        if not step_token_ids or step_token_logprobs is None:
            raise TextTrajectoryError(f"replay step {index} lacks generated token provenance")
        if len(step_token_ids) != len(step_token_logprobs):
            raise TextTrajectoryError(f"replay step {index} token ids and logprobs disagree")
        token_sequences.append(step_token_ids)
        logprob_sequences.append(step_token_logprobs)

    batch_size = len(artifact.steps)
    width = max(len(tokens) for tokens in token_sequences)
    token_ids = jnp.zeros((batch_size, width), dtype=jnp.int32)
    old_logprobs = jnp.zeros((batch_size, width), dtype=jnp.float32)
    token_mask = jnp.zeros((batch_size, width), dtype=bool)
    rewards = jnp.zeros((batch_size, width), dtype=jnp.float32)
    for index, (tokens, logprobs, step) in enumerate(
        zip(token_sequences, logprob_sequences, artifact.steps, strict=True)
    ):
        length = len(tokens)
        token_ids = token_ids.at[index, :length].set(jnp.asarray(tokens, dtype=jnp.int32))
        old_logprobs = old_logprobs.at[index, :length].set(jnp.asarray(logprobs, dtype=jnp.float32))
        token_mask = token_mask.at[index, :length].set(True)
        rewards = rewards.at[index, length - 1].set(float(step.reward))

    fallback_used = jnp.asarray([step.fallback_used for step in artifact.steps], dtype=bool)
    batch = TextTrajectoryBatch(
        token_ids=token_ids,
        old_logprobs=old_logprobs,
        token_mask=token_mask,
        policy_mask=jnp.logical_and(token_mask, ~fallback_used[:, None]),
        rewards=rewards,
        action_ids=jnp.asarray([step.action_id for step in artifact.steps], dtype=jnp.int32),
        terminated=jnp.asarray([step.terminated for step in artifact.steps], dtype=bool),
        fallback_used=fallback_used,
    )
    batch.validate()
    return batch


jax.tree_util.register_dataclass(
    TextTrajectoryBatch,
    data_fields=[
        "token_ids",
        "old_logprobs",
        "token_mask",
        "policy_mask",
        "rewards",
        "action_ids",
        "terminated",
        "fallback_used",
    ],
    meta_fields=[],
)
