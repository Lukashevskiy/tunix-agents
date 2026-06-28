"""PPO evaluation for real LLM actor/critic scores.

This module intentionally does not create a toy actor.  It consumes
``LlmActorScores`` produced by a real Tunix-backed actor and applies the same
token-level PPO objective used by the learner to verify actor log-probabilities,
critic values, entropy, returns and masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..artifacts.text_trajectory import TextTrajectoryBatch
from ..core.tensor_types import TokenBatchBool
from ..models.llm_actor import LlmActorScores
from ..models.tunix_actor import LlmActorTokenScores, LlmCriticValues, merge_actor_critic_scores
from .algorithms import masked_token_ppo_loss, masked_token_returns


@dataclass(frozen=True)
class LlmPpoEvaluation:
    """Inspectable PPO quantities for one real actor/critic scoring pass."""

    returns: jax.Array
    advantages: jax.Array
    loss: jax.Array
    metrics: dict[str, jax.Array]

    def validate(self, batch: TextTrajectoryBatch) -> None:
        """Validate that all evaluation arrays align with ``batch.token_ids``."""
        expected = tuple(batch.token_ids.shape)
        if tuple(self.returns.shape) != expected or tuple(self.advantages.shape) != expected:
            raise ValueError("returns and advantages must have shape [B, T]")
        if self.loss.shape != ():
            raise ValueError("loss must be scalar")


def evaluate_llm_actor_critic_ppo(
    batch: TextTrajectoryBatch,
    scores: LlmActorScores,
    *,
    learning_mask: TokenBatchBool | None = None,
    gamma: float = 0.99,
    clip_epsilon: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> LlmPpoEvaluation:
    """Compute token PPO loss from real LLM actor logprobs and critic values.

    :param batch: Text trajectory batch with replayed behaviour logprobs.
    :param scores: Real actor/critic scores recomputed for ``batch.token_ids``.
    :param learning_mask: Optional token mask. Defaults to ``batch.token_mask``.
    :param gamma: Discount for token reward-to-go.
    :param clip_epsilon: PPO clipping epsilon.
    :param value_coefficient: Value loss scale.
    :param entropy_coefficient: Entropy bonus scale.
    :returns: Returns, advantages, scalar loss and metrics.
    :raises ValueError: If score shapes, masks or finite checks fail.
    """
    batch.validate_static()
    scores.validate(batch.token_ids)
    mask = batch.token_mask if learning_mask is None else learning_mask
    if tuple(mask.shape) != tuple(batch.token_ids.shape):
        raise ValueError("learning_mask must have shape [B, T]")
    _validate_scores_are_finite(scores, mask)

    returns = masked_token_returns(batch.rewards, batch.token_mask, gamma)
    old_values = jax.lax.stop_gradient(scores.values)
    advantages = returns - old_values
    loss, metrics = masked_token_ppo_loss(
        new_log_prob=scores.token_logprobs,
        old_log_prob=batch.old_logprobs,
        advantages=advantages,
        new_value=scores.values,
        old_value=old_values,
        returns=returns,
        token_mask=mask,
        clip_epsilon=clip_epsilon,
        value_coefficient=value_coefficient,
        entropy=scores.entropy,
        entropy_coefficient=entropy_coefficient,
    )
    evaluation = LlmPpoEvaluation(
        returns=returns,
        advantages=advantages,
        loss=loss,
        metrics={**metrics, "loss": loss, "learned_tokens": jnp.sum(mask.astype(jnp.float32))},
    )
    evaluation.validate(batch)
    return evaluation


def evaluate_separate_llm_actor_critic_ppo(
    batch: TextTrajectoryBatch,
    actor_scores: LlmActorTokenScores,
    critic_values: LlmCriticValues,
    *,
    learning_mask: TokenBatchBool | None = None,
    gamma: float = 0.99,
    clip_epsilon: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> LlmPpoEvaluation:
    """Compute PPO loss from explicitly separate actor and critic outputs.

    This is the notebook/RLCluster-facing boundary: actor role recomputes
    ``new_logprobs`` and entropy, critic role recomputes values, then this
    helper merges the contract only at the objective boundary.
    """
    scores = merge_actor_critic_scores(actor_scores, critic_values, batch.token_ids)
    return evaluate_llm_actor_critic_ppo(
        batch,
        scores,
        learning_mask=learning_mask,
        gamma=gamma,
        clip_epsilon=clip_epsilon,
        value_coefficient=value_coefficient,
        entropy_coefficient=entropy_coefficient,
    )


def _validate_scores_are_finite(scores: LlmActorScores, mask: jax.Array) -> None:
    for name in ("token_logprobs", "values", "entropy"):
        values = getattr(scores, name)
        if not bool(jnp.all(jnp.isfinite(values[mask]))):
            raise ValueError(f"{name} must be finite on selected tokens")
    if bool(jnp.any(mask & ~scores.token_mask)):
        raise ValueError("learning_mask cannot select tokens missing from score token_mask")
