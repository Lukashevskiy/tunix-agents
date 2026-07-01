"""GRPO evaluation for externally collected vLLM/CrafText token evidence."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..core.tensor_types import TokenBatchBool
from ..models.tunix_actor import LlmActorTokenScores
from ..training.external_grpo import ExternalGrpoTokenBatch
from .algorithms import masked_token_grpo_loss


@dataclass(frozen=True)
class LlmGrpoEvaluation:
    """Inspectable GRPO quantities for one real actor scoring pass."""

    loss: jax.Array
    metrics: dict[str, jax.Array]

    def validate(self) -> None:
        """Validate scalar loss and finite metric values."""
        if self.loss.shape != ():
            raise ValueError("loss must be scalar")
        for name, value in self.metrics.items():
            if value.shape != ():
                raise ValueError(f"{name} must be scalar")
            if not bool(jnp.isfinite(value)):
                raise ValueError(f"{name} must be finite")


def evaluate_external_llm_actor_grpo(
    batch: ExternalGrpoTokenBatch,
    actor_scores: LlmActorTokenScores,
    *,
    learning_mask: TokenBatchBool | None = None,
    clip_epsilon: float = 0.2,
    entropy_coefficient: float = 0.0,
) -> LlmGrpoEvaluation:
    """Compute critic-free token GRPO loss from real actor log-probabilities.

    :param batch: Tokenized external GRPO evidence with replayed old log-probs
        and broadcast group advantages.
    :param actor_scores: Current actor token log-probs and entropy for
        ``batch.token_ids``.
    :param learning_mask: Optional token mask. Defaults to ``batch.token_mask``.
    :param clip_epsilon: PPO-style ratio clipping epsilon.
    :param entropy_coefficient: Entropy bonus scale.
    :returns: Scalar GRPO loss and inspectable metrics.
    :raises ValueError: If score/mask shapes disagree or selected scores are non-finite.
    """
    batch.validate_static()
    actor_scores.validate(batch.token_ids)
    mask = batch.token_mask if learning_mask is None else learning_mask
    if tuple(mask.shape) != tuple(batch.token_ids.shape):
        raise ValueError("learning_mask must have shape [N, T]")
    if bool(jnp.any(mask & ~actor_scores.token_mask)):
        raise ValueError("learning_mask cannot select tokens missing from actor scores")
    if not bool(jnp.all(jnp.isfinite(actor_scores.token_logprobs[mask]))):
        raise ValueError("token_logprobs must be finite on selected tokens")
    if not bool(jnp.all(jnp.isfinite(actor_scores.entropy[mask]))):
        raise ValueError("entropy must be finite on selected tokens")
    loss, metrics = masked_token_grpo_loss(
        new_log_prob=actor_scores.token_logprobs,
        old_log_prob=batch.old_logprobs,
        advantages=batch.advantages,
        token_mask=mask,
        clip_epsilon=clip_epsilon,
        entropy=actor_scores.entropy,
        entropy_coefficient=entropy_coefficient,
    )
    evaluation = LlmGrpoEvaluation(
        loss=loss,
        metrics={
            **metrics,
            "loss": loss,
            "learned_tokens": jnp.sum(mask.astype(jnp.float32)),
            "mean_sample_reward": jnp.mean(batch.sample_rewards),
        },
    )
    evaluation.validate()
    return evaluation
