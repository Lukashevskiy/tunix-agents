"""Pure JAX PPO primitives: discounted returns and generalized advantage estimation.

This module contains low-level PPO losses and trajectory utilities that operate
on explicit JAX arrays. It is intentionally kept small and stateless so it can
be reused across different actor-critic training pipelines.
"""

import jax
import jax.numpy as jnp

from ..tensor_types import (
    BatchFloat,
    TimeBatchBool,
    TimeBatchFloat,
    TokenBatchBool,
    TokenBatchFloat,
)


def ppo_loss(
    new_log_prob: BatchFloat,
    old_log_prob: BatchFloat,
    advantages: BatchFloat,
    new_value: BatchFloat,
    old_value: BatchFloat,
    returns: BatchFloat,
    clip_epsilon: float,
    value_coefficient: float,
    entropy: BatchFloat,
    entropy_coefficient: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return clipped PPO scalar loss and inspectable policy/value/KL metrics.

    :param new_log_prob: jax.Array input value
    :param old_log_prob: jax.Array input value
    :param advantages: jax.Array input value
    :param new_value: jax.Array input value
    :param old_value: jax.Array input value
    :param returns: jax.Array input value
    :param clip_epsilon: float input value
    :param value_coefficient: float input value
    :param entropy: jax.Array input value
    :param entropy_coefficient: float input value
    :returns: tuple[jax.Array, dict[str, jax.Array]]

    Example:
        >>> loss, metrics = ppo_loss(
        ...     new_log_prob, old_log_prob, advantages, new_value, old_value,
        ...     returns, 0.2, 0.5, entropy, 0.01,
        ... )
    """
    ratio = jnp.exp(new_log_prob - old_log_prob)
    policy = -jnp.mean(
        jnp.minimum(
            ratio * advantages, jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        )
    )
    clipped_value = old_value + jnp.clip(new_value - old_value, -clip_epsilon, clip_epsilon)
    value = 0.5 * jnp.mean(jnp.maximum((new_value - returns) ** 2, (clipped_value - returns) ** 2))
    entropy_loss = -jnp.mean(entropy)
    loss = policy + value_coefficient * value + entropy_coefficient * entropy_loss
    return loss, {
        "policy_loss": policy,
        "value_loss": value,
        "entropy": jnp.mean(entropy),
        "approx_kl": jnp.mean(old_log_prob - new_log_prob),
    }


def generalized_advantage_estimation(
    rewards: TimeBatchFloat,
    values: TimeBatchFloat,
    bootstrap_value: BatchFloat,
    terminated: TimeBatchBool,
    gamma: float,
    gae_lambda: float,
) -> tuple[TimeBatchFloat, TimeBatchFloat]:
    """Compute time-major GAE advantages and returns for ``[T, B]`` arrays.

    :param rewards: jax.Array input value
    :param values: jax.Array input value
    :param bootstrap_value: jax.Array input value
    :param terminated: jax.Array input value
    :param gamma: float input value
    :param gae_lambda: float input value
    :returns: tuple[jax.Array, jax.Array]

    Example:
        >>> advantages, returns = generalized_advantage_estimation(
        ...     rewards, values, bootstrap_value, terminated, gamma=0.99, gae_lambda=0.95,
        ... )
    """
    if rewards.ndim != 2 or values.shape != rewards.shape or terminated.shape != rewards.shape:
        raise ValueError("rewards, values and terminated must all have shape [T, B]")
    if bootstrap_value.shape != rewards.shape[1:]:
        raise ValueError("bootstrap_value must have shape [B]")

    def step(carry, inputs):
        reward, value, next_value, done = inputs
        delta = reward + gamma * (1.0 - done) * next_value - value
        advantage = delta + gamma * gae_lambda * (1.0 - done) * carry
        return advantage, advantage

    next_values = jnp.concatenate((values[1:], bootstrap_value[None]), axis=0)
    _, advantages = jax.lax.scan(
        step,
        jnp.zeros_like(bootstrap_value),
        (rewards[::-1], values[::-1], next_values[::-1], terminated[::-1]),
    )
    advantages = advantages[::-1]
    return advantages, advantages + values


def masked_token_returns(
    rewards: TokenBatchFloat, token_mask: TokenBatchBool, gamma: float
) -> TokenBatchFloat:
    """Compute reward-to-go for padded token sequences shaped ``[B, T]``.

    Padding tokens have return zero and reset the backward carry, allowing host-side
    text completions of unequal length to share a static learning batch.

    :param rewards: Per-token reward array shaped ``[B, T]``.
    :param token_mask: Boolean mask marking generated (non-padding) tokens shaped ``[B, T]``.
    :param gamma: Discount factor in [0, 1].
    :returns: Returns-to-go array shaped ``[B, T]``.
    :raises ValueError: If input shapes are incompatible or gamma is outside [0,1].

    Example:
        >>> returns = masked_token_returns(rewards, token_mask, gamma=0.99)
    """
    if rewards.shape != token_mask.shape or rewards.ndim != 2:
        raise ValueError("rewards and token_mask must have identical shape [B, T]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    def step(carry: jax.Array, inputs: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        reward, valid = inputs
        returned = jnp.where(valid, reward + gamma * carry, 0.0)
        return returned, returned

    _, reversed_returns = jax.lax.scan(
        step,
        jnp.zeros(rewards.shape[0], dtype=rewards.dtype),
        (rewards.T[::-1], token_mask.T[::-1]),
    )
    return reversed_returns[::-1].T


def masked_token_ppo_loss(
    new_log_prob: TokenBatchFloat,
    old_log_prob: TokenBatchFloat,
    advantages: TokenBatchFloat,
    new_value: TokenBatchFloat,
    old_value: TokenBatchFloat,
    returns: TokenBatchFloat,
    token_mask: TokenBatchBool,
    clip_epsilon: float,
    value_coefficient: float,
    entropy: TokenBatchFloat,
    entropy_coefficient: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Apply PPO only to valid token positions in ``[B, T]`` text trajectories.

    :param new_log_prob: Token-wise new log-probabilities shaped ``[B, T]``.
    :param old_log_prob: Token-wise old log-probabilities shaped ``[B, T]``.
    :param advantages: Token-wise advantage estimates shaped ``[B, T]``.
    :param new_value: Token-wise current value predictions shaped ``[B, T]``.
    :param old_value: Token-wise previous value predictions shaped ``[B, T]``.
    :param returns: Token-wise bootstrap returns shaped ``[B, T]``.
    :param token_mask: Boolean mask selecting valid policy tokens shaped ``[B, T]``.
    :param clip_epsilon: PPO clipping epsilon.
    :param value_coefficient: Value loss scale.
    :param entropy: Token-wise entropy values shaped ``[B, T]``.
    :param entropy_coefficient: Entropy bonus scale.
    :returns: Tuple of scalar loss and per-term metrics.
    :raises ValueError: If token fields differ in shape or no policy token is valid.

    Example:
        >>> loss, metrics = masked_token_ppo_loss(
        ...     new_log_prob, old_log_prob, advantages, new_value, old_value,
        ...     returns, token_mask, 0.2, 0.5, entropy, 0.01,
        ... )
    """
    fields = (new_log_prob, old_log_prob, advantages, new_value, old_value, returns, entropy)
    if token_mask.ndim != 2 or any(field.shape != token_mask.shape for field in fields):
        raise ValueError("all token PPO fields and token_mask must have shape [B, T]")
    if not bool(jnp.any(token_mask)):
        raise ValueError("token_mask must select at least one token")
    return ppo_loss(
        new_log_prob[token_mask],
        old_log_prob[token_mask],
        advantages[token_mask],
        new_value[token_mask],
        old_value[token_mask],
        returns[token_mask],
        clip_epsilon,
        value_coefficient,
        entropy[token_mask],
        entropy_coefficient,
    )
