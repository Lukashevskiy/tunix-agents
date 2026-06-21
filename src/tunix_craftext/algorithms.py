"""Pure JAX PPO primitives: discounted returns and generalized advantage estimation."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def ppo_loss(
    new_log_prob: jax.Array,
    old_log_prob: jax.Array,
    advantages: jax.Array,
    new_value: jax.Array,
    old_value: jax.Array,
    returns: jax.Array,
    clip_epsilon: float,
    value_coefficient: float,
    entropy: jax.Array,
    entropy_coefficient: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return clipped PPO scalar loss and inspectable policy/value/KL metrics."""
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
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_value: jax.Array,
    terminated: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute time-major GAE advantages and returns for ``[T, B]`` arrays."""
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
