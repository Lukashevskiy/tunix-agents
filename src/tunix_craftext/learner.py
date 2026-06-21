"""Minimal Flax/Optax actor-critic learner for PPO smoke updates."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from flax.training.train_state import TrainState

from .algorithms import ppo_loss


class ActorCritic(nn.Module):
    """Small dense actor-critic used for deterministic PPO smoke training.

    :param actions: Number of discrete actions represented by policy logits.
    :param hidden: Width of the shared hidden layer.
    """

    actions: int
    hidden: int = 32

    @nn.compact
    def __call__(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return action logits and scalar value predictions.

        :param observation: Batch of feature vectors shaped ``[B, features]``.
        :returns: ``(logits, values)`` shaped ``[B, actions]`` and ``[B]``.
        """
        x = nn.relu(nn.Dense(self.hidden)(observation))
        return nn.Dense(self.actions)(x), nn.Dense(1)(x).squeeze(-1)


def create_state(key: jax.Array, observation_dim: int, actions: int) -> TrainState:
    """Create a deterministic actor-critic state with Adam optimizer.

    :param key: JAX PRNG key for parameter initialization.
    :param observation_dim: Width of one flattened learner feature vector.
    :param actions: Number of discrete environment actions.
    :returns: Initialized Flax state with an Optax Adam transform.
    """
    model = ActorCritic(actions)
    params = model.init(key, jnp.zeros((1, observation_dim)))["params"]
    return TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(3e-4))


def ppo_update(
    state: TrainState,
    observations: jax.Array,
    actions: jax.Array,
    old_log_prob: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Apply one clipped PPO update to a flat synthetic or encoded minibatch.

    :param state: Current actor-critic state.
    :param observations: Feature matrix shaped ``[B, features]``.
    :param actions: Sampled action ids shaped ``[B]``.
    :param old_log_prob: Behaviour-policy log probabilities shaped ``[B]``.
    :param advantages: GAE advantages shaped ``[B]``.
    :param returns: Bootstrap returns shaped ``[B]``.
    :returns: Updated state and finite, inspectable PPO loss metrics.
    """

    def loss_fn(params):
        logits, values = state.apply_fn({"params": params}, observations)
        log_prob = jax.nn.log_softmax(logits)[jnp.arange(actions.shape[0]), actions]
        entropy = -jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1)
        return ppo_loss(
            log_prob,
            old_log_prob,
            advantages,
            values,
            jnp.zeros_like(values),
            returns,
            0.2,
            0.5,
            entropy,
            0.01,
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), {**metrics, "loss": loss}
