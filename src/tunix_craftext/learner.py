"""Minimal Flax/Optax actor-critic learner for PPO smoke updates.

This module provides a compact deterministic actor-critic network and
helper functions for initializing a Flax TrainState and applying a single
clipped PPO update. It is built for lightweight smoke tests, example
workflows, and flat feature inputs rather than production-scale RL.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from flax.training.train_state import TrainState

from .algorithms import ppo_loss


class ActorCritic(nn.Module):
    """Small dense actor-critic used for deterministic PPO smoke training.

    The model shares one hidden layer and produces both policy logits and
    a scalar state value for advantage estimation.

    :param actions: Number of discrete actions represented by policy logits.
        The network returns logits of shape ``[B, actions]``.
    :param hidden: Width of the shared hidden layer.
        Larger values increase model capacity at the cost of compute.
    """

    actions: int
    hidden: int = 32

    @nn.compact
    def __call__(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return action logits and scalar value predictions.

        The policy head produces unnormalized logits for discrete action
        selection, while the value head predicts a single scalar per example.

        :param observation: Batch of feature vectors shaped ``[B, features]``.
        :returns: ``(logits, values)`` where logits shape is ``[B, actions]``
        and values shape is ``[B]``.

        Example:
        >>> logits, values = model(observation)
        >>> logits.shape == (batch_size, actions)
        >>> values.shape == (batch_size,)
        """
        x = nn.relu(nn.Dense(self.hidden)(observation))
        return nn.Dense(self.actions)(x), nn.Dense(1)(x).squeeze(-1)


def create_state(key: jax.Array, observation_dim: int, actions: int) -> TrainState:
    """Create a deterministic actor-critic state with Adam optimizer.

    Builds the Flax model parameters from a dummy observation and returns a
    TrainState containing the model apply function, initialized params, and
    an Optax Adam optimizer.

    :param key: JAX PRNG key for parameter initialization.
    :param observation_dim: Width of one flattened learner feature vector.
    The model expects inputs shaped ``[B, observation_dim]``.
    :param actions: Number of discrete environment actions.
    :returns: Initialized Flax TrainState with an Optax Adam transform.

    Example:
    >>> state = create_state(key, observation_dim=16, actions=4)
    >>> state.params is not None
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

    This function computes the PPO loss, gradients, and returns an updated
    TrainState together with a metrics dictionary. It is intended for
    deterministic unit tests and example actor-critic training loops.

    :param state: Current actor-critic TrainState.
    :param observations: Feature matrix shaped ``[B, features]``.
        ``state.apply_fn`` is called with this input.
    :param actions: Sampled action ids shaped ``[B]``.
        Each value must be in ``[0, actions)``.
    :param old_log_prob: Behaviour-policy log probabilities shaped ``[B]``.
    :param advantages: GAE advantages shaped ``[B]``.
    :param returns: Bootstrap returns shaped ``[B]``.
    :returns: ``(new_state, metrics)`` where ``new_state`` is the updated
        TrainState and ``metrics`` contains inspectable PPO quantities.

    Example:
        >>> new_state, metrics = ppo_update(
        ...     state,
        ...     observations,
        ...     actions,
        ...     old_log_prob,
        ...     advantages,
        ...     returns,
        ... )
        >>> assert 'loss' in metrics
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
