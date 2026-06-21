"""Minimal Flax/Optax actor-critic learner for PPO smoke updates."""
from __future__ import annotations
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
from .algorithms import ppo_loss
class ActorCritic(nn.Module):
    actions: int
    hidden: int = 32
    @nn.compact
    def __call__(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = nn.relu(nn.Dense(self.hidden)(observation))
        return nn.Dense(self.actions)(x), nn.Dense(1)(x).squeeze(-1)

def create_state(key: jax.Array, observation_dim: int, actions: int) -> TrainState:
    model = ActorCritic(actions)
    params = model.init(key, jnp.zeros((1, observation_dim)))["params"]
    return TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(3e-4))

def ppo_update(state: TrainState, observations: jax.Array, actions: jax.Array, old_log_prob: jax.Array, advantages: jax.Array, returns: jax.Array) -> tuple[TrainState, dict[str, jax.Array]]:
    def loss_fn(params):
        logits, values = state.apply_fn({"params": params}, observations)
        log_prob = jax.nn.log_softmax(logits)[jnp.arange(actions.shape[0]), actions]
        entropy = -jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1)
        return ppo_loss(log_prob, old_log_prob, advantages, values, jnp.zeros_like(values), returns, .2, .5, entropy, .01)
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), {**metrics, "loss": loss}
