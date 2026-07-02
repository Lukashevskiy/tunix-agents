"""Minimal Flax/Optax actor-critic learners for PPO smoke updates.

This module provides compact deterministic actor-critic networks and helper
functions for initializing Flax TrainStates and applying clipped PPO updates.
It is built for lightweight smoke tests, example workflows, and typed data-flow
validation rather than production-scale Qwen/RLCluster training.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from flax.training.train_state import TrainState

from ..artifacts.text_trajectory import TextTrajectoryBatch
from ..core.tensor_types import (
    ActionLogits,
    BatchFeatureFloat,
    BatchFloat,
    BatchInt,
    JaxKey,
    PromptTokenBatchBool,
    PromptTokenBatchInt,
    ScalarFloat,
    TokenBatchBool,
    TokenBatchFloat,
    TokenBatchInt,
    TokenBatchLogits,
)
from ..models.tunix_actor import LlmActorTokenScores
from ..training.external_grpo import ExternalGrpoTokenBatch
from .algorithms import (
    masked_token_grpo_loss,
    masked_token_ppo_loss,
    masked_token_returns,
    ppo_loss,
)


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
    def __call__(self, observation: BatchFeatureFloat) -> tuple[ActionLogits, BatchFloat]:
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


def create_state(key: JaxKey, observation_dim: int, actions: int) -> TrainState:
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
    observations: BatchFeatureFloat,
    actions: BatchInt,
    old_log_prob: BatchFloat,
    advantages: BatchFloat,
    returns: BatchFloat,
) -> tuple[TrainState, dict[str, ScalarFloat]]:
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


class PromptConditionedTokenActorCritic(nn.Module):
    """Tiny token actor-critic conditioned on the rendered prompt token set.

    This model is intentionally small and bucketed: it is a trainable smoke
    bridge from :class:`TextTrajectoryBatch` to PPO loss, not a replacement for
    Qwen actor logprob recomputation inside Tunix ``RLCluster``. Token ids are
    mapped to ``token_bucket_count`` buckets so notebooks can run even when Qwen
    token ids are large.

    :param token_bucket_count: Number of token buckets predicted by the actor.
    :param hidden: Width of the shared token/prompt representation.
    """

    token_bucket_count: int
    hidden: int = 64

    @nn.compact
    def __call__(
        self,
        token_ids: TokenBatchInt,
        prompt_token_ids: PromptTokenBatchInt,
        prompt_token_mask: PromptTokenBatchBool,
    ) -> tuple[TokenBatchLogits, TokenBatchFloat]:
        """Return per-token bucket logits and critic values.

        :param token_ids: Generated token ids shaped ``[B, T]``.
        :param prompt_token_ids: Prompt token ids shaped ``[B, P]``.
        :param prompt_token_mask: Boolean prompt mask shaped ``[B, P]``.
        :returns: ``(logits, values)`` with logits ``[B, T, token_bucket_count]``
            and values ``[B, T]``.
        """
        token_buckets = jnp.mod(token_ids, self.token_bucket_count)
        prompt_buckets = jnp.mod(prompt_token_ids, self.token_bucket_count)
        embedding = nn.Embed(self.token_bucket_count, self.hidden)
        token_features = embedding(token_buckets)
        prompt_features = embedding(prompt_buckets)
        prompt_mask = prompt_token_mask[..., None].astype(prompt_features.dtype)
        prompt_count = jnp.maximum(jnp.sum(prompt_mask, axis=1), 1.0)
        prompt_context = jnp.sum(prompt_features * prompt_mask, axis=1) / prompt_count
        prompt_context = jnp.broadcast_to(prompt_context[:, None, :], token_features.shape)
        features = jnp.concatenate([token_features, prompt_context], axis=-1)
        hidden = nn.relu(nn.Dense(self.hidden)(features))
        return nn.Dense(self.token_bucket_count)(hidden), nn.Dense(1)(hidden).squeeze(-1)


def create_token_state(
    key: JaxKey,
    *,
    token_bucket_count: int = 512,
    hidden: int = 64,
    learning_rate: float = 3e-4,
) -> TrainState:
    """Create a prompt-conditioned token actor-critic TrainState.

    :param key: JAX PRNG key for parameter initialization.
    :param token_bucket_count: Number of actor output buckets. Generated Qwen
        ids are mapped with modulo into these buckets for smoke training.
    :param hidden: Width of the shared token/prompt representation.
    :param learning_rate: Adam optimizer learning rate.
    :returns: Initialized Flax TrainState.
    :raises ValueError: If dimensions or learning rate are non-positive.
    """
    if token_bucket_count <= 1 or hidden <= 0 or learning_rate <= 0:
        raise ValueError("token_bucket_count, hidden and learning_rate must be positive")
    model = PromptConditionedTokenActorCritic(token_bucket_count, hidden)
    params = model.init(
        key,
        jnp.zeros((1, 1), dtype=jnp.int32),
        jnp.zeros((1, 1), dtype=jnp.int32),
        jnp.ones((1, 1), dtype=bool),
    )["params"]
    return TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(learning_rate))


def token_actor_critic_outputs(
    state: TrainState, batch: TextTrajectoryBatch
) -> tuple[TokenBatchFloat, TokenBatchFloat, TokenBatchFloat]:
    """Recompute trainable token logprobs, critic values and entropies.

    :param state: Prompt-conditioned token actor-critic state.
    :param batch: Text trajectory batch with generated and prompt tokens.
    :returns: ``(selected_logprobs, values, entropy)`` arrays shaped like
        ``batch.token_ids``.
    """
    batch.validate_static()
    logits, values = state.apply_fn(
        {"params": state.params},
        batch.token_ids,
        batch.prompt_token_ids,
        batch.prompt_token_mask,
    )
    target_buckets = jnp.mod(batch.token_ids, logits.shape[-1])[..., None]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected_logprobs = jnp.take_along_axis(log_probs, target_buckets, axis=-1).squeeze(-1)
    probabilities = jax.nn.softmax(logits, axis=-1)
    entropy = -jnp.sum(probabilities * log_probs, axis=-1)
    return selected_logprobs, values, entropy


def external_grpo_actor_outputs(
    state: TrainState, batch: ExternalGrpoTokenBatch
) -> LlmActorTokenScores:
    """Recompute trainable actor logprobs for an external GRPO token batch.

    This is the compact trainable actor lane for externally collected vLLM
    rollout evidence. Production Qwen/Gemma/Qwix actors should implement the
    same ``LlmActorTokenScores`` contract, while this function gives us a real
    Optax-updateable baseline today.

    :param state: Prompt-conditioned token actor TrainState.
    :param batch: External GRPO token batch built from replay evidence.
    :returns: Actor-only token scores shaped like ``batch.token_ids``.
    """
    batch.validate_static()
    logits, _values = state.apply_fn(
        {"params": state.params},
        batch.token_ids,
        batch.prompt_token_ids,
        batch.prompt_token_mask,
    )
    target_buckets = jnp.mod(batch.token_ids, logits.shape[-1])[..., None]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected_logprobs = jnp.take_along_axis(log_probs, target_buckets, axis=-1).squeeze(-1)
    probabilities = jax.nn.softmax(logits, axis=-1)
    entropy = -jnp.sum(probabilities * log_probs, axis=-1)
    scores = LlmActorTokenScores(
        token_logprobs=selected_logprobs,
        entropy=entropy,
        token_mask=batch.token_mask,
    )
    scores.validate(batch.token_ids)
    return scores


def external_grpo_update(
    state: TrainState,
    batch: ExternalGrpoTokenBatch,
    *,
    clip_epsilon: float = 0.2,
    entropy_coefficient: float = 0.0,
) -> tuple[TrainState, dict[str, ScalarFloat]]:
    """Apply one critic-free GRPO update to external rollout evidence.

    :param state: Current compact token actor TrainState.
    :param batch: Tokenized external GRPO evidence with old logprobs and
        group-normalized advantages.
    :param clip_epsilon: PPO-style ratio clipping epsilon.
    :param entropy_coefficient: Entropy bonus scale.
    :returns: Updated TrainState and scalar metrics.
    """
    batch.validate_static()

    def loss_fn(params):
        candidate = state.replace(params=params)
        scores = external_grpo_actor_outputs(candidate, batch)
        return masked_token_grpo_loss(
            new_log_prob=scores.token_logprobs,
            old_log_prob=batch.old_logprobs,
            advantages=batch.advantages,
            token_mask=batch.token_mask,
            clip_epsilon=clip_epsilon,
            entropy=scores.entropy,
            entropy_coefficient=entropy_coefficient,
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), {
        **metrics,
        "loss": loss,
        "learned_tokens": jnp.sum(batch.token_mask.astype(jnp.float32)),
        "mean_sample_reward": jnp.mean(batch.sample_rewards),
    }


def token_ppo_update(
    state: TrainState,
    batch: TextTrajectoryBatch,
    *,
    gamma: float = 0.99,
    clip_epsilon: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> tuple[TrainState, dict[str, ScalarFloat]]:
    """Apply one masked token PPO update with a trainable actor and critic.

    The update recomputes ``new_logprobs`` from the actor head and values from
    the critic head. Behaviour logprobs remain the replayed ``old_logprobs``.
    Padding and fallback-only rows are excluded through ``batch.policy_mask``.

    :param state: Current token actor-critic TrainState.
    :param batch: Token trajectory batch produced from replay evidence.
    :param gamma: Discount for token reward-to-go.
    :param clip_epsilon: PPO clipping epsilon.
    :param value_coefficient: Value loss scale.
    :param entropy_coefficient: Entropy bonus scale.
    :returns: Updated TrainState and metrics dictionary.
    """
    return _token_ppo_update_with_mask(
        state,
        batch,
        learning_mask=batch.policy_mask,
        gamma=gamma,
        clip_epsilon=clip_epsilon,
        value_coefficient=value_coefficient,
        entropy_coefficient=entropy_coefficient,
        mode="policy",
    )


def full_token_ppo_update(
    state: TrainState,
    batch: TextTrajectoryBatch,
    *,
    gamma: float = 0.99,
    clip_epsilon: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> tuple[TrainState, dict[str, ScalarFloat]]:
    """Apply PPO to every generated token, including fallback-marked rows.

    This mode uses ``batch.token_mask`` rather than ``batch.policy_mask``. Padding
    remains excluded, but every real generated token contributes to actor,
    critic and entropy terms. Use it for full-token imitation/RL smoke tests
    where fallback completions are intentionally part of the training signal.

    :param state: Current token actor-critic TrainState.
    :param batch: Token trajectory batch produced from replay evidence.
    :param gamma: Discount for token reward-to-go.
    :param clip_epsilon: PPO clipping epsilon.
    :param value_coefficient: Value loss scale.
    :param entropy_coefficient: Entropy bonus scale.
    :returns: Updated TrainState and metrics dictionary.
    """
    return _token_ppo_update_with_mask(
        state,
        batch,
        learning_mask=batch.token_mask,
        gamma=gamma,
        clip_epsilon=clip_epsilon,
        value_coefficient=value_coefficient,
        entropy_coefficient=entropy_coefficient,
        mode="full-token",
    )


def _token_ppo_update_with_mask(
    state: TrainState,
    batch: TextTrajectoryBatch,
    *,
    learning_mask: TokenBatchBool,
    gamma: float,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
    mode: str,
) -> tuple[TrainState, dict[str, ScalarFloat]]:
    """Apply one token PPO update with an explicit learning mask."""
    batch.validate_static()
    if tuple(learning_mask.shape) != tuple(batch.token_ids.shape):
        raise ValueError("learning_mask must have shape [B, T]")
    returns = masked_token_returns(batch.rewards, batch.token_mask, gamma)

    def loss_fn(params):
        candidate = state.replace(params=params)
        new_logprobs, values, entropy = token_actor_critic_outputs(candidate, batch)
        old_values = jax.lax.stop_gradient(values)
        advantages = returns - old_values
        return masked_token_ppo_loss(
            new_log_prob=new_logprobs,
            old_log_prob=batch.old_logprobs,
            advantages=advantages,
            new_value=values,
            old_value=old_values,
            returns=returns,
            token_mask=learning_mask,
            clip_epsilon=clip_epsilon,
            value_coefficient=value_coefficient,
            entropy=entropy,
            entropy_coefficient=entropy_coefficient,
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), {
        **metrics,
        "loss": loss,
        "learned_tokens": jnp.sum(learning_mask.astype(jnp.float32)),
        "learning_mode": jnp.asarray(0 if mode == "policy" else 1, dtype=jnp.int32),
    }
