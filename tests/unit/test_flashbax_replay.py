"""Flashbax-backed text replay tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tunix_craftext.flashbax_replay import make_text_replay_buffer
from tunix_craftext.text_trajectory import TextTrajectoryBatch


def _batch() -> TextTrajectoryBatch:
    """Build two valid, fixed-shape text decisions for replay."""
    return TextTrajectoryBatch(
        token_ids=jnp.asarray([[11, 12], [21, 0]], dtype=jnp.int32),
        prompt_token_ids=jnp.asarray([[101, 102], [201, 0]], dtype=jnp.int32),
        prompt_token_mask=jnp.asarray([[True, True], [True, False]]),
        old_logprobs=jnp.asarray([[-1.0, -2.0], [-3.0, 0.0]], dtype=jnp.float32),
        token_mask=jnp.asarray([[True, True], [True, False]]),
        policy_mask=jnp.asarray([[True, True], [True, False]]),
        rewards=jnp.asarray([[0.0, 1.0], [2.0, 0.0]], dtype=jnp.float32),
        action_ids=jnp.asarray([1, 2], dtype=jnp.int32),
        terminated=jnp.asarray([False, True]),
        fallback_used=jnp.asarray([False, False]),
        invalid_action=jnp.asarray([False, False]),
    )


def test_flashbax_replay_adds_and_samples_text_decisions_under_jit() -> None:
    """The sync transport can add and sample typed text decisions inside JAX."""
    batch = _batch()
    replay = make_text_replay_buffer(
        template=batch, capacity=8, min_size=2, sample_batch_size=2
    )
    state = replay.initialize()

    state = jax.jit(replay.add)(state, batch)
    assert bool(replay.can_sample(state))

    sampled = jax.jit(replay.sample)(state, jax.random.key(0))
    sampled.validate()
    assert sampled.token_ids.shape == (2, 2)
    assert np.all(np.isin(np.asarray(sampled.action_ids), np.asarray([1, 2])))
