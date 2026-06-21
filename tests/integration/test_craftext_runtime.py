"""Real vendor smoke: canonical config builds and resets the CrafText runtime."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tunix_craftext.config import load_mvp_config
from tunix_craftext.runtime import build_craftext_runtime
from tunix_craftext.rollout import collect_rollout
from tunix_craftext.rollout import collect_rollout_scan_indexed


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_canonical_config_builds_real_craftext_and_resets_through_adapter() -> None:
    config = load_mvp_config(ROOT / "configs" / "mvp" / "tiny_craftext.yaml")
    runtime = build_craftext_runtime(config)

    reset = runtime.adapter.reset(jax.random.PRNGKey(config.run.seed))

    assert runtime.action_count == 17
    assert reset.action_mask.shape == (runtime.action_count,)


@pytest.mark.integration
def test_real_craftext_fixed_keys_and_actions_produce_deterministic_mini_trajectory() -> None:
    config = load_mvp_config(ROOT / "configs" / "mvp" / "tiny_craftext.yaml")

    def collect_rewards() -> tuple[list[float], list[bool]]:
        runtime = build_craftext_runtime(config)
        reset = runtime.adapter.reset(jax.random.PRNGKey(config.run.seed))
        state = reset.state
        rewards: list[float] = []
        terminals: list[bool] = []
        for offset, action in enumerate((0, 1, 0, 1)):
            transition = runtime.adapter.step(jax.random.PRNGKey(config.run.seed + offset + 1), state, action)
            rewards.append(float(transition.reward))
            terminals.append(bool(transition.terminated))
            state = transition.state
        return rewards, terminals

    first_rewards, first_terminals = collect_rewards()
    second_rewards, second_terminals = collect_rewards()

    np.testing.assert_array_equal(first_rewards, second_rewards)
    assert first_terminals == second_terminals
    assert len(first_rewards) == 4


@pytest.mark.integration
def test_real_craftext_collects_deterministic_batched_eight_step_trajectory() -> None:
    config = load_mvp_config(ROOT / "configs" / "mvp" / "tiny_craftext.yaml")
    batch_size, horizon = config.environment.batch_size, config.environment.horizon

    def collect() -> tuple[jax.Array, jax.Array, jax.Array]:
        runtime = build_craftext_runtime(config)
        reset = jax.vmap(runtime.adapter.reset)(
            jax.random.split(jax.random.PRNGKey(config.run.seed), batch_size)
        )
        step_keys = jax.random.split(jax.random.PRNGKey(101), horizon * batch_size).reshape(
            horizon, batch_size, 2
        )
        step_index = 0

        def policy(observation: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            return jnp.zeros((observation.shape[0],), dtype=jnp.int32), jnp.zeros(batch_size), jnp.zeros(batch_size)

        def step(state: object, action: jax.Array) -> tuple[object, jax.Array, jax.Array, jax.Array, jax.Array]:
            nonlocal step_index
            transition = jax.vmap(runtime.adapter.step)(step_keys[step_index], state, action)
            step_index += 1
            return transition.state, transition.observation, transition.reward, transition.terminated, transition.truncated

        _, _, rollout = collect_rollout(reset.state, reset.observation, horizon, policy, step)
        return rollout.transitions.reward, rollout.transitions.done, rollout.transitions.observation

    first_reward, first_done, first_observation = collect()
    second_reward, second_done, second_observation = collect()

    assert first_reward.shape == (horizon, batch_size)
    assert first_done.shape == (horizon, batch_size)
    assert first_observation.shape[:2] == (horizon, batch_size)
    np.testing.assert_array_equal(first_reward, second_reward)
    np.testing.assert_array_equal(first_done, second_done)


@pytest.mark.integration
def test_real_craftext_indexed_scan_matches_reference_discrete_trajectory() -> None:
    config = load_mvp_config(ROOT / "configs" / "mvp" / "tiny_craftext.yaml")
    batch_size, horizon = config.environment.batch_size, config.environment.horizon
    runtime = build_craftext_runtime(config)
    reset = jax.vmap(runtime.adapter.reset)(jax.random.split(jax.random.PRNGKey(config.run.seed), batch_size))
    step_keys = jax.random.split(jax.random.PRNGKey(101), horizon * batch_size).reshape(horizon, batch_size, 2)

    def policy(observation: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jnp.zeros((observation.shape[0],), dtype=jnp.int32), jnp.zeros(batch_size), jnp.zeros(batch_size)

    reference_index = 0

    def reference_step(state: object, action: jax.Array):
        nonlocal reference_index
        transition = jax.vmap(runtime.adapter.step)(step_keys[reference_index], state, action)
        reference_index += 1
        return transition.state, transition.observation, transition.reward, transition.terminated, transition.truncated

    def indexed_step(state: object, action: jax.Array, index: jax.Array):
        transition = jax.vmap(runtime.adapter.step)(step_keys[index], state, action)
        return transition.state, transition.observation, transition.reward, transition.terminated, transition.truncated

    _, _, reference = collect_rollout(reset.state, reset.observation, horizon, policy, reference_step)
    _, _, scanned = collect_rollout_scan_indexed(reset.state, reset.observation, horizon, policy, indexed_step)

    np.testing.assert_array_equal(scanned.transitions.reward, reference.transitions.reward)
    np.testing.assert_array_equal(scanned.transitions.done, reference.transitions.done)
