"""Real vendor smoke: canonical config builds and resets the CrafText runtime."""

from pathlib import Path

import jax
import numpy as np
import pytest

from tunix_craftext.config import load_mvp_config
from tunix_craftext.runtime import build_craftext_runtime


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
