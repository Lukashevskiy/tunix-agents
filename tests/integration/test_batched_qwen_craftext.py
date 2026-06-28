"""Optional real Qwen batch completion followed by parallel CrafText stepping."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from tunix_craftext.env.config import load_mvp_config
from tunix_craftext.env.prompts import MegaPromptRenderer
from tunix_craftext.env.runtime import build_craftext_runtime
from tunix_craftext.models.tunix_adapter import QwenTunixBackend
from tunix_craftext.rollouts.batched import (
    collect_batched_text_decision,
    collect_batched_text_rollout,
    replays_from_batched_rollout,
)

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "artifacts" / "models" / "qwen25-05b-instruct"


@pytest.mark.integration
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="download the local Qwen snapshot")
def test_real_qwen_batch_drives_parallel_craftext_step() -> None:
    """Two Qwen completions decode into one vmap CrafText transition batch."""
    config = load_mvp_config(ROOT / "configs" / "mvp" / "qwen_craftext.yaml")
    runtime = build_craftext_runtime(config)
    reset = jax.vmap(runtime.adapter.reset)(jax.random.split(jax.random.PRNGKey(7), 2))
    result = collect_batched_text_decision(
        runtime.adapter,
        MegaPromptRenderer(config.prompt.template),
        QwenTunixBackend(SNAPSHOT, cache_size=2048, seed=config.run.seed),
        states=reset.state,
        action_masks=reset.action_mask,
        actions=runtime.actions,
        keys=jax.random.split(jax.random.PRNGKey(8), 2),
        goal="Stay alive and inspect the world.",
        max_new_tokens=8,
        invalid_action="fallback",
        fallback_action_id=runtime.actions.index_of("NOOP"),
    )

    assert len(result.prompts) == len(result.responses) == len(result.actions) == 2
    assert result.transition.reward.shape == (2,)
    assert result.transition.terminated.shape == (2,)
    assert result.fallback_used.shape == (2,)


@pytest.mark.integration
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="download the local Qwen snapshot")
def test_real_qwen_collects_two_env_two_step_rollout_and_exports_replays() -> None:
    """The parallel path retains per-environment replay evidence for learning conversion."""
    config = load_mvp_config(ROOT / "configs" / "mvp" / "qwen_craftext.yaml")
    runtime = build_craftext_runtime(config)
    rollout = collect_batched_text_rollout(
        runtime.adapter,
        MegaPromptRenderer(config.prompt.template),
        QwenTunixBackend(SNAPSHOT, cache_size=2048, seed=config.run.seed),
        actions=runtime.actions,
        batch_size=2,
        horizon=2,
        seed=config.run.seed,
        goal="Stay alive and inspect the world.",
        max_new_tokens=8,
        invalid_action="fallback",
        fallback_action_id=runtime.actions.index_of("NOOP"),
    )

    replays = replays_from_batched_rollout(
        rollout,
        config_path="configs/mvp/qwen_craftext.yaml",
        commit="integration",
        backend="tunix-single-device:Qwen",
    )

    assert len(rollout.decisions) == len(rollout.reset_after_step) == 2
    assert len(replays) == 2
    assert all(len(replay.steps) == 2 for replay in replays)
    assert all(step.token_ids for replay in replays for step in replay.steps)
