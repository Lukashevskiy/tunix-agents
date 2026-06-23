"""Optional real Qwen batch completion followed by parallel CrafText stepping."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from tunix_craftext.batched_rollout import collect_batched_text_decision
from tunix_craftext.config import load_mvp_config
from tunix_craftext.prompts import MegaPromptRenderer
from tunix_craftext.runtime import build_craftext_runtime
from tunix_craftext.tunix_adapter import QwenTunixBackend

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
