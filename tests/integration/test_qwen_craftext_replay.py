"""Optional real Qwen-to-CrafText replay smoke using an observable fallback policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from tunix_craftext.artifacts.text_trajectory import text_trajectory_from_replay
from tunix_craftext.env.config import load_mvp_config
from tunix_craftext.env.prompts import MegaPromptRenderer
from tunix_craftext.env.runtime import build_craftext_runtime
from tunix_craftext.models.tunix_adapter import QwenTunixBackend
from tunix_craftext.rollouts.text_episode import collect_text_episode

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "artifacts" / "models" / "qwen25-05b-instruct"

@pytest.mark.integration
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="download the local Qwen snapshot")
def test_local_qwen_completion_records_real_craftext_fallback_replay() -> None:
    """Vendor MegaPrompts, Qwen and real CrafText produce an inspectable replay."""
    config = load_mvp_config(ROOT / "configs" / "mvp" / "tiny_craftext.yaml")
    runtime = build_craftext_runtime(config)
    artifact = collect_text_episode(
        runtime.adapter,
        MegaPromptRenderer(config.prompt.template),
        QwenTunixBackend(SNAPSHOT, cache_size=2048, seed=config.run.seed),
        goal="Stay safe.",
        actions=runtime.actions,
        horizon=1,
        seed=config.run.seed,
        config_path="configs/env/smoke/tiny_craftext.yaml",
        commit="test",
        max_new_tokens=8,
        invalid_action="fallback",
        fallback_action_id=runtime.actions.index_of("NOOP"),
    )

    step = artifact.steps[0]
    assert artifact.backend == "tunix-single-device:Qwen/Qwen2.5-0.5B-Instruct"
    assert len(artifact.steps) == 1
    assert step.fallback_used or step.action_label in runtime.actions.labels
    assert step.token_logprobs is not None
    assert step.token_ids is not None
    assert step.prompt_token_ids is not None
    assert len(step.token_ids) == len(step.token_logprobs)
    assert step.raw_completion

    batch = text_trajectory_from_replay(artifact)
    assert batch.token_ids.shape[0] == 1
    assert bool(batch.token_mask[0].any())
    assert bool(batch.policy_mask[0].any()) is not step.fallback_used
