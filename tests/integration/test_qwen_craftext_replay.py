"""Optional real Qwen-to-CrafText replay smoke using an observable fallback policy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from tunix_craftext.config import load_mvp_config
from tunix_craftext.episode import collect_text_episode
from tunix_craftext.prompts import MegaPromptRenderer
from tunix_craftext.runtime import build_craftext_runtime
from tunix_craftext.tunix_adapter import QwenTunixBackend

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "artifacts" / "models" / "qwen25-05b-instruct"


class ActionSmokeRenderer:
    """Small renderer that exposes the full real action catalog to Qwen."""

    def render(self, meta_info: Mapping[str, object]) -> str:
        actions = meta_info["act"]
        assert isinstance(actions, list)
        return (
            "Choose the safest action for this smoke test. "
            f"Allowed actions: {', '.join(str(action) for action in actions)}."
        )


@pytest.mark.integration
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="download the local Qwen snapshot")
def test_local_qwen_completion_records_real_craftext_fallback_replay() -> None:
    """A malformed Qwen action remains inspectable while the declared fallback steps CrafText."""
    config = load_mvp_config(ROOT / "configs" / "mvp" / "tiny_craftext.yaml")
    runtime = build_craftext_runtime(config)
    artifact = collect_text_episode(
        runtime.adapter,
        MegaPromptRenderer("qwen-craftext-smoke", ActionSmokeRenderer()),
        QwenTunixBackend(SNAPSHOT, cache_size=512, seed=config.run.seed),
        goal="Stay safe.",
        actions=runtime.actions,
        horizon=1,
        seed=config.run.seed,
        config_path="configs/mvp/tiny_craftext.yaml",
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
    assert step.raw_completion
