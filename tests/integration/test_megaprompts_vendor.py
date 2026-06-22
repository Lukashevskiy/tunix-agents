"""Smoke test for the vendored MegaPrompts package; skipped without its opt-in extra."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from tunix_craftext.config import load_mvp_config
from tunix_craftext.prompts import ActionCatalog, MegaPromptRenderer, PromptContext
from tunix_craftext.runtime import build_craftext_runtime

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
@pytest.mark.skipif(
    importlib.util.find_spec("megaprompt") is None, reason="install the prompts extra"
)
def test_vendored_megaprompt_renders_environment_shaped_observation() -> None:
    observation = SimpleNamespace(
        map=np.zeros((3, 3), dtype=np.int32),
        player_position=np.array([1, 1]),
        player_direction=np.array(1),
        inventory={"wood": 2},
    )

    rendered = MegaPromptRenderer("base").render(
        PromptContext(
            goal="collect wood",
            observation=observation,
            actions=ActionCatalog(("LEFT", "RIGHT", "DO")),
        )
    )

    assert "collect wood" in rendered.text
    assert "<action>" in rendered.text
    assert rendered.actions.index_of("DO") == 2


@pytest.mark.integration
def test_vendored_megaprompt_renders_real_craftext_structured_state() -> None:
    """Production CrafText prompt uses EnvState rather than the pixel observation array."""
    config = load_mvp_config(ROOT / "configs" / "mvp" / "tiny_craftext.yaml")
    runtime = build_craftext_runtime(config)
    reset = runtime.adapter.reset(jax.random.PRNGKey(config.run.seed))

    rendered = MegaPromptRenderer(config.prompt.template).render(
        PromptContext("stay alive", reset.state, runtime.actions)
    )

    assert "You are at coord" in rendered.text
    assert "<action>" in rendered.text
