"""Smoke test for the vendored MegaPrompts package; skipped without its opt-in extra."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from tunix_craftext.prompts import ActionCatalog, MegaPromptRenderer, PromptContext


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
