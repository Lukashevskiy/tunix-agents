from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from tunix_craftext.prompts import (
    ActionCatalog,
    MegaPromptRenderer,
    PromptContext,
    PromptContractError,
)


@dataclass
class FakeMegaPromptBackend:
    """Capture vendor-shaped metadata while avoiding a vendor dependency in unit tests."""

    received: Mapping[str, object] | None = None

    def render(self, meta_info: Mapping[str, object]) -> str:
        self.received = meta_info
        return f"Goal: {meta_info['goal']}\nActions: {', '.join(meta_info['act'])}"


def test_megaprompt_renderer_preserves_environment_goal_and_action_mapping() -> None:
    backend = FakeMegaPromptBackend()
    catalog = ActionCatalog(("LEFT", "RIGHT", "DO"))

    rendered = MegaPromptRenderer("base", backend).render(
        PromptContext(
            goal="collect wood", observation={"inventory": 0}, actions=catalog, safety="stay safe"
        )
    )

    assert rendered.template_name == "base"
    assert rendered.actions.index_of("DO") == 2
    assert "collect wood" in rendered.text
    assert backend.received == {
        "goal": "collect wood",
        "obs": {"inventory": 0},
        "act": ["LEFT", "RIGHT", "DO"],
        "dialog": [],
        "safety": "stay safe",
    }


def test_action_catalog_rejects_unknown_model_action() -> None:
    with pytest.raises(PromptContractError, match="unknown action"):
        ActionCatalog(("LEFT",)).index_of("FLY")
