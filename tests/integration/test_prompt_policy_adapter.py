"""Vertical smoke: environment-shaped state → prompt → text policy → adapter action."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pytest

from tunix_craftext.adapters import CrafTextAdapter
from tunix_craftext.env.prompts import ActionCatalog, MegaPromptRenderer, PromptContext
from tunix_craftext.env.text_policy import TextPolicy, act


@dataclass(frozen=True)
class TinyState:
    timestep: int


class TinyEnvironment:
    """Small vendor-shaped environment used only to prove the vertical boundary."""

    def reset(self, key: int, params: object) -> tuple[dict[str, int], TinyState]:
        return {"seed": key}, TinyState(0)

    def step(
        self, key: int, state: TinyState, action: int, params: object
    ) -> tuple[dict[str, int], TinyState, np.ndarray, np.ndarray, Mapping[str, np.ndarray]]:
        next_state = TinyState(state.timestep + 1)
        return (
            {"timestep": next_state.timestep, "action": action},
            next_state,
            np.asarray(float(action)),
            np.asarray(False),
            {"action_mask": np.asarray([True, True, True])},
        )


class FakePromptBackend:
    def render(self, meta_info: Mapping[str, object]) -> str:
        return f"goal={meta_info['goal']}; actions={meta_info['act']}"


class FixedTextPolicy(TextPolicy):
    def generate(self, prompt: object) -> str:
        return "<action>DO</action>"


@pytest.mark.integration
def test_prompt_policy_decoded_action_steps_environment_adapter() -> None:
    adapter = CrafTextAdapter(TinyEnvironment(), params=object(), action_count=3)
    reset = adapter.reset(key=7)
    rendered = MegaPromptRenderer("fake", FakePromptBackend()).render(
        PromptContext("gather resource", reset.observation, ActionCatalog(("LEFT", "RIGHT", "DO")))
    )

    decision, metrics = act(FixedTextPolicy(), rendered)
    transition = adapter.step(key=8, state=reset.state, action=decision.action_id)

    assert decision.action_id == 2
    assert metrics.invalid_format == metrics.unknown_action == 0
    np.testing.assert_array_equal(transition.reward, 2.0)
    assert transition.observation["action"] == 2
