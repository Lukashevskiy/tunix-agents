"""End-to-end host boundary: prompt, LLM completion, environment and replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pytest

from tunix_craftext.adapters import CrafTextAdapter
from tunix_craftext.env.prompts import ActionCatalog, MegaPromptRenderer
from tunix_craftext.models.llm import LlmRequest, LlmResponse, ScriptedLlmBackend
from tunix_craftext.rollouts.text_episode import collect_text_episode


@dataclass(frozen=True)
class State:
    """Minimal immutable vendor state."""

    timestep: int


class Environment:
    """Minimal vendor-shaped terminal environment for orchestration testing."""

    def reset(self, key: object, params: object) -> tuple[dict[str, int], State]:
        return {"timestep": 0}, State(0)

    def step(
        self, key: object, state: State, action: int, params: object
    ) -> tuple[dict[str, int], State, np.ndarray, np.ndarray, Mapping[str, np.ndarray]]:
        next_state = State(state.timestep + 1)
        return (
            {"timestep": next_state.timestep, "action": action},
            next_state,
            np.asarray(float(action)),
            np.asarray(next_state.timestep == 2),
            {"action_mask": np.asarray([True, True, True])},
        )


class Renderer:
    """Tiny deterministic MegaPrompt-compatible renderer injection."""

    def render(self, meta_info: Mapping[str, object]) -> str:
        return f"{meta_info['goal']} :: {meta_info['obs']} :: {meta_info['dialog']}"


@dataclass
class RecordingBackend:
    """Test backend that exposes the request passed through episode orchestration."""

    request: LlmRequest | None = None

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.request = request
        return LlmResponse("<action>DO</action>", "recording", "fixture")


@pytest.mark.integration
def test_text_episode_preserves_prompt_completion_and_real_action_replay() -> None:
    """Two model decisions reach the terminal environment and retain provenance."""
    artifact = collect_text_episode(
        CrafTextAdapter(Environment(), params=object(), action_count=3),
        MegaPromptRenderer("test", Renderer()),
        ScriptedLlmBackend("reasoning\n<action>DO</action>", model="fixture"),
        goal="collect safely",
        actions=ActionCatalog(("LEFT", "RIGHT", "DO")),
        horizon=4,
        seed=7,
        config_path="configs/test.yaml",
        commit="abc123",
    )

    assert artifact.backend == "scripted:fixture"
    assert len(artifact.steps) == 2
    assert artifact.steps[-1].terminated
    assert [step.action_id for step in artifact.steps] == [2, 2]
    assert artifact.steps[1].raw_completion.startswith("reasoning")
    assert "reasoning" in artifact.steps[1].prompt
    assert "State(timestep=0)" in artifact.steps[0].prompt


@pytest.mark.integration
def test_text_episode_passes_generation_cap_to_backend() -> None:
    """Short real-model smokes can cap decoding without changing replay semantics."""
    backend = RecordingBackend()
    collect_text_episode(
        CrafTextAdapter(Environment(), params=object(), action_count=3),
        MegaPromptRenderer("test", Renderer()),
        backend,
        goal="collect safely",
        actions=ActionCatalog(("LEFT", "RIGHT", "DO")),
        horizon=1,
        seed=7,
        config_path="configs/test.yaml",
        commit="abc123",
        max_new_tokens=4,
    )

    assert backend.request is not None
    assert backend.request.max_new_tokens == 4


@pytest.mark.integration
def test_text_episode_records_invalid_completion_and_explicit_fallback() -> None:
    """Fallback is visible in replay and never silently coerces a model completion."""
    artifact = collect_text_episode(
        CrafTextAdapter(Environment(), params=object(), action_count=3),
        MegaPromptRenderer("test", Renderer()),
        ScriptedLlmBackend("I cannot decide", model="fixture"),
        goal="collect safely",
        actions=ActionCatalog(("LEFT", "RIGHT", "DO")),
        horizon=1,
        seed=7,
        config_path="configs/test.yaml",
        commit="abc123",
        invalid_action="fallback",
        fallback_action_id=0,
    )

    step = artifact.steps[0]
    assert (step.action_id, step.action_label) == (0, "LEFT")
    assert step.invalid_format == 1
    assert step.unknown_action == 0
    assert step.fallback_used


@pytest.mark.integration
def test_text_episode_rejects_fallback_without_declared_action() -> None:
    """A fallback policy must name an in-range action rather than use a hidden default."""
    with pytest.raises(ValueError, match="fallback action"):
        collect_text_episode(
            CrafTextAdapter(Environment(), params=object(), action_count=3),
            MegaPromptRenderer("test", Renderer()),
            ScriptedLlmBackend("I cannot decide"),
            goal="collect safely",
            actions=ActionCatalog(("LEFT", "RIGHT", "DO")),
            horizon=1,
            seed=7,
            config_path="configs/test.yaml",
            commit="abc123",
            invalid_action="fallback",
        )
