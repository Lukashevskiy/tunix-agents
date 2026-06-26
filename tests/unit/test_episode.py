"""Unit tests for host-side text episode orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pytest

from tunix_craftext.adapters import CraftaxAdapter
from tunix_craftext.episode import collect_text_episode
from tunix_craftext.llm import LlmRequest, LlmResponse, ScriptedLlmBackend
from tunix_craftext.prompts import ActionCatalog, MegaPromptRenderer


@dataclass(frozen=True)
class _State:
    timestep: int


class _Env:
    def reset(self, key: object, params: object) -> tuple[dict[str, int], _State]:
        del key, params
        return {"timestep": 0}, _State(0)

    def step(
        self,
        key: object,
        state: _State,
        action: int,
        params: object,
    ) -> tuple[dict[str, int], _State, np.ndarray, np.ndarray, Mapping[str, np.ndarray]]:
        del key, params
        next_state = _State(state.timestep + 1)
        return (
            {"timestep": next_state.timestep, "action": int(action)},
            next_state,
            np.asarray(float(action)),
            np.asarray(next_state.timestep == 2),
            {"action_mask": np.asarray([True, True, True])},
        )


class _Renderer:
    def render(self, meta_info: Mapping[str, object]) -> str:
        return f"{meta_info['goal']} :: {meta_info['obs']} :: {meta_info['dialog']}"


@dataclass
class _RecordingBackend:
    raw_text: str = "<action>DO</action>"
    request: LlmRequest | None = None

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.request = request
        return LlmResponse(
            self.raw_text,
            "recording",
            "fixture",
            token_logprobs=(-0.1, -0.2),
            token_ids=(10, 11),
            prompt_token_ids=(1, 2, 3),
        )


def _adapter() -> CraftaxAdapter[object, dict[str, int], _State]:
    return CraftaxAdapter(_Env(), params=object(), action_count=3)


def _renderer() -> MegaPromptRenderer[dict[str, int]]:
    return MegaPromptRenderer("unit", _Renderer())


def _actions() -> ActionCatalog:
    return ActionCatalog(("LEFT", "RIGHT", "DO"))


def test_collect_text_episode_records_prompt_action_tokens_and_terminal_step() -> None:
    artifact = collect_text_episode(
        _adapter(),
        _renderer(),
        _RecordingBackend(),
        goal="collect safely",
        actions=_actions(),
        horizon=4,
        seed=7,
        config_path="configs/test.yaml",
        commit="abc123",
    )

    assert artifact.backend == "recording:fixture"
    assert len(artifact.steps) == 2
    assert [step.action_id for step in artifact.steps] == [2, 2]
    assert artifact.steps[-1].terminated
    assert "collect safely" in artifact.steps[0].prompt
    assert "State" in artifact.steps[1].prompt
    assert "<action>DO</action>" in artifact.steps[1].prompt
    assert artifact.steps[0].token_logprobs == (-0.1, -0.2)
    assert artifact.steps[0].token_ids == (10, 11)
    assert artifact.steps[0].prompt_token_ids == (1, 2, 3)


def test_collect_text_episode_passes_generation_cap_to_backend() -> None:
    backend = _RecordingBackend()

    collect_text_episode(
        _adapter(),
        _renderer(),
        backend,
        goal="collect safely",
        actions=_actions(),
        horizon=1,
        seed=7,
        config_path="configs/test.yaml",
        commit="abc123",
        max_new_tokens=4,
    )

    assert backend.request is not None
    assert backend.request.max_new_tokens == 4


def test_collect_text_episode_records_invalid_completion_with_explicit_fallback() -> None:
    artifact = collect_text_episode(
        _adapter(),
        _renderer(),
        ScriptedLlmBackend("I cannot decide", model="fixture"),
        goal="collect safely",
        actions=_actions(),
        horizon=1,
        seed=7,
        config_path="configs/test.yaml",
        commit="abc123",
        invalid_action="fallback",
        fallback_action_id=0,
    )

    [step] = artifact.steps
    assert (step.action_id, step.action_label) == (0, "LEFT")
    assert step.invalid_format == 1
    assert step.unknown_action == 0
    assert step.fallback_used


def test_collect_text_episode_raises_for_invalid_completion_without_fallback() -> None:
    with pytest.raises(ValueError, match="action"):
        collect_text_episode(
            _adapter(),
            _renderer(),
            ScriptedLlmBackend("I cannot decide"),
            goal="collect safely",
            actions=_actions(),
            horizon=1,
            seed=7,
            config_path="configs/test.yaml",
            commit="abc123",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"horizon": 0}, "horizon"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"actions": ActionCatalog(("LEFT",))}, "action catalog"),
        ({"invalid_action": "coerce"}, "invalid_action"),
        ({"invalid_action": "fallback"}, "fallback action"),
        ({"invalid_action": "fallback", "fallback_action_id": 99}, "fallback action"),
    ],
)
def test_collect_text_episode_rejects_invalid_contract(
    kwargs: dict[str, object], message: str
) -> None:
    params = {
        "adapter": _adapter(),
        "renderer": _renderer(),
        "backend": ScriptedLlmBackend("<action>DO</action>"),
        "goal": "collect safely",
        "actions": _actions(),
        "horizon": 1,
        "seed": 7,
        "config_path": "configs/test.yaml",
        "commit": "abc123",
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        collect_text_episode(**params)  # type: ignore[arg-type]
