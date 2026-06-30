"""Unit tests for the Tunix Agentic RL CrafText bridge."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pytest

pytest.importorskip("tunix", reason="install the tunix extra")

from tunix.rl.agentic.agents.agent_types import Action
from tunix.rl.agentic.trajectory.trajectory_collect_engine import TrajectoryCollectEngine
from tunix.rl.rollout.base_rollout import RolloutOutput

from tunix_craftext.env.agentic_craftext import (
    CrafTextAgenticEnvironment,
    agentic_environment_kwargs,
    agentic_task,
    build_craftext_agentic_environment,
    build_craftext_tool_agent,
)
from tunix_craftext.env.prompts import ActionCatalog, PromptContext, RenderedPrompt
from tunix_craftext.training.agentic_grpo_smoke import (
    collect_scripted_grpo_group_sync,
    grouped_advantages,
)

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Transition:
    state: int
    reward: float
    terminated: bool
    truncated: bool = False
    action_mask: tuple[bool, bool] = (True, True)


@dataclass(frozen=True)
class _Reset:
    state: int
    action_mask: tuple[bool, bool]


class _Adapter:
    action_count = 2

    def reset(self, _key: object) -> _Reset:
        return _Reset(state=0, action_mask=(True, True))

    def step(self, _key: object, state: int, action_id: int) -> _Transition:
        next_state = state + 1
        return _Transition(
            next_state,
            float(action_id + 1),
            terminated=next_state == 2,
            action_mask=(True, False),
        )


class _InstructionAdapter(_Adapter):
    def __init__(self) -> None:
        self.last_instruction_index: int | None = None

    def reset_with_instruction(self, _key: object, instruction_index: int) -> _Reset:
        self.last_instruction_index = instruction_index
        return _Reset(state=instruction_index, action_mask=(True, True))


class _EightTurnAdapter:
    """Fixed eight-turn environment for the versioned Agentic GRPO fixture."""

    action_count = 2

    def reset(self, _key: object) -> _Reset:
        return _Reset(state=0, action_mask=(True, True))

    def step(self, _key: object, state: int, action_id: int) -> _Transition:
        assert action_id == 0
        next_state = state + 1
        return _Transition(
            state=next_state,
            reward=float(next_state),
            terminated=next_state == 8,
            action_mask=(True, False),
        )


class _Renderer:
    def render(self, context: PromptContext[int]) -> RenderedPrompt:
        return RenderedPrompt(
            f"state={context.observation}; goal={context.goal}", context.actions, "test"
        )


def _action(label: str, call_id: str = "call-1") -> Action:
    return Action(
        action=[
            {
                "id": call_id,
                "function": {"name": "craftext_step", "arguments": json.dumps({"action": label})},
            }
        ]
    )


def test_agentic_task_and_environment_kwargs_are_serializable() -> None:
    task = agentic_task(goal="collect wood", seed=7, horizon=3)

    assert task == {"goal": "collect wood", "seed": 7, "horizon": 3}
    assert agentic_environment_kwargs("configs/mvp/qwen_craftext.yaml") == {
        "config_path": "configs/mvp/qwen_craftext.yaml"
    }


def test_module_level_environment_preserves_tunix_group_metadata() -> None:
    environment = CrafTextAgenticEnvironment(
        agentic_task(goal="collect wood", seed=7, horizon=3),
        adapter=_Adapter(),
        renderer=_Renderer(),
        actions=ActionCatalog(("LEFT", "RIGHT")),
        group_id=11,
        pair_index=1,
    )

    environment.reset()

    assert environment.extra_kwargs == {"group_id": 11, "pair_index": 1}


def test_module_level_environment_uses_host_key_fallback_when_cuda_backend_is_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_prng_key(_seed: int) -> object:
        raise RuntimeError("Unable to initialize backend 'cuda': no supported devices found")

    monkeypatch.setattr(jax.random, "PRNGKey", broken_prng_key)
    environment = CrafTextAgenticEnvironment(
        agentic_task(goal="collect wood", seed=7, horizon=3),
        adapter=_Adapter(),
        renderer=_Renderer(),
        actions=ActionCatalog(("LEFT", "RIGHT")),
    )

    observation, _ = environment.reset()

    assert observation == {"question": "state=0; goal=collect wood"}


def test_module_level_environment_accepts_tunix_numpy_task_scalars() -> None:
    environment = CrafTextAgenticEnvironment(
        {"goal": np.str_("collect wood"), "seed": np.int32(7), "horizon": np.int32(3)},
        adapter=_Adapter(),
        renderer=_Renderer(),
        actions=ActionCatalog(("LEFT", "RIGHT")),
    )

    observation, _ = environment.reset()

    assert observation == {"question": "state=0; goal=collect wood"}


def test_agentic_environment_uses_task_instruction_index_for_reset() -> None:
    adapter = _InstructionAdapter()
    environment = CrafTextAgenticEnvironment(
        {
            "goal": "from craftext instruction",
            "seed": np.int32(7),
            "horizon": np.int32(3),
            "instruction_index": np.int32(1),
        },
        adapter=adapter,
        renderer=_Renderer(),
        actions=ActionCatalog(("LEFT", "RIGHT")),
    )

    observation, _ = environment.reset()

    assert adapter.last_instruction_index == 1
    assert observation == {"question": "state=1; goal=from craftext instruction"}


def test_agentic_environment_logs_reset_and_invalid_action_without_prompt_content(caplog) -> None:
    environment = build_craftext_agentic_environment(
        _Adapter(),
        _Renderer(),
        goal="collect wood",
        actions=ActionCatalog(("LEFT", "RIGHT")),
        horizon=3,
        seed=7,
    )

    with caplog.at_level(logging.INFO, logger="tunix_craftext.env.agentic_craftext"):
        environment.reset()
        environment.step(_action("FLY"))

    messages = [record.getMessage() for record in caplog.records]
    assert any("'event': 'reset'" in message for message in messages)
    assert any("'event': 'invalid_action'" in message for message in messages)
    assert all("state=0" not in message for message in messages)


def test_agentic_environment_runs_multiple_craftext_turns() -> None:
    environment = build_craftext_agentic_environment(
        _Adapter(),
        _Renderer(),
        goal="collect wood",
        actions=ActionCatalog(("LEFT", "RIGHT")),
        horizon=3,
        seed=7,
    )

    observation, _ = environment.reset()
    assert observation == {"question": "state=0; goal=collect wood"}

    observation, reward, done, info = environment.step(_action("RIGHT"))
    assert observation == {"tool_outputs": {"call-1": "state=1; goal=collect wood"}}
    assert reward == 2.0
    assert not done
    assert info == {"action_id": 1, "action_label": "RIGHT"}

    observation, reward, done, _ = environment.step(_action("LEFT", "call-2"))
    assert observation == {"tool_outputs": {"call-2": "Episode finished."}}
    assert reward == 1.0
    assert done


def test_agentic_environment_reports_invalid_tool_action_without_stepping() -> None:
    environment = build_craftext_agentic_environment(
        _Adapter(),
        _Renderer(),
        goal="collect wood",
        actions=ActionCatalog(("LEFT", "RIGHT")),
        horizon=3,
        seed=7,
    )
    environment.reset()

    observation, reward, done, info = environment.step(_action("FLY"))

    assert observation["tool_outputs"]["call-1"].startswith("Invalid action:")
    assert reward == 0.0
    assert not done
    assert "invalid_action" in info


def test_agentic_environment_rejects_an_action_masked_by_the_current_state() -> None:
    environment = build_craftext_agentic_environment(
        _Adapter(),
        _Renderer(),
        goal="collect wood",
        actions=ActionCatalog(("LEFT", "RIGHT")),
        horizon=3,
        seed=7,
    )
    environment.reset()
    environment.step(_action("LEFT"))

    observation, reward, done, info = environment.step(_action("RIGHT", "call-2"))

    assert observation["tool_outputs"]["call-2"].startswith("Invalid action:")
    assert reward == 0.0
    assert not done
    assert "unavailable" in info["invalid_action"]


def test_tool_agent_exposes_the_craftext_action_schema() -> None:
    agent = build_craftext_tool_agent("Choose a CrafText action.")

    assert agent.tool_manager.names == ["craftext_step"]
    schema = agent.tool_manager.get_json_schema()[0]
    assert schema["function"]["parameters"]["required"] == ["action"]


def test_grouped_advantages_are_normalized_inside_one_grpo_task_group() -> None:
    advantages = grouped_advantages((1.0, 3.0))

    assert advantages[0] == pytest.approx(-1.0, abs=1e-5)
    assert advantages[1] == pytest.approx(1.0, abs=1e-5)
    assert grouped_advantages((2.0, 2.0)) == (0.0, 0.0)


def test_scripted_grpo_sync_wrapper_rejects_running_notebook_loop() -> None:
    async def call_sync_wrapper_inside_loop() -> None:
        collect_scripted_grpo_group_sync(
            config_path=ROOT / "configs/mvp/qwen_craftext.yaml",
            goal="collect wood",
            seed=0,
            group_id=0,
            action_sequences=(("NOOP",), ("LEFT",)),
            horizon=1,
        )

    with pytest.raises(RuntimeError, match="await collect_scripted_grpo_group"):
        asyncio.run(call_sync_wrapper_inside_loop())


def test_tunix_trajectory_engine_collects_a_multi_turn_craftext_episode() -> None:
    environment = build_craftext_agentic_environment(
        _Adapter(),
        _Renderer(),
        goal="collect wood",
        actions=ActionCatalog(("LEFT", "RIGHT")),
        horizon=3,
        seed=7,
    )
    responses = iter(("RIGHT", "LEFT"))

    def model_call(*_args: object, **_kwargs: object) -> RolloutOutput:
        label = next(responses)
        return RolloutOutput(
            text=[f'<tool_call>{{"name":"craftext_step","arguments":{{"action":"{label}"}}}}</tool_call>'],
            logits=[],
            tokens=[np.asarray([1])],
            left_padded_prompt_tokens=np.asarray([[1]]),
            logprobs=[np.asarray([0.0])],
        )

    trajectory = asyncio.run(
        TrajectoryCollectEngine(
            build_craftext_tool_agent("Choose a CrafText action."),
            environment,
            model_call=model_call,
        ).collect(mode="Trajectory")
    )

    assert len(trajectory.steps) == 2
    assert [step.reward for step in trajectory.steps] == [2.0, 1.0]
    assert trajectory.steps[-1].done


def test_versioned_agentic_fixture_covers_two_groups_two_generations_eight_turns() -> None:
    """Lock group/task/tool semantics before a real GRPOLearner update is introduced."""
    fixture = json.loads((ROOT / "tests/fixtures/agentic_grpo_golden_v1.json").read_text())

    assert fixture["schema"] == "tunix-craftext.agentic-golden/v1"
    action_catalog = ActionCatalog(tuple(fixture["action_labels"]))
    observations: list[str] = []
    for task in fixture["tasks"]:
        for pair_index in range(fixture["generations_per_task"]):
            environment = CrafTextAgenticEnvironment(
                agentic_task(
                    goal=task["goal"], seed=task["seed"], horizon=fixture["horizon"]
                ),
                adapter=_EightTurnAdapter(),
                renderer=_Renderer(),
                actions=action_catalog,
                group_id=task["group_id"],
                pair_index=pair_index,
            )
            observation, _ = environment.reset()
            observations.append(observation["question"])
            rewards: list[float] = []
            done: list[bool] = []
            masks: list[tuple[bool, bool]] = []
            for turn, label in enumerate(fixture["tool_actions"]):
                _, reward, terminal, info = environment.step(_action(label, f"call-{turn}"))
                rewards.append(reward)
                done.append(terminal)
                assert environment._action_mask is not None
                masks.append(tuple(bool(value) for value in environment._action_mask))
                assert info == {"action_id": 0, "action_label": "LEFT"}

            assert rewards == fixture["expected_rewards"]
            assert done == fixture["expected_done"]
            assert masks == [tuple(mask) for mask in fixture["expected_next_action_masks"]]

    assert len(observations) == len(fixture["tasks"]) * fixture["generations_per_task"]
    assert "goal=collect wood" in observations[0]
    assert "goal=find water" in observations[-1]
