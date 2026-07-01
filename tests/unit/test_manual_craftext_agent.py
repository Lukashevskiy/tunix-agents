"""Manual CrafText control script contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from tunix_craftext.artifacts.replay import ReplayArtifact, ReplayStep
from tunix_craftext.env.config import load_mvp_config

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "manual_craftext_agent", ROOT / "scripts" / "manual_craftext_agent.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_parse_manual_action_accepts_ids_and_labels_and_quit() -> None:
    labels = ("NOOP", "LEFT", "RIGHT")
    mask = (True, True, False)

    assert runner.parse_manual_action("1", labels=labels, action_mask=mask).label == "LEFT"
    assert runner.parse_manual_action("left", labels=labels, action_mask=mask).action_id == 1
    assert runner.parse_manual_action("q", labels=labels, action_mask=mask) is None


def test_parse_manual_action_rejects_masked_and_unknown_actions() -> None:
    labels = ("NOOP", "LEFT", "RIGHT")
    mask = (True, True, False)

    with pytest.raises(ValueError, match="masked"):
        runner.parse_manual_action("RIGHT", labels=labels, action_mask=mask)
    with pytest.raises(ValueError, match="unknown"):
        runner.parse_manual_action("JUMP", labels=labels, action_mask=mask)


def test_legal_actions_text_lists_only_unmasked_actions() -> None:
    assert runner.legal_actions_text(("NOOP", "LEFT", "RIGHT"), (True, False, True)) == (
        "0:NOOP, 2:RIGHT"
    )


def test_observation_text_summarizes_shape_dtype_range_and_preview() -> None:
    text = runner.observation_text([[0, 1, 2], [3, 4, 5]], max_values=4)

    assert "Observation:" in text
    assert "shape=(2, 3)" in text
    assert "dtype=int32" in text or "dtype=int64" in text
    assert "min=0" in text
    assert "max=5" in text
    assert "preview=[0, 1, 2, 3 ...]" in text


def test_manual_state_text_renders_concrete_ascii_map_and_player_state() -> None:
    env_state = SimpleNamespace(
        map=jnp.asarray(
            [
                [1, 1, 1, 1, 1],
                [1, 2, 5, 4, 1],
                [1, 3, 2, 8, 1],
                [1, 2, 15, 10, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=jnp.int32,
        ),
        player_position=jnp.asarray([2, 2], dtype=jnp.int32),
        player_direction=jnp.asarray(3, dtype=jnp.int32),
        player_health=jnp.asarray(9, dtype=jnp.int32),
        player_food=jnp.asarray(8, dtype=jnp.int32),
        player_drink=jnp.asarray(7, dtype=jnp.int32),
        player_energy=jnp.asarray(6, dtype=jnp.int32),
        is_sleeping=jnp.asarray(False),
        timestep=jnp.asarray(4, dtype=jnp.int32),
        inventory=SimpleNamespace(
            wood=jnp.asarray(2, dtype=jnp.int32),
            stone=jnp.asarray(0, dtype=jnp.int32),
            coal=jnp.asarray(1, dtype=jnp.int32),
        ),
        zombies=SimpleNamespace(
            position=jnp.asarray([[1, 2]], dtype=jnp.int32),
            mask=jnp.asarray([True]),
        ),
    )

    text = runner.manual_state_text(env_state, radius=1)

    assert "Vitals: pos=(2, 2), facing=UP, hp=9, food=8, drink=7, energy=6" in text
    assert "Inventory: wood=2, coal=1" in text
    assert "Map view:" in text
    assert ".ZS" in text
    assert "~@C" in text
    assert ".pD" in text


def test_manual_episode_metrics_summarizes_replay() -> None:
    artifact = ReplayArtifact(
        "configs/env/text/qwen_craftext.yaml",
        "abc",
        "manual-human",
        (
            ReplayStep(0, "p0", "<manual_action>NOOP</manual_action>", 0, "NOOP", 1.0, False),
            ReplayStep(1, "p1", "<manual_action>LEFT</manual_action>", 1, "LEFT", -0.5, True),
        ),
        schema="tunix-craftext.manual-replay/v1",
    )

    metrics = runner.manual_episode_metrics(artifact)

    assert metrics["steps"] == 2
    assert metrics["reward_sum"] == 0.5
    assert metrics["terminated"] is True
    assert metrics["manual_actions"] == ["NOOP", "LEFT"]


def test_parse_args_exposes_manual_artifact_paths() -> None:
    args = runner.parse_args(["--horizon", "3", "--seed", "9", "--show-full-prompt"])

    assert args.config == Path("configs/env/manual/caged_wood_achievements_energy.yaml")
    assert args.horizon == 3
    assert args.seed == 9
    assert args.show_full_prompt is True
    assert args.replay_output == Path("artifacts/trajectories/manual-craftext-latest.json")


def test_default_manual_config_targets_full_caged_wood_energy_scenario() -> None:
    config = load_mvp_config(ROOT / runner.DEFAULT_CONFIG)
    scenario_path = (
        ROOT
        / "vendor"
        / "caged-craftext"
        / "caged_craftext"
        / "dataset"
        / "configs"
        / "budget"
        / "achievements"
        / "easy"
        / "wood_achievements.yaml"
    )
    assert config.run.name == "manual-caged-wood-achievements-energy"
    assert config.environment.implementation == "caged-craftext"
    assert config.environment.scenario_config == "budget/achievements/easy/wood_achievements"
    assert config.environment.batch_size == 1
    assert scenario_path.is_file()
    assert "dataset_key: wood_achievements" in scenario_path.read_text(encoding="utf-8")
    world_preset_path = (
        ROOT
        / "vendor"
        / "caged-craftext"
        / "caged_craftext"
        / "world_presets"
        / f"{config.environment.world_preset}.yaml"
    )
    assert world_preset_path.is_file()
