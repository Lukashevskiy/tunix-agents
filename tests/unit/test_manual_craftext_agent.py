"""Manual CrafText control script contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tunix_craftext.config import load_mvp_config
from tunix_craftext.replay import ReplayArtifact, ReplayStep

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


def test_manual_episode_metrics_summarizes_replay() -> None:
    artifact = ReplayArtifact(
        "configs/mvp/qwen_craftext.yaml",
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

    assert args.config == Path("configs/manual/caged_wood_achievements_energy.yaml")
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
    world_preset_path = (
        ROOT
        / "vendor"
        / "caged-craftext"
        / "caged_craftext"
        / "world_presets"
        / "caged_craftext_play.yaml"
    )

    assert config.run.name == "manual-caged-wood-achievements-energy"
    assert config.environment.implementation == "caged-craftext"
    assert config.environment.world_preset == "caged_craftext_play"
    assert config.environment.scenario_config == "budget/achievements/easy/wood_achievements"
    assert config.environment.batch_size == 1
    assert config.environment.horizon == 450
    assert scenario_path.is_file()
    assert "dataset_key: wood_achievements" in scenario_path.read_text(encoding="utf-8")
    assert world_preset_path.is_file()
    world_preset = world_preset_path.read_text(encoding="utf-8")
    assert "player_energy:" in world_preset
    assert "action_energy_drain" in world_preset
    assert "action_energy_cost:" in world_preset
