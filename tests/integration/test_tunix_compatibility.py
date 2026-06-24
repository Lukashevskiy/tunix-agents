"""Compatibility smoke for the exact public Tunix boundary we pin in Git."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import pytest
import yaml

pytest.importorskip("tunix", reason="install the tunix extra")
from tunix.rl.agentic.agents.tool_agent import ToolAgent
from tunix.rl.agentic.environments.base_environment import BaseTaskEnv
from tunix.rl.ppo.ppo_learner import PPOConfig, PPOLearner
from tunix.rl.rl_cluster import Role

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_pinned_tunix_exposes_declared_public_ppo_api() -> None:
    """The source record and imported public PPO classes stay in agreement."""
    record = yaml.safe_load((ROOT / "compatibility" / "tunix.yaml").read_text(encoding="utf-8"))

    assert record["distribution"] == "google-tunix"
    assert record["import_package"] == "tunix"
    assert record["source"]["revision"].startswith("608733cc")
    assert version("google-tunix") == record["source"]["snapshot_version"]
    assert PPOConfig.__name__ == "PPOConfig"
    assert PPOLearner.__name__ == "PPOLearner"
    assert record["public_api_boundary"]["role"] == "tunix.rl.rl_cluster.Role"
    assert record["public_api_boundary"]["tool_agent"] == (
        "tunix.rl.agentic.agents.tool_agent.ToolAgent"
    )
    assert record["public_api_boundary"]["task_environment"] == (
        "tunix.rl.agentic.environments.base_environment.BaseTaskEnv"
    )
    assert Role.ACTOR.value == "actor"
    assert ToolAgent.__name__ == "ToolAgent"
    assert BaseTaskEnv.__name__ == "BaseTaskEnv"
