"""RLCluster configuration contract tests without accelerator model loading."""

from pathlib import Path

import pytest

from tunix_craftext.rlcluster_workload import (
    RLClusterWorkloadError,
    RLClusterWorkloadSpec,
    build_rlcluster_config,
)
from tunix_craftext.tunix_topology import load_tunix_topology

ROOT = Path(__file__).resolve().parents[2]


def test_rlcluster_config_binds_all_roles_and_static_batch_knobs() -> None:
    spec = RLClusterWorkloadSpec(10, 5, 4, 2, 2, 128, 8, 256)
    config = build_rlcluster_config(
        load_tunix_topology(ROOT / "configs/topology/qwen_local_smoke.yaml"), spec
    )

    assert {role.value for role in config.role_to_mesh} == {
        "actor", "rollout", "critic", "reference"
    }
    assert config.training_config.mini_batch_size == 4
    assert config.rollout_config.return_logprobs is True


def test_rlcluster_workload_rejects_invalid_static_contract() -> None:
    with pytest.raises(RLClusterWorkloadError, match="kv_cache_size"):
        RLClusterWorkloadSpec(10, 5, 4, 2, 2, 128, 8, 128)
