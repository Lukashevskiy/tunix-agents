"""JAX-LM-inspired static workload preflight tests."""

from pathlib import Path

import pytest

from tunix_craftext.preflight import QwenTensorShape, validate_agentic_grpo_preflight
from tunix_craftext.rlcluster_workload import AgenticGrpoWorkloadSpec, RLClusterWorkloadError
from tunix_craftext.tunix_topology import TunixTopology, load_tunix_topology

ROOT = Path(__file__).resolve().parents[2]


def _spec() -> AgenticGrpoWorkloadSpec:
    return AgenticGrpoWorkloadSpec(1, 1, 2, 2, 2, 128, 8, 256, num_generations=2)


def test_agentic_preflight_accepts_pinned_shape_on_local_profile() -> None:
    topology = load_tunix_topology(ROOT / "configs/topology/qwen_agentic_grpo_local.yaml")

    validate_agentic_grpo_preflight(topology, _spec(), QwenTensorShape(896, 14, 151936))


def test_agentic_preflight_rejects_non_divisible_tensor_parallel_degree() -> None:
    topology = TunixTopology(
        "bad", "data", {"actor": (0, 1, 2), "rollout": (0,), "reference": (0,)}
    )

    with pytest.raises(RLClusterWorkloadError, match="num_heads"):
        validate_agentic_grpo_preflight(topology, _spec(), QwenTensorShape(896, 14, 151936))
