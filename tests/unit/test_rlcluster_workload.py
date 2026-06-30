"""RLCluster configuration contract tests without accelerator model loading."""

from pathlib import Path

import jax.numpy as jnp
import pytest

import tunix_craftext.tunix.rlcluster_workload as package_workload
import tunix_craftext.tunix.rlcluster_workload as workload
from tunix_craftext.inference import TunixGenerationContract
from tunix_craftext.tunix.rlcluster_workload import (
    AgenticGrpoWorkloadSpec,
    PpoModelAssets,
    RLClusterWorkloadError,
    RLClusterWorkloadSpec,
    build_agentic_grpo_cluster,
    build_agentic_grpo_cluster_config,
    build_ppo_cluster,
    build_rlcluster_config,
    create_value_critic_from_actor,
    load_agentic_grpo_qwen_assets,
    load_ppo_gemma_assets,
    load_ppo_qwen_assets,
)
from tunix_craftext.tunix.topology import load_tunix_topology

ROOT = Path(__file__).resolve().parents[2]


def test_tunix_workload_package_preserves_legacy_shim_identity() -> None:
    assert package_workload.RLClusterWorkloadSpec is RLClusterWorkloadSpec
    assert package_workload.AgenticGrpoWorkloadSpec is AgenticGrpoWorkloadSpec
    assert package_workload.RLClusterWorkloadError is RLClusterWorkloadError
    assert package_workload.build_rlcluster_config is build_rlcluster_config


def test_rlcluster_config_binds_all_roles_and_static_batch_knobs() -> None:
    spec = RLClusterWorkloadSpec(
        10,
        5,
        4,
        2,
        2,
        128,
        8,
        256,
        checkpoint_root_directory=ROOT / "artifacts/runs/test/checkpoints",
    )
    config = build_rlcluster_config(
        load_tunix_topology(ROOT / "configs/topology/qwen_local_smoke.yaml"), spec
    )

    assert {role.value for role in config.role_to_mesh} == {
        "actor", "rollout", "critic", "reference"
    }
    assert config.training_config.mini_batch_size == 4
    assert config.training_config.checkpoint_root_directory.endswith("test/checkpoints")
    assert config.rollout_config.return_logprobs is True


def test_rlcluster_workload_rejects_invalid_static_contract() -> None:
    with pytest.raises(RLClusterWorkloadError, match="kv_cache_size"):
        RLClusterWorkloadSpec(10, 5, 4, 2, 2, 128, 8, 128)


def test_agentic_grpo_config_has_no_critic_and_recomputes_logprobs() -> None:
    spec = AgenticGrpoWorkloadSpec(
        10,
        5,
        4,
        2,
        1,
        128,
        8,
        256,
        checkpoint_root_directory=ROOT / "artifacts/runs/grpo/checkpoints",
        num_generations=2,
    )
    config = build_agentic_grpo_cluster_config(
        load_tunix_topology(ROOT / "configs/topology/qwen_agentic_grpo_local.yaml"), spec
    )

    assert {role.value for role in config.role_to_mesh} == {"actor", "rollout", "reference"}
    assert config.training_config.critic_optimizer is None
    assert config.training_config.compute_logps_micro_batch_size == 2
    assert config.training_config.checkpoint_root_directory.endswith("grpo/checkpoints")
    assert config.rollout_config.temperature == 1.0


def test_agentic_grpo_config_accepts_strict_vllm_generation_contract() -> None:
    spec = AgenticGrpoWorkloadSpec(10, 5, 4, 2, 1, 128, 8, 256, num_generations=2)
    generation = TunixGenerationContract(
        engine="vllm",
        max_prompt_length=128,
        max_tokens_to_generate=8,
        kv_cache_size=256,
        temperature=0.7,
        tensor_parallel_size=1,
        vllm_server_mode=True,
        vllm_async_scheduling=True,
        vllm_hbm_utilization=0.35,
        vllm_model_version="qwen2.5-0.5b",
    )

    config = build_agentic_grpo_cluster_config(
        load_tunix_topology(ROOT / "configs/topology/qwen_agentic_grpo_local.yaml"),
        spec,
        generation,
    )

    assert config.rollout_engine == "vllm"
    assert config.rollout_config.temperature == 0.7
    assert config.rollout_config.rollout_vllm_server_mode is True
    assert config.rollout_config.rollout_vllm_async_scheduling is True
    assert config.rollout_config.rollout_vllm_hbm_utilization == 0.35
    assert config.rollout_config.rollout_vllm_model_version == "qwen2.5-0.5b"


def test_agentic_grpo_rejects_unsupported_generation_count_or_critic_topology() -> None:
    with pytest.raises(RLClusterWorkloadError, match="at least two"):
        AgenticGrpoWorkloadSpec(10, 5, 4, 2, 1, 128, 8, 256, num_generations=1)
    with pytest.raises(RLClusterWorkloadError, match="must not declare a critic"):
        build_agentic_grpo_cluster_config(
            load_tunix_topology(ROOT / "configs/topology/qwen_local_smoke.yaml"),
            AgenticGrpoWorkloadSpec(10, 5, 4, 2, 1, 128, 8, 256),
        )


def test_agentic_grpo_assets_use_distinct_role_meshes_and_storage_dtypes(monkeypatch) -> None:
    calls: list[tuple[object, object]] = []

    def load_model(snapshot, mesh, *, dtype):
        calls.append((mesh, dtype))
        return {"snapshot": snapshot, "mesh": mesh, "dtype": dtype}

    monkeypatch.setattr(package_workload, "load_qwen_model_on_mesh", load_model)
    monkeypatch.setattr(
        package_workload, "load_qwen_tokenizer", lambda snapshot: {"snapshot": snapshot}
    )
    topology = load_tunix_topology(ROOT / "configs/topology/qwen_agentic_grpo_local.yaml")

    assets = load_agentic_grpo_qwen_assets(ROOT / "artifacts/models/qwen25-05b-instruct", topology)

    assert len(calls) == 2
    assert calls[0][1] == jnp.float32
    assert calls[1][1] == jnp.bfloat16
    assert assets.actor["mesh"] == assets.reference["mesh"]


def test_ppo_qwen_assets_bind_actor_critic_reference_and_tokenizer(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def load_model(snapshot, mesh, *, dtype):
        calls.append((str(snapshot), dtype))
        return {"snapshot": snapshot, "mesh": mesh, "dtype": dtype}

    monkeypatch.setattr(package_workload, "load_qwen_model_on_mesh", load_model)
    monkeypatch.setattr(
        package_workload, "load_qwen_tokenizer", lambda snapshot: {"snapshot": snapshot}
    )
    monkeypatch.setattr(
        package_workload,
        "create_value_critic_from_actor",
        lambda actor, *, seed=0: {"critic_from": actor, "seed": seed},
    )
    topology = load_tunix_topology(ROOT / "configs/topology/qwen_local_smoke.yaml")

    assets = load_ppo_qwen_assets(
        ROOT / "artifacts/models/qwen25-05b-instruct", topology, critic_seed=9
    )

    assert len(calls) == 2
    assert calls[0][1] == jnp.float32
    assert calls[1][1] == jnp.bfloat16
    assert assets.critic["critic_from"] == assets.actor
    assert assets.critic["seed"] == 9
    assert assets.tokenizer["snapshot"].name == "qwen25-05b-instruct"


def test_ppo_gemma_assets_use_gemma_loader_and_native_critic(monkeypatch) -> None:
    calls: list[jnp.dtype] = []

    def load_model(snapshot, mesh, *, dtype):
        del snapshot, mesh
        calls.append(dtype)
        return {"family": "gemma", "dtype": dtype}

    monkeypatch.setattr(package_workload, "load_gemma_model_on_mesh", load_model)
    monkeypatch.setattr(
        package_workload, "load_gemma_tokenizer", lambda snapshot: {"snapshot": snapshot}
    )
    monkeypatch.setattr(
        package_workload,
        "create_value_critic_from_actor",
        lambda actor, *, seed=0: {"critic_from": actor, "seed": seed},
    )
    topology = load_tunix_topology(ROOT / "configs/topology/qwen_local_smoke.yaml")

    assets = load_ppo_gemma_assets(ROOT / "artifacts/models/gemma3-270m-it", topology)

    assert calls == [jnp.float32, jnp.bfloat16]
    assert assets.actor["family"] == "gemma"
    assert assets.critic["critic_from"] == assets.actor
    assert assets.reference["dtype"] == jnp.bfloat16


def test_create_value_critic_rejects_unknown_actor_shape(monkeypatch) -> None:
    def fail_create_critic(actor, seed=0):
        del actor, seed
        raise AttributeError("lm_head")

    monkeypatch.setattr("tunix.rl.utils.create_critic_model", fail_create_critic)

    with pytest.raises(RLClusterWorkloadError, match="critic head boundary"):
        create_value_critic_from_actor(object())


def test_ppo_cluster_uses_public_actor_critic_reference_constructor(monkeypatch) -> None:
    constructed: dict[str, object] = {}

    class FakeCluster:
        def __init__(self, **kwargs) -> None:
            constructed.update(kwargs)

    monkeypatch.setattr("tunix.rl.rl_cluster.RLCluster", FakeCluster)
    topology = load_tunix_topology(ROOT / "configs/topology/qwen_local_smoke.yaml")
    assets = PpoModelAssets("actor", "critic", "reference", "tokenizer")

    cluster = build_ppo_cluster(
        topology, RLClusterWorkloadSpec(10, 5, 4, 2, 2, 128, 8, 256), assets
    )

    assert isinstance(cluster, FakeCluster)
    assert constructed["actor"] == "actor"
    assert constructed["critic"] == "critic"
    assert constructed["reference"] == "reference"
    assert constructed["tokenizer"] == "tokenizer"
    assert {role.value for role in constructed["cluster_config"].role_to_mesh} == {
        "actor",
        "rollout",
        "critic",
        "reference",
    }


def test_agentic_grpo_cluster_uses_the_public_three_role_constructor(monkeypatch) -> None:
    constructed: dict[str, object] = {}

    class FakeCluster:
        def __init__(self, **kwargs) -> None:
            constructed.update(kwargs)

    monkeypatch.setattr("tunix.rl.rl_cluster.RLCluster", FakeCluster)
    topology = load_tunix_topology(ROOT / "configs/topology/qwen_agentic_grpo_local.yaml")
    assets = workload.AgenticGrpoModelAssets("actor", "reference", "tokenizer")

    cluster = build_agentic_grpo_cluster(
        topology, AgenticGrpoWorkloadSpec(10, 5, 4, 2, 1, 128, 8, 256), assets
    )

    assert isinstance(cluster, FakeCluster)
    assert constructed["actor"] == "actor"
    assert constructed["reference"] == "reference"
    assert constructed["tokenizer"] == "tokenizer"
    assert {role.value for role in constructed["cluster_config"].role_to_mesh} == {
        "actor",
        "rollout",
        "reference",
    }
