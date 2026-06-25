"""Contracts for reproducible Agentic GRPO profiles and evidence manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tunix_craftext.grpo_profile import (
    GrpoProfileError,
    build_grpo_evidence_manifest,
    load_agentic_grpo_profile,
    profile_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/grpo/qwen_agentic_local.yaml"


def test_agentic_grpo_profile_loads_static_workload_and_paths() -> None:
    profile = load_agentic_grpo_profile(PROFILE)

    assert profile.run.name == "qwen-agentic-craftext-local-smoke"
    assert profile.environment_config == Path("configs/mvp/qwen_craftext.yaml")
    assert profile.topology_config == Path("configs/topology/qwen_agentic_grpo_local.yaml")
    assert profile.model.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert profile.workload.num_generations == 2
    assert profile.workload.kv_cache_size == 2048
    assert profile.evidence.provenance.name == "provenance.json"


def test_agentic_grpo_profile_rejects_unknown_keys(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        PROFILE.read_text(encoding="utf-8") + "\nextra: nope\n",
        encoding="utf-8",
    )

    with pytest.raises(GrpoProfileError, match="root keys"):
        load_agentic_grpo_profile(invalid)


def test_grpo_evidence_manifest_records_config_hashes_and_dependency_versions() -> None:
    profile = load_agentic_grpo_profile(PROFILE)

    manifest = build_grpo_evidence_manifest(
        profile,
        profile_path=PROFILE,
        repo_root=ROOT,
        packages=("jax", "definitely-not-installed-for-profile-test"),
    )

    assert manifest["schema_version"] == "tunix-craftext.agentic-grpo-evidence/v1"
    assert manifest["profile"]["sha256"] == profile_sha256(PROFILE)
    assert manifest["inputs"]["vendor_manifest_sha256"] != "missing"
    assert manifest["packages"]["jax"] != "not-installed"
    assert manifest["packages"]["definitely-not-installed-for-profile-test"] == "not-installed"


def test_runner_profile_writes_provenance_before_model_allocation(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_agentic_grpo", ROOT / "scripts/run_agentic_grpo.py"
    )
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)
    evidence_root = tmp_path / "run"
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        f"""
schema_version: 1
run:
  name: temp-profile-smoke
  seed: 3
  goal: Use craftext_step.
environment_config: {ROOT / "configs/mvp/qwen_craftext.yaml"}
topology_config: {ROOT / "configs/topology/qwen_agentic_grpo_local.yaml"}
model:
  model_id: Qwen/Qwen2.5-0.5B-Instruct
  snapshot: {tmp_path / "missing-model"}
  revision: local-test
  license: apache-2.0
workload:
  max_steps: 1
  eval_every_n_steps: 1
  mini_batch_size: 2
  train_micro_batch_size: 2
  rollout_micro_batch_size: 1
  max_prompt_length: 32
  max_new_tokens: 8
  kv_cache_size: 64
  learning_rate: 0.00001
  num_generations: 2
  max_concurrency: 2
evidence:
  root: {evidence_root}
  trajectories: {evidence_root / "trajectories.jsonl"}
  metrics: {evidence_root / "metrics.jsonl"}
  checkpoints: {evidence_root / "checkpoints"}
  provenance: {evidence_root / "provenance.json"}
vendor_manifest: {ROOT / "vendor/manifest.json"}
""",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        runner.main(["--profile", str(profile)])

    manifest = json.loads((evidence_root / "provenance.json").read_text(encoding="utf-8"))
    assert manifest["profile"]["name"] == "temp-profile-smoke"
    assert manifest["model"]["snapshot_exists"] is False
