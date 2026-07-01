"""Server-readiness evidence checks before accelerator experiments."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tunix_craftext.artifacts.observability import read_jsonl
from tunix_craftext.training.server_readiness import check_server_readiness

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/grpo/qwen_agentic_local.yaml"


def test_server_readiness_writes_observability_evidence(tmp_path: Path) -> None:
    report = check_server_readiness(PROFILE, run_dir=tmp_path / "run")

    assert report.schema == "tunix-craftext.server-readiness/v1"
    assert report.run_id == "qwen-agentic-craftext-local-smoke"
    assert {check.name for check in report.checks} >= {
        "profile",
        "tunix_preflight",
        "jax_devices",
        "model_snapshot",
        "validation_evidence",
        "checkpoint_evidence",
        "observability",
    }
    assert Path(report.evidence["provenance"]).is_file()
    assert Path(report.evidence["validation_artifact"]).is_file()
    assert Path(report.evidence["checkpoint_probe"]).is_dir()

    [metric] = read_jsonl(Path(report.evidence["metrics"]))
    [validation] = read_jsonl(Path(report.evidence["validation_trajectories"]))
    artifacts = read_jsonl(Path(report.evidence["artifacts"]))

    assert metric["phase"] == "server_readiness"
    assert metric["trajectory_path"] == report.evidence["validation_artifact"]
    assert validation["schema"] == "tunix-craftext.validation-trajectory/v1"
    assert validation["task_id"] == "server-readiness-evidence"
    assert {artifact["kind"] for artifact in artifacts} >= {
        "validation_trajectory",
        "checkpoint",
        "config",
        "profile",
    }


def test_server_readiness_resolves_profile_paths_outside_repo_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Notebook cwd must not break topology/generation/vendor paths from profiles."""
    monkeypatch.chdir(tmp_path)

    report = check_server_readiness(PROFILE, run_dir=tmp_path / "run")

    assert Path(report.evidence["provenance"]).is_file()
    manifest = json.loads(Path(report.evidence["provenance"]).read_text(encoding="utf-8"))
    assert manifest["inputs"]["generation_config"] == "configs/generation/qwen_vllm_sync.yaml"
    assert manifest["generation"]["engine"]["backend"] == "vllm-offload"
    assert manifest["inputs"]["vendor_manifest_sha256"] != "missing"


def test_server_readiness_can_require_model_snapshot(tmp_path: Path) -> None:
    report = check_server_readiness(
        _profile_with_missing_snapshot(tmp_path),
        run_dir=tmp_path / "run",
        require_snapshot=True,
    )

    snapshot_check = next(check for check in report.checks if check.name == "model_snapshot")
    assert snapshot_check.status == "fail"
    assert not report.ok


def test_readiness_script_returns_non_zero_for_failed_required_check(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "check_server_readiness", ROOT / "scripts/check_server_readiness.py"
    )
    script = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(script)
    output = tmp_path / "report.json"

    exit_code = script.main(
        [
                "--profile",
                str(_profile_with_missing_snapshot(tmp_path)),
                "--run-dir",
                str(tmp_path / "run"),
                "--require-snapshot",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["evidence"]["metrics"].endswith("metrics.jsonl")


def _profile_with_missing_snapshot(tmp_path: Path) -> Path:
    profile = tmp_path / "profile-missing-snapshot.yaml"
    run_dir = tmp_path / "profile-run"
    profile.write_text(
        f"""
schema_version: 1
run:
  name: readiness-missing-snapshot
  seed: 7
  goal: Use craftext_step.
environment_config: {ROOT / "configs/mvp/qwen_craftext.yaml"}
topology_config: {ROOT / "configs/topology/qwen_agentic_grpo_local.yaml"}
generation_config: {ROOT / "configs/generation/qwen_vllm_sync.yaml"}
model:
  model_id: Qwen/Qwen2.5-0.5B-Instruct
  snapshot: {tmp_path / "definitely-missing-model"}
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
  root: {run_dir}
  trajectories: {run_dir / "trajectories.jsonl"}
  metrics: {run_dir / "metrics.jsonl"}
  checkpoints: {run_dir / "checkpoints"}
  provenance: {run_dir / "provenance.json"}
vendor_manifest: {ROOT / "vendor/manifest.json"}
""",
        encoding="utf-8",
    )
    return profile
