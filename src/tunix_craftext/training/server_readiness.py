"""Server-side readiness checks for accelerator GRPO/PPO experiments.

The checks in this module intentionally stop before a long training loop.  They
validate the static Tunix/CrafText profile, inspect JAX devices, and write the
same local evidence files that real training is expected to produce: provenance,
metrics, validation trajectory references, artifact manifests and a checkpoint
directory probe.  A heavier scripted mode can also exercise the real CrafText
agentic tool loop without allocating model weights.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import jax

from ..artifacts.observability import (
    JsonlRunLogger,
    JsonScalar,
    MetricRecord,
    RunArtifact,
    ValidationTrajectoryRecord,
    checkpoint_artifact,
    validation_trajectory_artifact,
)
from ..artifacts.replay import ReplayArtifact, ReplayStep, save_replay
from ..tunix import (
    RLClusterWorkloadError,
    load_tunix_topology,
    pinned_qwen_tensor_shape,
    validate_agentic_grpo_preflight,
)
from .grpo_profile import (
    AgenticGrpoProfile,
    build_grpo_evidence_manifest,
    load_agentic_grpo_profile,
    resolve_profile_path,
)

ReadinessMode = Literal["evidence", "scripted"]
CheckStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class ReadinessCheck:
    """One named readiness check result.

    :param name: Stable machine-readable check name.
    :param status: ``pass``, ``warn`` or ``fail``.
    :param message: Human-readable summary.
    :param details: Optional scalar details safe to include in JSON evidence.
    """

    name: str
    status: CheckStatus
    message: str
    details: dict[str, JsonScalar] | None = None

    @property
    def ok(self) -> bool:
        """Return ``True`` for non-failing checks."""
        return self.status != "fail"


@dataclass(frozen=True)
class ServerReadinessReport:
    """Full readiness result written before a target-hardware run.

    :param schema: Versioned JSON schema.
    :param mode: Readiness mode that was executed.
    :param run_id: Run identity from the GRPO profile.
    :param run_dir: Evidence directory used for the probe.
    :param backend: JAX default backend observed by the probe.
    :param devices: JAX devices visible to the current process.
    :param checks: Ordered check results.
    :param evidence: Paths written by the probe.
    """

    schema: str
    mode: ReadinessMode
    run_id: str
    run_dir: str
    backend: str
    devices: tuple[str, ...]
    checks: tuple[ReadinessCheck, ...]
    evidence: dict[str, str]

    @property
    def ok(self) -> bool:
        """Return whether every required readiness check passed."""
        return all(check.ok for check in self.checks)

    def to_json_dict(self) -> dict[str, object]:
        """Serialize the report as a stable JSON dictionary."""
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def check_server_readiness(
    profile_path: Path,
    *,
    mode: ReadinessMode = "evidence",
    run_dir: Path | None = None,
    require_accelerator: bool = False,
    require_snapshot: bool = False,
    scripted_horizon: int = 2,
) -> ServerReadinessReport:
    """Validate target-server prerequisites and write local evidence probes.

    :param profile_path: Canonical Agentic GRPO profile.
    :param mode: ``evidence`` for file/log contracts only, ``scripted`` to also
        execute the real CrafText agentic tool-call loop without model weights.
    :param run_dir: Optional evidence directory override. Defaults to
        ``profile.evidence.root``.
    :param require_accelerator: Fail when JAX resolves to CPU.
    :param require_snapshot: Fail when the configured local model snapshot is
        missing.
    :param scripted_horizon: Short horizon for scripted validation mode.
    :returns: Versioned readiness report with paths to written evidence.
    """
    if mode not in {"evidence", "scripted"}:
        raise ValueError("mode must be 'evidence' or 'scripted'")
    if scripted_horizon <= 0:
        raise ValueError("scripted_horizon must be positive")

    profile = load_agentic_grpo_profile(profile_path)
    output_dir = run_dir or resolve_profile_path(profile_path, profile.evidence.root)
    environment_config = resolve_profile_path(profile_path, profile.environment_config)
    topology_config = resolve_profile_path(profile_path, profile.topology_config)
    model_snapshot = resolve_profile_path(profile_path, profile.model.snapshot)
    logger = JsonlRunLogger(output_dir)
    checks: list[ReadinessCheck] = []

    _ensure_evidence_directories(output_dir)
    manifest_path = output_dir / "provenance.json"
    manifest = build_grpo_evidence_manifest(profile, profile_path=profile_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checks.append(ReadinessCheck("profile", "pass", "GRPO profile loaded and provenance written"))

    topology = load_tunix_topology(topology_config)
    try:
        validate_agentic_grpo_preflight(topology, profile.workload, pinned_qwen_tensor_shape())
    except RLClusterWorkloadError as error:
        checks.append(
            ReadinessCheck(
                "tunix_preflight",
                "fail",
                "Real Tunix Qwen vanilla rollout preflight failed",
                {"error": str(error)},
            )
        )
    else:
        checks.append(
            ReadinessCheck("tunix_preflight", "pass", "Tunix topology/workload preflight passed")
        )

    backend = jax.default_backend()
    devices = tuple(str(device) for device in jax.devices())
    device_status: CheckStatus = "pass"
    device_message = f"JAX backend is {backend} with {len(devices)} visible device(s)"
    if require_accelerator and backend == "cpu":
        device_status = "fail"
        device_message = "JAX resolved to CPU while --require-accelerator was requested"
    elif backend == "cpu":
        device_status = "warn"
    checks.append(
        ReadinessCheck(
            "jax_devices",
            device_status,
            device_message,
            {"backend": backend, "device_count": len(devices)},
        )
    )

    snapshot_status: CheckStatus = "pass"
    snapshot_message = f"model snapshot exists: {model_snapshot}"
    if not model_snapshot.is_dir():
        snapshot_status = "fail" if require_snapshot else "warn"
        snapshot_message = f"model snapshot is missing: {model_snapshot}"
    checks.append(
        ReadinessCheck(
            "model_snapshot",
            snapshot_status,
            snapshot_message,
            {"snapshot": str(model_snapshot), "exists": model_snapshot.is_dir()},
        )
    )

    validation_path, return_sum, episode_length = _write_validation_probe(
        profile,
        output_dir=output_dir,
        environment_config=environment_config,
        mode=mode,
        scripted_horizon=scripted_horizon,
    )
    validation_record = ValidationTrajectoryRecord(
        run_id=profile.run.name,
        step=0,
        task_id=f"server-readiness-{mode}",
        trajectory_path=str(validation_path),
        return_sum=return_sum,
        episode_length=episode_length,
        success=True,
        policy_version=0,
        metrics={"mode": mode, "backend": backend},
    )
    logger.log_validation_trajectory(validation_record)
    logger.log_artifact(
        validation_trajectory_artifact(
            profile.run.name,
            validation_path,
            step=0,
            task_id=f"server-readiness-{mode}",
            policy_version=0,
        )
    )
    checks.append(
        ReadinessCheck("validation_evidence", "pass", "validation trajectory evidence written")
    )

    checkpoint_dir = output_dir / "checkpoints" / "readiness-probe"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "README.json").write_text(
        json.dumps(
            {
                "schema": "tunix-craftext.checkpoint-probe/v1",
                "purpose": "filesystem and artifact logging readiness check",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.log_artifact(
        checkpoint_artifact(profile.run.name, checkpoint_dir, step=0, role="readiness")
    )
    checks.append(
        ReadinessCheck("checkpoint_evidence", "pass", "checkpoint directory probe written")
    )

    logger.log_artifact(RunArtifact(profile.run.name, str(profile_path), "config", step=0))
    logger.log_artifact(RunArtifact(profile.run.name, str(manifest_path), "profile", step=0))
    logger.log_metric(
        MetricRecord(
            run_id=profile.run.name,
            step=0,
            split="eval",
            phase="server_readiness",
            metrics={
                "ok": all(check.ok for check in checks),
                "device_count": len(devices),
                "snapshot_exists": model_snapshot.is_dir(),
                "python": platform.python_version(),
            },
            policy_version=0,
            checkpoint_path=str(checkpoint_dir),
            trajectory_path=str(validation_path),
        )
    )
    checks.append(ReadinessCheck("observability", "pass", "metrics/artifact JSONL records written"))

    evidence = {
        "run_dir": str(output_dir),
        "provenance": str(manifest_path),
        "metrics": str(logger.metrics_path),
        "validation_trajectories": str(logger.validation_trajectories_path),
        "artifacts": str(logger.artifacts_path),
        "validation_artifact": str(validation_path),
        "checkpoint_probe": str(checkpoint_dir),
    }
    return ServerReadinessReport(
        schema="tunix-craftext.server-readiness/v1",
        mode=mode,
        run_id=profile.run.name,
        run_dir=str(output_dir),
        backend=backend,
        devices=devices,
        checks=tuple(checks),
        evidence=evidence,
    )


def _ensure_evidence_directories(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


def _write_validation_probe(
    profile: AgenticGrpoProfile,
    *,
    output_dir: Path,
    environment_config: Path,
    mode: ReadinessMode,
    scripted_horizon: int,
) -> tuple[Path, float, int]:
    if mode == "scripted":
        return _write_scripted_validation_probe(
            profile,
            output_dir=output_dir,
            environment_config=environment_config,
            scripted_horizon=scripted_horizon,
        )
    path = output_dir / "validation" / "server-readiness-evidence-replay.json"
    artifact = ReplayArtifact(
        config_path=str(environment_config),
        commit="server-readiness",
        backend="evidence-probe",
        steps=(
            ReplayStep(
                index=0,
                prompt=profile.run.goal,
                raw_completion="<action>NOOP</action>",
                action_id=0,
                action_label="NOOP",
                reward=0.0,
                terminated=True,
                observation={"source": "synthetic readiness probe"},
            ),
        ),
    )
    save_replay(path, artifact)
    return path, 0.0, 1


def _write_scripted_validation_probe(
    profile: AgenticGrpoProfile,
    *,
    output_dir: Path,
    environment_config: Path,
    scripted_horizon: int,
) -> tuple[Path, float, int]:
    from .agentic_grpo_smoke import collect_scripted_grpo_group_sync, save_scripted_grpo_smoke

    action_sequences = (("NOOP",) * scripted_horizon, ("LEFT",) * scripted_horizon)
    results = collect_scripted_grpo_group_sync(
        config_path=environment_config,
        goal=profile.run.goal,
        seed=profile.run.seed,
        group_id=0,
        action_sequences=action_sequences,
        horizon=scripted_horizon,
    )
    path = output_dir / "validation" / "server-readiness-scripted-grpo.json"
    save_scripted_grpo_smoke(path, results)
    first = results[0]
    return path, first.total_reward, max(len(first.rewards), 1)
