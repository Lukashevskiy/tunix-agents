"""Tests for versioned train/validation observability records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tunix_craftext.observability import (
    ArtifactSink,
    JsonlRunLogger,
    MetricRecord,
    RunArtifact,
    ValidationTrajectoryRecord,
    read_jsonl,
)


def test_jsonl_run_logger_writes_train_metrics_and_validation_trajectory(
    tmp_path: Path,
) -> None:
    logger = JsonlRunLogger(tmp_path / "artifacts/runs/test")

    metrics_path = logger.write_metric(
        MetricRecord(
            run_id="test",
            step=3,
            split="train",
            phase="update",
            policy_version=2,
            checkpoint_path="checkpoints/actor/3",
            metrics={"loss": 1.25, "kl": 0.01, "invalid_action_rate": 0.0},
        )
    )
    trajectories_path = logger.write_validation_trajectory(
        ValidationTrajectoryRecord(
            run_id="test",
            step=3,
            task_id="safe-avoid-enemy",
            trajectory_path="trajectory/val/safe-avoid-enemy-step-3.json",
            return_sum=0.5,
            episode_length=8,
            success=False,
            policy_version=2,
            metrics={"fallback_count": 1, "mean_token_logprob": -2.0},
        )
    )
    artifact_path = logger.write_artifact(
        RunArtifact(
            run_id="test",
            path="trajectory/val/safe-avoid-enemy-step-3.json",
            kind="trajectory",
            step=3,
            policy_version=2,
            metadata={"task_id": "safe-avoid-enemy"},
        )
    )

    [metric] = read_jsonl(metrics_path)
    [trajectory] = read_jsonl(trajectories_path)
    [artifact] = read_jsonl(artifact_path)

    assert metric["schema"] == "tunix-craftext.metric/v1"
    assert metric["split"] == "train"
    assert metric["metrics"]["loss"] == 1.25
    assert metric["checkpoint_path"] == "checkpoints/actor/3"
    assert trajectory["schema"] == "tunix-craftext.validation-trajectory/v1"
    assert trajectory["trajectory_path"].endswith("safe-avoid-enemy-step-3.json")
    assert trajectory["metrics"]["fallback_count"] == 1
    assert artifact["schema"] == "tunix-craftext.artifact/v1"
    assert artifact["kind"] == "trajectory"
    assert artifact["metadata"]["task_id"] == "safe-avoid-enemy"


def test_jsonl_run_logger_appends_records_in_order(tmp_path: Path) -> None:
    logger = JsonlRunLogger(tmp_path / "run")

    for step in range(2):
        logger.write_metric(
            MetricRecord(
                run_id="append",
                step=step,
                split="train",
                phase="rollout",
                metrics={"reward": float(step)},
            )
        )

    lines = logger.metrics_path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]

    assert [payload["step"] for payload in payloads] == [0, 1]
    assert [payload["metrics"]["reward"] for payload in payloads] == [0.0, 1.0]


def test_metric_record_rejects_non_scalar_metrics() -> None:
    with pytest.raises(ValueError, match="JSON scalar"):
        MetricRecord(
            run_id="bad",
            step=0,
            split="train",
            phase="update",
            metrics={"loss_curve": [1.0, 0.5]},  # type: ignore[dict-item]
        )


def test_validation_trajectory_requires_full_artifact_reference() -> None:
    with pytest.raises(ValueError, match="trajectory_path"):
        ValidationTrajectoryRecord(
            run_id="bad",
            step=0,
            task_id="task",
            trajectory_path="",
            return_sum=0.0,
            episode_length=1,
        )


def test_jsonl_run_logger_implements_artifact_sink_protocol(tmp_path: Path) -> None:
    sink: ArtifactSink = JsonlRunLogger(tmp_path / "run")

    sink.log_metric(
        MetricRecord("protocol", 0, "train", "update", {"loss": 0.0})
    )
    sink.log_validation_trajectory(
        ValidationTrajectoryRecord(
            "protocol",
            0,
            "task",
            "trajectory/task.json",
            return_sum=0.0,
            episode_length=1,
        )
    )
    sink.log_artifact(RunArtifact("protocol", "trajectory/task.json", "trajectory"))

    assert (tmp_path / "run" / "metrics.jsonl").is_file()
    assert (tmp_path / "run" / "validation_trajectories.jsonl").is_file()
    assert (tmp_path / "run" / "artifacts.jsonl").is_file()


def test_run_artifact_rejects_non_scalar_metadata() -> None:
    with pytest.raises(ValueError, match="JSON scalar"):
        RunArtifact(
            run_id="bad",
            path="artifact.json",
            kind="other",
            metadata={"array": [1, 2]},  # type: ignore[dict-item]
        )
