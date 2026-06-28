"""Tests for versioned train/validation observability records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tunix_craftext.artifacts.observability import (
    ArtifactSink,
    JsonlRunLogger,
    LoggerMethodMapping,
    MappedLoggerSink,
    MetricRecord,
    RunArtifact,
    ValidationTrajectoryRecord,
    checkpoint_artifact,
    optimizer_state_artifact,
    read_jsonl,
    training_trajectory_artifact,
    validation_trajectory_artifact,
    validation_visualization_artifact,
    weights_artifact,
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
    sink.log_artifact(RunArtifact("protocol", "trajectory/task.json", "validation_trajectory"))

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


class TeamLogger:
    def __init__(self) -> None:
        self.scalars: list[tuple[dict[str, object], int | None]] = []
        self.files: list[tuple[str, str | None, str | None, int | None]] = []
        self.texts: list[tuple[str, str, int | None]] = []
        self.images: list[tuple[str, str | None, int | None]] = []

    def scalars_write(self, values: dict[str, object], *, step: int | None = None) -> None:
        self.scalars.append((values, step))

    def file_write(
        self,
        path: str,
        *,
        name: str | None = None,
        kind: str | None = None,
        step: int | None = None,
    ) -> None:
        self.files.append((path, name, kind, step))

    def text_write(self, text: str, *, name: str, step: int | None = None) -> None:
        self.texts.append((name, text, step))

    def image_write(self, path: str, *, name: str | None = None, step: int | None = None) -> None:
        self.images.append((path, name, step))


def test_mapped_logger_sink_adapts_team_logger_method_names() -> None:
    logger = TeamLogger()
    sink = MappedLoggerSink(
        logger,
        mapping=LoggerMethodMapping(
            log_metrics="scalars_write",
            log_artifact="file_write",
            log_text="text_write",
            log_image="image_write",
        ),
    )

    sink.log_metric(
        MetricRecord("mapped", 2, "train", "update", {"loss": 0.5, "quality": "ok"})
    )
    sink.log_artifact(
        RunArtifact(
            "mapped",
            "trajectory/val/frame.png",
            "validation_visualization",
            name="val-frame",
            step=2,
        )
    )
    sink.log_validation_trajectory(
        ValidationTrajectoryRecord(
            "mapped",
            2,
            "task",
            "trajectory/val/task.json",
            return_sum=1.0,
            episode_length=8,
        )
    )

    assert logger.scalars[0] == ({"train/update/loss": 0.5}, 2)
    assert logger.images == [("trajectory/val/frame.png", "val-frame", 2)]
    assert logger.files == [("trajectory/val/task.json", "task-step-2", "trajectory", 2)]
    assert [name for name, _, _ in logger.texts] == [
        "metric",
        "artifact",
        "validation_trajectory",
        "artifact",
    ]


def test_mapped_logger_sink_accepts_direct_callables() -> None:
    events: list[tuple[str, object]] = []

    sink = MappedLoggerSink(
        object(),
        log_metrics=lambda metrics, **kwargs: events.append(("metrics", metrics)),
        log_artifact=lambda path, **kwargs: events.append(("artifact", path)),
        log_text=lambda text, **kwargs: events.append(("text", text)),
    )

    sink.log_metric(MetricRecord("callable", 0, "train", "update", {"loss": 1.0}))
    sink.log_artifact(RunArtifact("callable", "checkpoint/actor", "checkpoint"))

    assert events[0] == ("metrics", {"train/update/loss": 1.0})
    assert events[1][0] == "text"
    assert events[2] == ("artifact", "checkpoint/actor")


def test_standard_training_artifact_factories_attach_expected_metadata() -> None:
    artifacts = [
        checkpoint_artifact("run", "checkpoints/actor/10", step=10, role="actor", policy_version=3),
        weights_artifact(
            "run",
            "weights/actor.safetensors",
            step=10,
            role="actor",
            policy_version=3,
            quantization="bf16",
        ),
        optimizer_state_artifact("run", "checkpoints/opt/10", step=10, role="critic"),
        training_trajectory_artifact("run", "trajectory/train/task.json", step=10, task_id="task"),
        validation_trajectory_artifact("run", "trajectory/val/task.json", step=10, task_id="task"),
        validation_visualization_artifact(
            "run", "trajectory/val/task.png", step=10, task_id="task"
        ),
    ]

    assert [artifact.kind for artifact in artifacts] == [
        "checkpoint",
        "weights",
        "optimizer_state",
        "training_trajectory",
        "validation_trajectory",
        "validation_visualization",
    ]
    assert artifacts[0].metadata == {"role": "actor"}
    assert artifacts[1].metadata == {"role": "actor", "quantization": "bf16"}
    assert artifacts[-1].display_name == "val-task-visualization-step-10"
