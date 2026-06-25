"""Tests for the optional Comet ML observability sink without real network calls."""

from __future__ import annotations

from pathlib import Path

from tunix_craftext.comet_adapter import CometMlSink
from tunix_craftext.observability import MetricRecord, RunArtifact, ValidationTrajectoryRecord


class FakeExperiment:
    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, object], int | None]] = []
        self.assets: list[tuple[str, str | None, int | None]] = []
        self.asset_folders: list[tuple[str, int | None]] = []
        self.images: list[tuple[str, str | None, int | None]] = []
        self.other: list[tuple[str, object, int | None]] = []
        self.ended = False

    def log_metrics(self, metrics: dict[str, object], *, step: int | None = None) -> None:
        self.metrics.append((metrics, step))

    def log_asset(
        self, path: str, *, file_name: str | None = None, step: int | None = None
    ) -> None:
        self.assets.append((path, file_name, step))

    def log_asset_folder(
        self, folder: str, *, step: int | None = None, log_file_name: bool = True
    ) -> None:
        del log_file_name
        self.asset_folders.append((folder, step))

    def log_image(self, path: str, *, name: str | None = None, step: int | None = None) -> None:
        self.images.append((path, name, step))

    def log_other(self, name: str, value: object, *, step: int | None = None) -> None:
        self.other.append((name, value, step))

    def end(self) -> None:
        self.ended = True


def test_comet_sink_logs_numeric_metrics_and_scalar_context() -> None:
    experiment = FakeExperiment()
    sink = CometMlSink(experiment)

    sink.log_metric(
        MetricRecord(
            run_id="run",
            step=5,
            split="train",
            phase="update",
            metrics={"loss": 1.0, "quality": "ok", "success": True},
            policy_version=3,
            checkpoint_path="checkpoints/actor/5",
        )
    )

    assert experiment.metrics == [({"train/update/loss": 1.0}, 5)]
    assert ("policy_version", 3, 5) in experiment.other
    assert ("checkpoint_path", "checkpoints/actor/5", 5) in experiment.other
    assert ("metric/quality", "ok", 5) in experiment.other
    assert ("metric/success", True, 5) in experiment.other


def test_comet_sink_logs_validation_trajectory_summary_and_full_artifact() -> None:
    experiment = FakeExperiment()
    sink = CometMlSink(experiment)

    sink.log_validation_trajectory(
        ValidationTrajectoryRecord(
            run_id="run",
            step=7,
            task_id="safe-task",
            trajectory_path="trajectory/val/safe-task.json",
            return_sum=1.5,
            episode_length=8,
            success=True,
            policy_version=4,
            metrics={"fallback_count": 0},
        )
    )

    metrics, step = experiment.metrics[0]
    assert step == 7
    assert metrics["val/trajectory/return_sum"] == 1.5
    assert metrics["val/trajectory/success"] == 1
    assert metrics["val/trajectory/fallback_count"] == 0
    assert experiment.assets == [("trajectory/val/safe-task.json", "safe-task-step-7", 7)]
    assert ("task_id", "safe-task", 7) in experiment.other


def test_comet_sink_routes_visualizations_and_folders(tmp_path: Path) -> None:
    experiment = FakeExperiment()
    sink = CometMlSink(experiment)
    folder = tmp_path / "checkpoints"
    folder.mkdir()

    sink.log_artifact(
        RunArtifact(
            run_id="run",
            path="artifacts/val/frame.png",
            kind="validation_visualization",
            name="val-frame",
            step=9,
        )
    )
    sink.log_artifact(RunArtifact("run", str(folder), "checkpoint", step=9))
    sink.log_artifact(
        RunArtifact(
            "run",
            "artifacts/weights/actor.safetensors",
            "weights",
            name="actor-weights",
            step=9,
        )
    )
    sink.end()

    assert experiment.images == [("artifacts/val/frame.png", "val-frame", 9)]
    assert experiment.asset_folders == [(str(folder), 9)]
    assert experiment.assets == [("artifacts/weights/actor.safetensors", "actor-weights", 9)]
    assert experiment.ended is True
