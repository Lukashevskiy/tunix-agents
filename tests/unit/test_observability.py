"""Tests for versioned train/validation observability records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tunix_craftext.artifacts.metric_pipeline import MetricLoggerFactory, MetricPipelineError
from tunix_craftext.artifacts.observability import (
    ArtifactSink,
    CompositeArtifactSink,
    JsonlRunLogger,
    LoggerMethodMapping,
    MappedLoggerSink,
    MetricRecord,
    MetricSnapshotRecord,
    RunArtifact,
    ValidationTrajectoryRecord,
    checkpoint_artifact,
    flatten_scalar_metrics,
    optimizer_state_artifact,
    read_jsonl,
    training_trajectory_artifact,
    validation_trajectory_artifact,
    validation_visualization_artifact,
    weights_artifact,
)
from tunix_craftext.artifacts.replay import ReplayArtifact, ReplayStep
from tunix_craftext.training.external_grpo import external_grpo_batch_from_replays


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


def test_metric_logger_factory_logs_live_nested_metrics_and_scalar_stream(tmp_path: Path) -> None:
    logger = JsonlRunLogger(tmp_path / "run")
    pipeline = (
        MetricLoggerFactory(logger, run_id="live")
        .add_input_metrics(input_name="rollout_metrics", name="rollout", phase="rollout")
        .build()
    )

    pipeline.log(
        step=7,
        inputs={
            "rollout_metrics": {
                "loss": 1.5,
                "action_distribution": {"NOOP": 0.75, "DO": 0.25},
                "top_actions": [{"action": "NOOP", "count": 3}],
            }
        },
    )

    [metric] = read_jsonl(logger.metrics_path)
    [snapshot] = read_jsonl(logger.metric_snapshots_path)

    assert metric["metrics"]["loss"] == 1.5
    assert metric["metrics"]["action_distribution/NOOP"] == 0.75
    assert "top_actions" not in metric["metrics"]
    assert snapshot["schema"] == "tunix-craftext.metric-snapshot/v1"
    assert snapshot["metrics"]["top_actions"][0]["action"] == "NOOP"


def test_flatten_scalar_metrics_rejects_payload_without_scalar_leaf() -> None:
    with pytest.raises(ValueError, match="at least one scalar"):
        flatten_scalar_metrics({"top_actions": [{"action": "NOOP"}]})


def test_metric_record_rejects_non_scalar_metrics() -> None:
    with pytest.raises(ValueError, match="JSON scalar"):
        MetricRecord(
            run_id="bad",
            step=0,
            split="train",
            phase="update",
            metrics={"loss_curve": [1.0, 0.5]},  # type: ignore[dict-item]
        )


def test_metric_snapshot_record_accepts_nested_json_metrics() -> None:
    record = MetricSnapshotRecord(
        run_id="snapshot",
        step=1,
        split="train",
        phase="rollout",
        metrics={"action_counts": {"NOOP": 2}, "top_actions": [{"action": "NOOP"}]},
    )

    assert record.schema == "tunix-craftext.metric-snapshot/v1"


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
    sink.log_metric_snapshot(
        MetricSnapshotRecord("protocol", 0, "train", "rollout", {"action_counts": {"NOOP": 1}})
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
    assert (tmp_path / "run" / "metric_snapshots.jsonl").is_file()
    assert (tmp_path / "run" / "validation_trajectories.jsonl").is_file()
    assert (tmp_path / "run" / "artifacts.jsonl").is_file()


def test_composite_artifact_sink_fans_out_records(tmp_path: Path) -> None:
    first = JsonlRunLogger(tmp_path / "first")
    second = JsonlRunLogger(tmp_path / "second")
    sink = CompositeArtifactSink(first, second)

    sink.log_metric(MetricRecord("fanout", 1, "train", "update", {"loss": 0.2}))
    sink.log_metric_snapshot(
        MetricSnapshotRecord("fanout", 1, "train", "rollout", {"action_counts": {"NOOP": 2}})
    )

    assert read_jsonl(first.metrics_path) == read_jsonl(second.metrics_path)
    assert read_jsonl(first.metric_snapshots_path) == read_jsonl(second.metric_snapshots_path)


def test_composite_artifact_sink_rejects_empty_sink_list() -> None:
    with pytest.raises(ValueError, match="at least one sink"):
        CompositeArtifactSink()


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
    sink.log_metric_snapshot(
        MetricSnapshotRecord(
            "mapped",
            2,
            "train",
            "rollout",
            {"action_distribution": {"NOOP": 1.0}, "top_actions": [{"action": "NOOP"}]},
        )
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
    assert logger.scalars[1] == ({"train/rollout/action_distribution/NOOP": 1.0}, 2)
    assert logger.images == [("trajectory/val/frame.png", "val-frame", 2)]
    assert logger.files == [("trajectory/val/task.json", "task-step-2", "trajectory", 2)]
    assert [name for name, _, _ in logger.texts] == [
        "metric",
        "metric_snapshot",
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


def test_metric_logger_factory_composes_transitive_sources(tmp_path: Path) -> None:
    logger = JsonlRunLogger(tmp_path / "run")
    batch = external_grpo_batch_from_replays(
        goal="collect wood",
        group_prefix="wood",
        group_size=2,
        replays=(_replay_for_metrics("NOOP", 0.0), _replay_for_metrics("DO", 1.0)),
    )
    pipeline = (
        MetricLoggerFactory(logger, run_id="factory")
        .add_external_grpo_summary()
        .add_source(
            name="derived",
            phase="update",
            compute=lambda context: {
                "sample_count_seen": context.require_computed("external_grpo_summary")[
                    "sample_count"
                ],
            },
            write_snapshot=False,
        )
        .build()
    )

    computed = pipeline.log(step=4, inputs={"external_grpo_batch": batch})
    metrics = read_jsonl(logger.metrics_path)
    snapshots = read_jsonl(logger.metric_snapshots_path)

    assert computed["external_grpo_summary"]["action_counts"] == {"DO": 1, "NOOP": 1}
    assert computed["derived"]["sample_count_seen"] == 2
    assert len(metrics) == 2
    assert len(snapshots) == 1
    assert metrics[0]["phase"] == "rollout"
    assert metrics[1]["phase"] == "update"


def test_metric_logger_factory_writes_through_generic_artifact_sink(tmp_path: Path) -> None:
    jsonl = JsonlRunLogger(tmp_path / "jsonl")
    team = TeamLogger()
    sink = CompositeArtifactSink(
        jsonl,
        MappedLoggerSink(
            team,
            mapping=LoggerMethodMapping(
                log_metrics="scalars_write",
                log_artifact="file_write",
                log_text="text_write",
                log_image="image_write",
            ),
        ),
    )
    pipeline = (
        MetricLoggerFactory(sink, run_id="generic")
        .add_source(
            name="rollout",
            phase="rollout",
            compute=lambda context: {
                "reward": context.require("reward"),
                "action_distribution": {"NOOP": 1.0},
                "top_actions": [{"action": "NOOP", "count": 1}],
            },
        )
        .build()
    )

    pipeline.log(step=3, inputs={"reward": 1.0}, policy_version=2)

    [metric] = read_jsonl(jsonl.metrics_path)
    [snapshot] = read_jsonl(jsonl.metric_snapshots_path)
    assert metric["metrics"]["reward"] == 1.0
    assert snapshot["policy_version"] == 2
    assert team.scalars[0][0] == {
        "train/rollout/reward": 1.0,
        "train/rollout/action_distribution/NOOP": 1.0,
    }
    assert [name for name, _, _ in team.texts] == ["metric", "metric_snapshot"]


def test_metric_logger_factory_rejects_non_mapping_input_metrics(tmp_path: Path) -> None:
    pipeline = (
        MetricLoggerFactory(JsonlRunLogger(tmp_path / "run"), run_id="generic")
        .add_input_metrics(input_name="metrics", name="metrics", phase="update")
        .build()
    )

    with pytest.raises(MetricPipelineError, match="metric mapping"):
        pipeline.log(step=0, inputs={"metrics": 1.0})


def test_metric_logger_factory_rejects_missing_external_grpo_input(tmp_path: Path) -> None:
    pipeline = MetricLoggerFactory(
        JsonlRunLogger(tmp_path / "run"),
        run_id="factory",
    ).add_external_grpo_summary().build()

    with pytest.raises(MetricPipelineError, match="missing metric input"):
        pipeline.log(step=0, inputs={})


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


def _replay_for_metrics(action: str, reward: float) -> ReplayArtifact:
    return ReplayArtifact(
        config_path="configs/env/text/qwen_craftext.yaml",
        commit="abc123",
        backend="vllm-offload",
        steps=(
            ReplayStep(
                index=0,
                prompt="goal",
                raw_completion=f"<action>{action}</action>",
                action_id=0,
                action_label=action,
                reward=reward,
                terminated=True,
                token_ids=(1,),
                token_logprobs=(-0.1,),
                prompt_token_ids=(1, 2),
            ),
        ),
    )
