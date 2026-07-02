"""Optional Comet ML sink for run metrics, trajectories and artifacts.

The adapter is intentionally thin and optional. Core training writes local
JSONL/trajectory evidence first; this sink mirrors that evidence into a Comet
experiment when the user explicitly provides or creates one.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .observability import (
    MetricRecord,
    MetricSnapshotRecord,
    RunArtifact,
    ValidationTrajectoryRecord,
    flatten_scalar_metrics,
)


class CometMlSink:
    """Mirror versioned observability records into a Comet ML experiment.

    :param experiment: Existing ``comet_ml.Experiment``-compatible object.
        Tests may pass a fake object implementing ``log_metrics``, ``log_asset``
        and ``log_other``; importing this module never imports Comet.
    """

    def __init__(self, experiment: object) -> None:
        self.experiment = experiment

    @classmethod
    def create_experiment(cls, **kwargs: Any) -> CometMlSink:
        """Create a Comet experiment lazily from ``comet_ml.Experiment``.

        :param kwargs: Forwarded to ``comet_ml.Experiment``. Typical values are
            ``project_name``, ``workspace`` and ``api_key``.
        :raises RuntimeError: If ``comet_ml`` is not installed.
        """
        try:
            from comet_ml import Experiment  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - unit tests use fake experiment.
            raise RuntimeError(
                "Install comet_ml and pass Comet credentials to use CometMlSink"
            ) from error
        return cls(Experiment(**kwargs))

    def log_metric(self, record: MetricRecord) -> None:
        """Log numeric metrics and scalar context to Comet."""
        prefix = f"{record.split}/{record.phase}"
        numeric_metrics = {
            f"{prefix}/{name}": value
            for name, value in record.metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if numeric_metrics:
            _call_with_fallback(
                getattr(self.experiment, "log_metrics"),
                numeric_metrics,
                step=record.step,
            )
        self._log_context(
            {
                "schema": record.schema,
                "run_id": record.run_id,
                "split": record.split,
                "phase": record.phase,
                "policy_version": record.policy_version,
                "checkpoint_path": record.checkpoint_path,
                "trajectory_path": record.trajectory_path,
                **{
                    f"metric/{key}": value
                    for key, value in record.metrics.items()
                    if not isinstance(value, (int, float)) or isinstance(value, bool)
                },
            },
            step=record.step,
        )

    def log_metric_snapshot(self, record: MetricSnapshotRecord) -> None:
        """Log scalar leaves and preserve nested metric context in Comet."""
        prefix = f"{record.split}/{record.phase}"
        numeric_metrics = {
            f"{prefix}/{name}": value
            for name, value in flatten_scalar_metrics(record.metrics).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if numeric_metrics:
            _call_with_fallback(
                getattr(self.experiment, "log_metrics"),
                numeric_metrics,
                step=record.step,
            )
        self._log_context(
            {
                "schema": record.schema,
                "run_id": record.run_id,
                "split": record.split,
                "phase": record.phase,
                "policy_version": record.policy_version,
                "metric_snapshot": json.dumps(record.metrics, ensure_ascii=False, sort_keys=True),
            },
            step=record.step,
        )

    def log_validation_trajectory(self, record: ValidationTrajectoryRecord) -> None:
        """Log validation summary metrics and upload the full trajectory file."""
        metrics: dict[str, int | float] = {
            "val/trajectory/return_sum": record.return_sum,
            "val/trajectory/episode_length": record.episode_length,
        }
        if record.success is not None:
            metrics["val/trajectory/success"] = int(record.success)
        for name, value in (record.metrics or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"val/trajectory/{name}"] = value
        _call_with_fallback(getattr(self.experiment, "log_metrics"), metrics, step=record.step)
        self._log_context(
            {
                "schema": record.schema,
                "run_id": record.run_id,
                "task_id": record.task_id,
                "policy_version": record.policy_version,
                "trajectory_path": record.trajectory_path,
            },
            step=record.step,
        )
        self.log_artifact(
            RunArtifact(
                run_id=record.run_id,
                path=record.trajectory_path,
                kind="validation_trajectory",
                name=f"{record.task_id}-step-{record.step}",
                step=record.step,
                policy_version=record.policy_version,
                metadata={"task_id": record.task_id, "success": record.success},
            )
        )

    def log_artifact(self, artifact: RunArtifact) -> None:
        """Upload an artifact path to Comet and attach scalar metadata."""
        path = Path(artifact.path)
        if path.is_dir() and hasattr(self.experiment, "log_asset_folder"):
            _call_with_fallback(
                getattr(self.experiment, "log_asset_folder"),
                str(path),
                step=artifact.step,
                log_file_name=True,
            )
        elif artifact.kind == "validation_visualization" and hasattr(self.experiment, "log_image"):
            _call_with_fallback(
                getattr(self.experiment, "log_image"),
                str(path),
                name=artifact.display_name,
                step=artifact.step,
            )
        else:
            _call_with_fallback(
                getattr(self.experiment, "log_asset"),
                str(path),
                file_name=artifact.display_name,
                step=artifact.step,
            )
        self._log_context(
            {
                "schema": artifact.schema,
                "run_id": artifact.run_id,
                "artifact_kind": artifact.kind,
                "artifact_name": artifact.display_name,
                "artifact_path": artifact.path,
                "policy_version": artifact.policy_version,
                **{
                    f"artifact/{name}": value
                    for name, value in (artifact.metadata or {}).items()
                },
            },
            step=artifact.step,
        )

    def end(self) -> None:
        """End the underlying Comet experiment when it exposes ``end``."""
        end = getattr(self.experiment, "end", None)
        if end is not None:
            end()

    def _log_context(self, values: dict[str, object], *, step: int | None) -> None:
        log_other = getattr(self.experiment, "log_other", None)
        if log_other is None:
            return
        for name, value in values.items():
            if value is not None:
                _call_with_fallback(log_other, name, value, step=step)


def _call_with_fallback(method: Callable[..., object], *args: object, **kwargs: object) -> object:
    """Call a Comet method with kwargs, then fall back for older/fake SDKs."""
    try:
        return method(*args, **kwargs)
    except TypeError:
        return method(*args)
