"""Versioned training observability events for GRPO/PPO runs.

This module is the stable JSONL boundary for training control: scalar train/eval
metrics, checkpoint links and full validation trajectory references are written
as append-only records. Rich dashboards, TensorBoard or W&B integrations may be
added later, but they should consume these records rather than becoming the
primary source of truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

JsonScalar: TypeAlias = None | bool | int | float | str
RunSplit: TypeAlias = Literal["train", "val", "eval", "benchmark"]


@dataclass(frozen=True)
class MetricRecord:
    """One scalar metric event emitted by train/validation/evaluation loops.

    :param run_id: Stable run identifier shared with checkpoints and trajectories.
    :param step: Global learner step or rollout/update counter.
    :param split: Event split: ``train``, ``val``, ``eval`` or ``benchmark``.
    :param phase: Fine-grained phase such as ``rollout``, ``update`` or ``eval``.
    :param metrics: Scalar JSON-compatible metrics.
    :param policy_version: Optional policy version attached to rollout/update.
    :param checkpoint_path: Optional checkpoint root or role checkpoint path.
    :param trajectory_path: Optional related trajectory JSON/JSONL artifact.
    :param schema: Versioned record schema.
    """

    run_id: str
    step: int
    split: RunSplit
    phase: str
    metrics: dict[str, JsonScalar]
    policy_version: int | None = None
    checkpoint_path: str | None = None
    trajectory_path: str | None = None
    schema: str = "tunix-craftext.metric/v1"

    def __post_init__(self) -> None:
        """Validate scalar metrics and identifiers before writing evidence."""
        _validate_non_empty(self.run_id, "run_id")
        _validate_non_empty(self.phase, "phase")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.policy_version is not None and self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if not self.metrics:
            raise ValueError("metrics must not be empty")
        for name, value in self.metrics.items():
            _validate_non_empty(name, "metric name")
            _validate_scalar(value, f"metric {name!r}")


@dataclass(frozen=True)
class ValidationTrajectoryRecord:
    """Reference to a fully persisted validation trajectory artifact.

    :param run_id: Stable run identifier shared with metrics and checkpoints.
    :param step: Global learner step when validation was collected.
    :param task_id: Fixed validation task identifier.
    :param trajectory_path: Path to full replay/trajectory evidence.
    :param return_sum: Total environment return for this trajectory.
    :param episode_length: Number of environment/tool steps.
    :param success: Optional task success flag.
    :param policy_version: Optional policy version used for the rollout.
    :param metrics: Additional scalar summary metrics such as invalid-action rate.
    :param schema: Versioned record schema.
    """

    run_id: str
    step: int
    task_id: str
    trajectory_path: str
    return_sum: float
    episode_length: int
    success: bool | None = None
    policy_version: int | None = None
    metrics: dict[str, JsonScalar] | None = None
    schema: str = "tunix-craftext.validation-trajectory/v1"

    def __post_init__(self) -> None:
        """Validate trajectory references before logging validation evidence."""
        _validate_non_empty(self.run_id, "run_id")
        _validate_non_empty(self.task_id, "task_id")
        _validate_non_empty(self.trajectory_path, "trajectory_path")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.policy_version is not None and self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        for name, value in (self.metrics or {}).items():
            _validate_non_empty(name, "metric name")
            _validate_scalar(value, f"metric {name!r}")


class JsonlRunLogger:
    """Append-only writer for versioned run observability JSONL files."""

    def __init__(self, run_dir: Path) -> None:
        """Create a logger rooted at ``artifacts/runs/<run-id>`` style directory."""
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.validation_trajectories_path = run_dir / "validation_trajectories.jsonl"

    def write_metric(self, record: MetricRecord) -> Path:
        """Append one scalar metric event and return the metrics JSONL path."""
        _append_jsonl(self.metrics_path, _record_payload(record))
        return self.metrics_path

    def write_validation_trajectory(self, record: ValidationTrajectoryRecord) -> Path:
        """Append one validation trajectory reference and return its JSONL path."""
        _append_jsonl(self.validation_trajectories_path, _record_payload(record))
        return self.validation_trajectories_path


def read_jsonl(path: Path) -> tuple[dict[str, JsonScalar | dict[str, JsonScalar]], ...]:
    """Read a JSONL observability file into immutable dictionaries for tests/tools."""
    if not path.exists():
        return ()
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)


def _record_payload(record: MetricRecord | ValidationTrajectoryRecord) -> dict[str, object]:
    return asdict(record)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _validate_non_empty(value: str, field: str) -> None:
    if not value:
        raise ValueError(f"{field} must be non-empty")


def _validate_scalar(value: JsonScalar, field: str) -> None:
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        raise ValueError(f"{field} must be a JSON scalar")
