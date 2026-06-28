"""Versioned training observability events for GRPO/PPO runs.

This module is the stable JSONL boundary for training control: scalar train/eval
metrics, checkpoint links and full validation trajectory references are written
as append-only records. Rich dashboards, TensorBoard or W&B integrations may be
added later, but they should consume these records rather than becoming the
primary source of truth.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

JsonScalar: TypeAlias = None | bool | int | float | str
RunSplit: TypeAlias = Literal["train", "val", "eval", "benchmark"]
ArtifactKind: TypeAlias = Literal[
    "trajectory",
    "training_trajectory",
    "validation_trajectory",
    "validation_visualization",
    "profile",
    "checkpoint",
    "weights",
    "optimizer_state",
    "config",
    "model_card",
    "dataset_snapshot",
    "report",
    "other",
]


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


@dataclass(frozen=True)
class RunArtifact:
    """One versioned artifact reference produced by a run.

    :param run_id: Stable run identifier shared with metrics and checkpoints.
    :param path: Local path to the artifact. Upload sinks should read from this path.
    :param kind: Artifact kind such as ``trajectory``, ``checkpoint`` or
        ``validation_visualization``.
    :param name: Human-readable artifact name. Defaults to the local filename.
    :param step: Optional global step associated with the artifact.
    :param policy_version: Optional policy version associated with the artifact.
    :param metadata: Additional scalar metadata to attach to the artifact.
    :param schema: Versioned artifact reference schema.
    """

    run_id: str
    path: str
    kind: ArtifactKind
    name: str | None = None
    step: int | None = None
    policy_version: int | None = None
    metadata: dict[str, JsonScalar] | None = None
    schema: str = "tunix-craftext.artifact/v1"

    def __post_init__(self) -> None:
        """Validate artifact references before logging or uploading."""
        _validate_non_empty(self.run_id, "run_id")
        _validate_non_empty(self.path, "path")
        if self.name is not None:
            _validate_non_empty(self.name, "name")
        if self.step is not None and self.step < 0:
            raise ValueError("step must be non-negative")
        if self.policy_version is not None and self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        for name, value in (self.metadata or {}).items():
            _validate_non_empty(name, "metadata name")
            _validate_scalar(value, f"metadata {name!r}")

    @property
    def display_name(self) -> str:
        """Return the explicit artifact name or the local filename."""
        return self.name or Path(self.path).name


def checkpoint_artifact(
    run_id: str,
    path: Path | str,
    *,
    step: int,
    role: str,
    policy_version: int | None = None,
) -> RunArtifact:
    """Create a standard checkpoint artifact for actor/critic/reference roles."""
    _validate_non_empty(role, "role")
    return RunArtifact(
        run_id=run_id,
        path=str(path),
        kind="checkpoint",
        name=f"{role}-checkpoint-step-{step}",
        step=step,
        policy_version=policy_version,
        metadata={"role": role},
    )


def weights_artifact(
    run_id: str,
    path: Path | str,
    *,
    step: int,
    role: str,
    policy_version: int | None = None,
    quantization: str | None = None,
) -> RunArtifact:
    """Create a standard model-weights artifact for train/eval snapshots."""
    _validate_non_empty(role, "role")
    metadata: dict[str, JsonScalar] = {"role": role}
    if quantization is not None:
        metadata["quantization"] = quantization
    return RunArtifact(
        run_id=run_id,
        path=str(path),
        kind="weights",
        name=f"{role}-weights-step-{step}",
        step=step,
        policy_version=policy_version,
        metadata=metadata,
    )


def optimizer_state_artifact(
    run_id: str,
    path: Path | str,
    *,
    step: int,
    role: str,
    policy_version: int | None = None,
) -> RunArtifact:
    """Create a standard optimizer-state artifact."""
    _validate_non_empty(role, "role")
    return RunArtifact(
        run_id=run_id,
        path=str(path),
        kind="optimizer_state",
        name=f"{role}-optimizer-state-step-{step}",
        step=step,
        policy_version=policy_version,
        metadata={"role": role},
    )


def training_trajectory_artifact(
    run_id: str,
    path: Path | str,
    *,
    step: int,
    task_id: str,
    policy_version: int | None = None,
) -> RunArtifact:
    """Create a full training trajectory artifact reference."""
    _validate_non_empty(task_id, "task_id")
    return RunArtifact(
        run_id=run_id,
        path=str(path),
        kind="training_trajectory",
        name=f"train-{task_id}-step-{step}",
        step=step,
        policy_version=policy_version,
        metadata={"task_id": task_id},
    )


def validation_trajectory_artifact(
    run_id: str,
    path: Path | str,
    *,
    step: int,
    task_id: str,
    policy_version: int | None = None,
) -> RunArtifact:
    """Create a full validation trajectory artifact reference."""
    _validate_non_empty(task_id, "task_id")
    return RunArtifact(
        run_id=run_id,
        path=str(path),
        kind="validation_trajectory",
        name=f"val-{task_id}-step-{step}",
        step=step,
        policy_version=policy_version,
        metadata={"task_id": task_id},
    )


def validation_visualization_artifact(
    run_id: str,
    path: Path | str,
    *,
    step: int,
    task_id: str,
    policy_version: int | None = None,
) -> RunArtifact:
    """Create a validation visualization artifact reference."""
    _validate_non_empty(task_id, "task_id")
    return RunArtifact(
        run_id=run_id,
        path=str(path),
        kind="validation_visualization",
        name=f"val-{task_id}-visualization-step-{step}",
        step=step,
        policy_version=policy_version,
        metadata={"task_id": task_id},
    )


class ArtifactSink(Protocol):
    """Protocol implemented by local, Comet ML and future observability sinks."""

    def log_metric(self, record: MetricRecord) -> None:
        """Log one scalar metric record."""

    def log_validation_trajectory(self, record: ValidationTrajectoryRecord) -> None:
        """Log a validation trajectory summary and its full artifact reference."""

    def log_artifact(self, artifact: RunArtifact) -> None:
        """Log or upload one run artifact."""


@dataclass(frozen=True)
class LoggerMethodMapping:
    """Names of methods exposed by an arbitrary team/local experiment logger.

    The default mapping matches common logger APIs, but a project-specific
    adapter can override any method name or provide direct callables to
    :class:`MappedLoggerSink`.
    """

    log_metrics: str = "log_metrics"
    log_artifact: str = "log_artifact"
    log_text: str = "log_text"
    log_image: str = "log_image"


class MappedLoggerSink:
    """Adapt an arbitrary local/team logger to the ``ArtifactSink`` protocol.

    The wrapped logger only needs a subset of common methods. Numeric metrics
    are sent to ``log_metrics`` when available; scalar context and artifact
    manifests are sent to ``log_text`` as JSON; artifacts are routed to
    ``log_image`` for validation visualizations and to ``log_artifact`` for all
    other artifact kinds.
    """

    def __init__(
        self,
        logger: object,
        *,
        mapping: LoggerMethodMapping | None = None,
        log_metrics: Callable[..., object] | None = None,
        log_artifact: Callable[..., object] | None = None,
        log_text: Callable[..., object] | None = None,
        log_image: Callable[..., object] | None = None,
    ) -> None:
        self.logger = logger
        self.mapping = mapping or LoggerMethodMapping()
        self._log_metrics = log_metrics
        self._log_artifact = log_artifact
        self._log_text = log_text
        self._log_image = log_image

    def log_metric(self, record: MetricRecord) -> None:
        """Log numeric metrics plus full JSON context through the mapped logger."""
        numeric = {
            f"{record.split}/{record.phase}/{name}": value
            for name, value in record.metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        metrics = self._method(self._log_metrics, self.mapping.log_metrics)
        if numeric and metrics is not None:
            _call_logger(metrics, numeric, step=record.step)
        self._log_json_text("metric", _record_payload(record), step=record.step)

    def log_validation_trajectory(self, record: ValidationTrajectoryRecord) -> None:
        """Log validation summary and route the full trajectory as an artifact."""
        metrics = self._method(self._log_metrics, self.mapping.log_metrics)
        if metrics is not None:
            payload: dict[str, int | float] = {
                "val/trajectory/return_sum": record.return_sum,
                "val/trajectory/episode_length": record.episode_length,
            }
            if record.success is not None:
                payload["val/trajectory/success"] = int(record.success)
            _call_logger(metrics, payload, step=record.step)
        self._log_json_text("validation_trajectory", _record_payload(record), step=record.step)
        self.log_artifact(
            RunArtifact(
                run_id=record.run_id,
                path=record.trajectory_path,
                kind="trajectory",
                name=f"{record.task_id}-step-{record.step}",
                step=record.step,
                policy_version=record.policy_version,
                metadata={"task_id": record.task_id, "success": record.success},
            )
        )

    def log_artifact(self, artifact: RunArtifact) -> None:
        """Route one artifact path through the mapped artifact/image methods."""
        if artifact.kind == "validation_visualization":
            image = self._method(self._log_image, self.mapping.log_image)
            if image is not None:
                _call_logger(image, artifact.path, name=artifact.display_name, step=artifact.step)
                self._log_json_text("artifact", _record_payload(artifact), step=artifact.step)
                return
        artifact_logger = self._method(self._log_artifact, self.mapping.log_artifact)
        if artifact_logger is not None:
            _call_logger(
                artifact_logger,
                artifact.path,
                name=artifact.display_name,
                kind=artifact.kind,
                step=artifact.step,
            )
        self._log_json_text("artifact", _record_payload(artifact), step=artifact.step)

    def _method(
        self, explicit: Callable[..., object] | None, method_name: str
    ) -> Callable[..., object] | None:
        if explicit is not None:
            return explicit
        method = getattr(self.logger, method_name, None)
        if callable(method):
            return method
        return None

    def _log_json_text(self, name: str, payload: dict[str, object], *, step: int | None) -> None:
        log_text = self._method(self._log_text, self.mapping.log_text)
        if log_text is not None:
            _call_logger(
                log_text,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                name=name,
                step=step,
            )


class JsonlRunLogger:
    """Append-only writer for versioned run observability JSONL files."""

    def __init__(self, run_dir: Path) -> None:
        """Create a logger rooted at ``artifacts/runs/<run-id>`` style directory."""
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.validation_trajectories_path = run_dir / "validation_trajectories.jsonl"
        self.artifacts_path = run_dir / "artifacts.jsonl"

    def write_metric(self, record: MetricRecord) -> Path:
        """Append one scalar metric event and return the metrics JSONL path."""
        _append_jsonl(self.metrics_path, _record_payload(record))
        return self.metrics_path

    def log_metric(self, record: MetricRecord) -> None:
        """Append one metric event through the generic ``ArtifactSink`` protocol."""
        self.write_metric(record)

    def write_validation_trajectory(self, record: ValidationTrajectoryRecord) -> Path:
        """Append one validation trajectory reference and return its JSONL path."""
        _append_jsonl(self.validation_trajectories_path, _record_payload(record))
        return self.validation_trajectories_path

    def log_validation_trajectory(self, record: ValidationTrajectoryRecord) -> None:
        """Append one validation trajectory event through the sink protocol."""
        self.write_validation_trajectory(record)

    def write_artifact(self, artifact: RunArtifact) -> Path:
        """Append one artifact reference and return the artifacts JSONL path."""
        _append_jsonl(self.artifacts_path, _record_payload(artifact))
        return self.artifacts_path

    def log_artifact(self, artifact: RunArtifact) -> None:
        """Append one artifact reference through the generic sink protocol."""
        self.write_artifact(artifact)


def read_jsonl(path: Path) -> tuple[dict[str, JsonScalar | dict[str, JsonScalar]], ...]:
    """Read a JSONL observability file into immutable dictionaries for tests/tools."""
    if not path.exists():
        return ()
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)


def _record_payload(
    record: MetricRecord | ValidationTrajectoryRecord | RunArtifact,
) -> dict[str, object]:
    return asdict(record)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _call_logger(method: Callable[..., object], *args: object, **kwargs: object) -> object:
    """Call a loosely-typed logger method while tolerating simpler local APIs."""
    try:
        return method(*args, **kwargs)
    except TypeError:
        return method(*args)


def _validate_non_empty(value: str, field: str) -> None:
    if not value:
        raise ValueError(f"{field} must be non-empty")


def _validate_scalar(value: JsonScalar, field: str) -> None:
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        raise ValueError(f"{field} must be a JSON scalar")
