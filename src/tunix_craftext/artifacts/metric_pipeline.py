"""Composable live metric pipeline for notebooks and training loops.

The pipeline lets a notebook or trainer declare *what* metrics are needed and
how they are computed, then execute those computations in order while logging
both scalar chart records and rich nested snapshots through ``JsonlRunLogger``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from .observability import (
    ArtifactSink,
    JsonValue,
    MetricRecord,
    MetricSnapshotRecord,
    RunSplit,
    flatten_scalar_metrics,
)

MetricPayload = Mapping[str, JsonValue]
MetricCompute = Callable[["MetricComputationContext"], MetricPayload]


class MetricPipelineError(ValueError):
    """Raised when live metric computation cannot proceed."""


@dataclass(frozen=True)
class MetricComputationContext:
    """Inputs and transitive computed metrics visible to one metric source."""

    inputs: Mapping[str, object]
    computed: Mapping[str, MetricPayload]

    def require(self, name: str) -> object:
        """Return one required input by name or fail with an actionable error."""
        try:
            return self.inputs[name]
        except KeyError as error:
            raise MetricPipelineError(f"missing metric input: {name}") from error

    def require_computed(self, name: str) -> MetricPayload:
        """Return metrics produced by an earlier source."""
        try:
            return self.computed[name]
        except KeyError as error:
            raise MetricPipelineError(f"missing computed metric source: {name}") from error


@dataclass(frozen=True)
class MetricSource:
    """One named metric computation and logging target."""

    name: str
    phase: str
    compute: MetricCompute
    split: RunSplit = "train"
    write_snapshot: bool = True

    def __post_init__(self) -> None:
        """Validate source names before pipeline construction."""
        if not self.name:
            raise MetricPipelineError("metric source name must be non-empty")
        if not self.phase:
            raise MetricPipelineError("metric source phase must be non-empty")


@dataclass(frozen=True)
class LiveMetricPipeline:
    """Ordered live metric pipeline bound to one observability sink and run id."""

    sink: ArtifactSink
    run_id: str
    sources: tuple[MetricSource, ...]

    def log(
        self,
        *,
        step: int,
        inputs: Mapping[str, object],
        policy_version: int | None = None,
        checkpoint_path: str | None = None,
        trajectory_path: str | None = None,
    ) -> dict[str, MetricPayload]:
        """Compute all sources in order and append their metric records.

        Each source sees both original ``inputs`` and metrics already computed
        by previous sources. Its output is stored under ``source.name`` and is
        also added to the next source's input context.
        """
        if not self.sources:
            raise MetricPipelineError("metric pipeline must contain at least one source")
        working_inputs: dict[str, object] = dict(inputs)
        computed: dict[str, MetricPayload] = {}
        for source in self.sources:
            payload = dict(
                source.compute(
                    MetricComputationContext(
                        inputs=working_inputs,
                        computed=computed,
                    )
                )
            )
            if not payload:
                raise MetricPipelineError(f"metric source {source.name!r} returned no metrics")
            computed[source.name] = payload
            working_inputs[source.name] = payload
            self.sink.log_metric(
                MetricRecord(
                    run_id=self.run_id,
                    step=step,
                    split=source.split,
                    phase=source.phase,
                    metrics=flatten_scalar_metrics(payload),
                    policy_version=policy_version,
                    checkpoint_path=checkpoint_path,
                    trajectory_path=trajectory_path,
                )
            )
            if source.write_snapshot:
                self.sink.log_metric_snapshot(
                    MetricSnapshotRecord(
                        run_id=self.run_id,
                        step=step,
                        split=source.split,
                        phase=source.phase,
                        metrics=payload,
                        policy_version=policy_version,
                    )
                )
        return computed


class MetricLoggerFactory:
    """Builder for live metric pipelines around a concrete observability sink."""

    def __init__(self, sink: ArtifactSink, *, run_id: str) -> None:
        """Create an empty metric pipeline factory."""
        if not run_id:
            raise MetricPipelineError("run_id must be non-empty")
        self.sink = sink
        self.run_id = run_id
        self._sources: list[MetricSource] = []

    def add_source(
        self,
        *,
        name: str,
        phase: str,
        compute: MetricCompute,
        split: RunSplit = "train",
        write_snapshot: bool = True,
    ) -> MetricLoggerFactory:
        """Register one custom source and return this factory for chaining."""
        self._sources.append(
            MetricSource(
                name=name,
                phase=phase,
                split=split,
                compute=compute,
                write_snapshot=write_snapshot,
            )
        )
        return self

    def add_external_grpo_summary(
        self,
        *,
        input_name: str = "external_grpo_batch",
        name: str = "external_grpo_summary",
        phase: str = "rollout",
        split: RunSplit = "train",
        write_snapshot: bool = True,
    ) -> MetricLoggerFactory:
        """Register summary metrics for an ``ExternalGrpoBatch`` input."""

        def compute(context: MetricComputationContext) -> MetricPayload:
            from ..training.external_grpo import (  # Local import avoids mandatory train import.
                ExternalGrpoBatch,
                summarize_external_grpo_batch,
            )

            value = context.require(input_name)
            if not isinstance(value, ExternalGrpoBatch):
                raise MetricPipelineError(f"{input_name} must be ExternalGrpoBatch")
            return cast(MetricPayload, summarize_external_grpo_batch(value))

        return self.add_source(
            name=name,
            phase=phase,
            split=split,
            compute=compute,
            write_snapshot=write_snapshot,
        )

    def add_input_metrics(
        self,
        *,
        input_name: str,
        name: str,
        phase: str,
        split: RunSplit = "train",
        write_snapshot: bool = True,
    ) -> MetricLoggerFactory:
        """Register a source that logs a JSON metric payload from pipeline inputs.

        This is the generic bridge for notebook/train-loop scalar or nested
        metrics that are already computed by the surrounding code. Prefer this
        over ad-hoc ``MetricRecord`` construction in notebooks: it keeps all
        live logging on the same ``MetricLoggerFactory -> ArtifactSink`` path.
        """

        def compute(context: MetricComputationContext) -> MetricPayload:
            value = context.require(input_name)
            if not isinstance(value, Mapping):
                raise MetricPipelineError(f"{input_name} must be a metric mapping")
            return cast(MetricPayload, value)

        return self.add_source(
            name=name,
            phase=phase,
            split=split,
            compute=compute,
            write_snapshot=write_snapshot,
        )

    def build(self) -> LiveMetricPipeline:
        """Build an immutable live metric pipeline."""
        if not self._sources:
            raise MetricPipelineError("metric pipeline must contain at least one source")
        return LiveMetricPipeline(
            sink=self.sink,
            run_id=self.run_id,
            sources=tuple(self._sources),
        )
