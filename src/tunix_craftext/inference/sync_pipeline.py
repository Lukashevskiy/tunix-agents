"""Synchronous generation collection utilities using the shared inference contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from .contracts import GenerationBatch, GenerationResult, InferenceEngine


@dataclass(frozen=True)
class GenerationRecord:
    """One ordered generation result with submission index provenance."""

    index: int
    batch: GenerationBatch
    result: GenerationResult


@dataclass(frozen=True)
class GenerationTiming:
    """Wall-clock timing for one generation batch collection step."""

    queued_ms: float
    backend_ms: float
    total_ms: float
    response_latency_ms: tuple[float, ...]


@dataclass(frozen=True)
class ProfiledGenerationRecord:
    """Generation record with host-side collection timing."""

    record: GenerationRecord
    timing: GenerationTiming


def collect_generation_results_sync(
    engine: InferenceEngine,
    batches: Sequence[GenerationBatch],
) -> tuple[GenerationRecord, ...]:
    """Collect generation batches sequentially while preserving submission order."""
    return tuple(
        GenerationRecord(index, batch, engine.generate(batch))
        for index, batch in enumerate(tuple(batches))
    )


def collect_generation_results_sync_profiled(
    engine: InferenceEngine,
    batches: Sequence[GenerationBatch],
) -> tuple[ProfiledGenerationRecord, ...]:
    """Collect generation batches sequentially with per-batch timing evidence."""
    records: list[ProfiledGenerationRecord] = []
    for index, batch in enumerate(tuple(batches)):
        started = perf_counter()
        result = engine.generate(batch)
        elapsed_ms = (perf_counter() - started) * 1000.0
        response_latency = tuple(
            float(response.latency_ms)
            for response in result.responses
            if response.latency_ms is not None
        )
        records.append(
            ProfiledGenerationRecord(
                GenerationRecord(index, batch, result),
                GenerationTiming(
                    queued_ms=0.0,
                    backend_ms=elapsed_ms,
                    total_ms=elapsed_ms,
                    response_latency_ms=response_latency,
                ),
            )
        )
    return tuple(records)
