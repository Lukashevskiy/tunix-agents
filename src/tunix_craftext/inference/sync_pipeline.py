"""Synchronous generation collection utilities using the shared inference contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .contracts import GenerationBatch, GenerationResult, InferenceEngine


@dataclass(frozen=True)
class GenerationRecord:
    """One ordered generation result with submission index provenance."""

    index: int
    batch: GenerationBatch
    result: GenerationResult


def collect_generation_results_sync(
    engine: InferenceEngine,
    batches: Sequence[GenerationBatch],
) -> tuple[GenerationRecord, ...]:
    """Collect generation batches sequentially while preserving submission order."""
    return tuple(
        GenerationRecord(index, batch, engine.generate(batch))
        for index, batch in enumerate(tuple(batches))
    )
