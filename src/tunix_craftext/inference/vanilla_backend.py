"""Vanilla synchronous inference engine adapters."""

from __future__ import annotations

from dataclasses import dataclass

from ..models.llm import BatchLlmBackend
from .contracts import EngineProfile, GenerationBatch, GenerationResult, InferenceBackendError


@dataclass(frozen=True)
class VanillaInferenceEngine:
    """Wrap an existing `BatchLlmBackend` behind the strict inference contract."""

    profile: EngineProfile
    backend: BatchLlmBackend

    def __post_init__(self) -> None:
        """Validate that this adapter is only used for vanilla-like backends."""
        if self.profile.backend not in {
            "vanilla",
            "scripted",
            "tunix-single-device",
            "tunix-vanilla",
        }:
            raise InferenceBackendError(
                "VanillaInferenceEngine requires backend='vanilla', 'scripted', "
                "'tunix-single-device' or 'tunix-vanilla'"
            )

    def generate(self, batch: GenerationBatch) -> GenerationResult:
        """Generate one ordered batch through the wrapped synchronous backend."""
        responses = self.backend.complete_batch(batch.requests)
        if len(responses) != len(batch.requests):
            raise InferenceBackendError("vanilla backend changed batch cardinality")
        return GenerationResult(
            self.profile,
            tuple(responses),
            group_id=batch.group_id,
            policy_version=batch.policy_version,
        )
