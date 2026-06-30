"""SGLang inference adapter placeholder with explicit unsupported behavior."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import EngineProfile, GenerationBatch, GenerationResult, InferenceBackendError


@dataclass(frozen=True)
class SglangInferenceEngine:
    """Reserved SGLang adapter boundary for future rollout engines."""

    profile: EngineProfile

    @classmethod
    def from_profile(cls, profile: EngineProfile) -> SglangInferenceEngine:
        """Create a placeholder SGLang engine from an explicit profile."""
        if profile.backend not in {"sglang", "sglang-jax", "tunix-sglang_jax"}:
            raise InferenceBackendError("SglangInferenceEngine requires an sglang backend profile")
        return cls(profile)

    def generate(self, batch: GenerationBatch) -> GenerationResult:
        """Fail until SGLang runtime integration is implemented and tested."""
        del batch
        raise InferenceBackendError(
            "SGLang inference backend is declared but not implemented yet. "
            "Use vanilla/tunix-single-device or vllm-offload for current runs."
        )
