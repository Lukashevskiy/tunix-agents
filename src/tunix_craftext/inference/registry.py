"""Inference backend registry/factory for explicit engine profiles."""

from __future__ import annotations

from collections.abc import Mapping

from ..models.llm import BatchLlmBackend
from .contracts import EngineProfile, InferenceBackendError, InferenceEngine
from .sglang_backend import SglangInferenceEngine
from .vanilla_backend import VanillaInferenceEngine
from .vllm_backend import VllmInferenceEngine


def build_inference_engine(
    profile: EngineProfile,
    *,
    vanilla_backends: Mapping[str, BatchLlmBackend] | None = None,
) -> InferenceEngine:
    """Build one inference engine from a strict `EngineProfile`.

    Vanilla/scripted/Tunix-single-device engines need an already constructed
    `BatchLlmBackend`, because their model loading is project-specific. vLLM
    and future server backends own their allocation through the profile.
    """
    backend = profile.backend
    if backend == "vllm-offload":
        return VllmInferenceEngine.from_profile(profile)
    if backend in {"sglang", "sglang-jax", "tunix-sglang_jax"}:
        return SglangInferenceEngine.from_profile(profile)
    if backend in {"vanilla", "scripted", "tunix-single-device", "tunix-vanilla"}:
        backends = vanilla_backends or {}
        if profile.name not in backends:
            raise InferenceBackendError(
                f"vanilla backend profile '{profile.name}' requires a supplied BatchLlmBackend"
            )
        return VanillaInferenceEngine(profile, backends[profile.name])
    raise InferenceBackendError(f"unsupported inference backend: {backend}")
