"""Inference engine contracts and optional rollout-generation backends."""

from .async_pipeline import (
    AsyncGenerationRecord,
    collect_generation_results,
    collect_generation_results_from_sync_engine,
)
from .contracts import (
    AsyncInferenceEngine,
    EngineProfile,
    GenerationBatch,
    GenerationResult,
    InferenceBackendError,
    InferenceEngine,
    RequestsLlmBackend,
    SyncToAsyncInferenceEngine,
    as_async_engine,
)
from .tunix_config import TunixGenerationContract
from .vllm_backend import VllmInferenceEngine

__all__ = [
    "EngineProfile",
    "GenerationBatch",
    "GenerationResult",
    "AsyncInferenceEngine",
    "AsyncGenerationRecord",
    "InferenceBackendError",
    "InferenceEngine",
    "RequestsLlmBackend",
    "SyncToAsyncInferenceEngine",
    "TunixGenerationContract",
    "VllmInferenceEngine",
    "as_async_engine",
    "collect_generation_results",
    "collect_generation_results_from_sync_engine",
]
