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
from .registry import build_inference_engine
from .sglang_backend import SglangInferenceEngine
from .sync_pipeline import GenerationRecord, collect_generation_results_sync
from .tunix_config import TunixGenerationContract
from .vanilla_backend import VanillaInferenceEngine
from .vllm_backend import VllmInferenceEngine

__all__ = [
    "EngineProfile",
    "GenerationBatch",
    "GenerationRecord",
    "GenerationResult",
    "AsyncInferenceEngine",
    "AsyncGenerationRecord",
    "InferenceBackendError",
    "InferenceEngine",
    "RequestsLlmBackend",
    "SyncToAsyncInferenceEngine",
    "TunixGenerationContract",
    "SglangInferenceEngine",
    "VanillaInferenceEngine",
    "VllmInferenceEngine",
    "as_async_engine",
    "build_inference_engine",
    "collect_generation_results",
    "collect_generation_results_from_sync_engine",
    "collect_generation_results_sync",
]
