"""Inference engine contracts and optional rollout-generation backends."""

from .async_pipeline import (
    AsyncGenerationRecord,
    collect_generation_results,
    collect_generation_results_from_sync_engine,
    collect_generation_results_profiled,
)
from .config import (
    AsyncCollectionConfig,
    GenerationPipelineConfig,
    generation_config_to_manifest,
    load_generation_pipeline_config,
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
from .sync_pipeline import (
    GenerationRecord,
    GenerationTiming,
    ProfiledGenerationRecord,
    collect_generation_results_sync,
    collect_generation_results_sync_profiled,
)
from .tunix_config import TunixGenerationContract
from .vanilla_backend import VanillaInferenceEngine
from .vllm_backend import VllmInferenceEngine

__all__ = [
    "EngineProfile",
    "GenerationBatch",
    "GenerationRecord",
    "GenerationResult",
    "GenerationTiming",
    "AsyncInferenceEngine",
    "AsyncCollectionConfig",
    "AsyncGenerationRecord",
    "GenerationPipelineConfig",
    "InferenceBackendError",
    "InferenceEngine",
    "ProfiledGenerationRecord",
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
    "collect_generation_results_profiled",
    "collect_generation_results_sync",
    "collect_generation_results_sync_profiled",
    "generation_config_to_manifest",
    "load_generation_pipeline_config",
]
