"""Optional vLLM rollout-generation adapter.

This module deliberately imports vLLM lazily.  CPU/unit tests and documentation
must be able to load the project without installing a Linux/GPU inference stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ..models.llm import LlmResponse
from .contracts import EngineProfile, GenerationBatch, GenerationResult, InferenceBackendError


@dataclass
class VllmInferenceEngine:
    """Synchronous vLLM engine implementing the project inference contract."""

    profile: EngineProfile
    _llm: object | None = None

    @classmethod
    def from_profile(cls, profile: EngineProfile) -> VllmInferenceEngine:
        """Create a vLLM engine from an explicit profile.

        :raises InferenceBackendError: If vLLM is not installed or the profile is invalid.
        """
        if profile.backend != "vllm-offload":
            raise InferenceBackendError("VllmInferenceEngine requires backend='vllm-offload'")
        _validate_model_snapshot(profile.model)
        _configure_vllm_environment(profile)
        try:
            from vllm import LLM  # type: ignore[import-not-found]
        except ImportError as error:
            raise InferenceBackendError(
                "vLLM is not installed. Install the optional inference stack on the "
                "target Linux/GPU runner before using backend='vllm-offload'."
            ) from error
        except RuntimeError as error:
            _raise_vllm_runtime_import_error(error)
        kwargs: dict[str, object] = {
            "model": profile.model,
            "tensor_parallel_size": profile.tensor_parallel_size,
        }
        if profile.max_model_len is not None:
            kwargs["max_model_len"] = profile.max_model_len
        if profile.dtype is not None:
            kwargs["dtype"] = profile.dtype
        kwargs.update(_vllm_kwargs_from_profile_metadata(profile))
        try:
            llm = LLM(**kwargs)
        except RuntimeError as error:
            _raise_vllm_engine_start_error(error, profile)
        return cls(profile=profile, _llm=llm)

    def generate(self, batch: GenerationBatch) -> GenerationResult:
        """Generate an ordered batch with vLLM and normalize it to `LlmResponse`."""
        if self._llm is None:
            raise InferenceBackendError("vLLM engine is not initialized; call from_profile()")
        try:
            from vllm import SamplingParams  # type: ignore[import-not-found]
        except ImportError as error:
            raise InferenceBackendError("vLLM SamplingParams is unavailable") from error
        except RuntimeError as error:
            _raise_vllm_runtime_import_error(error)
        stop = tuple(
            dict.fromkeys(stop for request in batch.requests for stop in request.stop_sequences)
        )
        sampling_params = SamplingParams(
            max_tokens=batch.max_new_tokens,
            temperature=batch.temperature,
            stop=list(stop),
            logprobs=1,
        )
        prompts = [request.prompt.text for request in batch.requests]
        started = perf_counter()
        outputs = self._llm.generate(prompts, sampling_params=sampling_params)  # type: ignore[attr-defined]
        latency_ms = (perf_counter() - started) * 1000.0
        if len(outputs) != len(batch.requests):
            raise InferenceBackendError("vLLM changed batch cardinality")
        responses = tuple(
            _response_from_vllm_output(output, self.profile, latency_ms=latency_ms)
            for output in outputs
        )
        return GenerationResult(
            self.profile,
            responses,
            group_id=batch.group_id,
            policy_version=batch.policy_version,
        )

    async def generate_async(self, batch: GenerationBatch) -> GenerationResult:
        """Reject fake async execution for the synchronous vLLM engine."""
        raise InferenceBackendError(
            "VllmInferenceEngine is synchronous. Use AsyncVllmInferenceEngine for "
            "native vLLM async generation instead of wrapping vLLM.LLM in a worker thread."
        )


def _response_from_vllm_output(
    output: object, profile: EngineProfile, *, latency_ms: float
) -> LlmResponse:
    choices = getattr(output, "outputs", None)
    if not choices:
        raise InferenceBackendError("vLLM output did not contain generated choices")
    choice = choices[0]
    raw_text = str(getattr(choice, "text", ""))
    token_ids = tuple(int(token) for token in getattr(choice, "token_ids", ()) or ())
    token_logprobs = _extract_logprobs(getattr(choice, "logprobs", None))
    prompt_token_ids = tuple(int(token) for token in getattr(output, "prompt_token_ids", ()) or ())
    return LlmResponse(
        raw_text=raw_text,
        backend=profile.backend,
        model=profile.model,
        latency_ms=latency_ms,
        token_logprobs=token_logprobs,
        token_ids=token_ids or None,
        prompt_token_ids=prompt_token_ids or None,
    )


def _extract_logprobs(raw_logprobs: object) -> tuple[float, ...] | None:
    if raw_logprobs is None:
        return None
    values: list[float] = []
    for item in raw_logprobs if isinstance(raw_logprobs, list) else []:
        if isinstance(item, dict) and item:
            first = next(iter(item.values()))
            values.append(float(getattr(first, "logprob", first)))
        elif isinstance(item, (int, float)):
            values.append(float(item))
        else:  # pragma: no cover - defensive for third-party objects
            logprob = getattr(item, "logprob", None)
            if logprob is not None:
                values.append(float(logprob))
    return tuple(values) if values else None


def _raise_vllm_runtime_import_error(error: RuntimeError) -> None:
    """Convert common binary-stack import failures into actionable project errors."""
    message = str(error)
    if "torchvision::nms" in message:
        raise InferenceBackendError(
            "vLLM import reached transformers/torchvision, but torchvision is not "
            "ABI-compatible with the installed torch build: missing operator "
            "`torchvision::nms`. For text-only Qwen rollout either remove torchvision "
            "from the vLLM environment, or reinstall a matching torch/torchvision CUDA "
            "wheel pair for the target runner. Then verify with "
            "`uv run python -c \"import torch, torchvision; print(torch.__version__, "
            "torchvision.__version__)\"` before opening this notebook."
        ) from error
    raise InferenceBackendError(
        "vLLM is installed but failed during import. Check the Linux/GPU binary stack "
        "(torch, torchvision, CUDA wheels, flashinfer/flash-attn) before creating "
        "VllmInferenceEngine."
    ) from error


def _raise_vllm_engine_start_error(error: RuntimeError, profile: EngineProfile) -> None:
    """Convert vLLM subprocess startup failures into actionable project errors."""
    message = str(error)
    if "Engine core initialization failed" in message:
        raise InferenceBackendError(
            "vLLM EngineCore failed while creating the rollout engine. The real root "
            "cause is usually printed by vLLM just before this traceback in the notebook "
            "or server stderr. Run `make accelerator-stack` first and verify: torch CUDA "
            "is available, JAX CUDA is either healthy or forced to CPU for tests, "
            "torchvision is not broken for this torch build, the model snapshot exists, "
            "and the profile fits GPU memory. Profile: "
            f"name={profile.name!r}, model={profile.model!r}, "
            f"dtype={profile.dtype!r}, max_model_len={profile.max_model_len!r}, "
            f"tensor_parallel_size={profile.tensor_parallel_size}, "
            f"metadata={dict(profile.metadata)!r}."
        ) from error
    _raise_vllm_runtime_import_error(error)


def _validate_model_snapshot(model: str) -> None:
    """Fail early when a profile points at a missing local model snapshot."""
    path = Path(model).expanduser()
    if not (path.is_absolute() or model.startswith((".", "~"))):
        return
    if not path.exists():
        raise InferenceBackendError(
            "Local vLLM model snapshot is missing: "
            f"{path}. Download it before creating VllmInferenceEngine, or use a "
            "Hugging Face model id in the generation profile."
        )


def _configure_vllm_environment(profile: EngineProfile) -> None:
    """Configure vLLM/JAX-safe process environment before vLLM imports."""
    _configure_vllm_process_start_method(profile)
    _configure_vllm_use_v1(profile)


def _configure_vllm_process_start_method(profile: EngineProfile) -> None:
    """Prefer a JAX-safe vLLM worker start method before vLLM imports.

    Python's default ``fork`` start method is unsafe after JAX has started its
    multithreaded runtime.  vLLM reads ``VLLM_WORKER_MULTIPROC_METHOD`` during
    startup, so the project sets a conservative default for in-process notebook
    rollouts while still allowing users to override it explicitly in the shell.
    """
    raw_method = profile.metadata.get("multiprocessing_method", "spawn")
    if raw_method is None:
        return
    method = str(raw_method).strip()
    if not method:
        return
    if method not in {"spawn", "forkserver", "fork"}:
        raise InferenceBackendError(
            "engine.metadata.multiprocessing_method must be one of "
            "'spawn', 'forkserver' or 'fork'"
        )
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", method)


def _configure_vllm_use_v1(profile: EngineProfile) -> None:
    """Optionally force vLLM V0/V1 engine selection from profile metadata."""
    if "vllm_use_v1" not in profile.metadata:
        return
    raw = profile.metadata["vllm_use_v1"]
    if not isinstance(raw, bool):
        raise InferenceBackendError("engine.metadata.vllm_use_v1 must be boolean")
    os.environ.setdefault("VLLM_USE_V1", "1" if raw else "0")


def _vllm_kwargs_from_profile_metadata(profile: EngineProfile) -> dict[str, object]:
    """Return explicit vLLM constructor kwargs encoded in engine metadata.

    The Tunix rollout config has its own `vllm_hbm_utilization` knobs.  The direct
    notebook path uses `VllmInferenceEngine`, so production-critical vLLM knobs must be
    present on the engine profile as well; otherwise vLLM falls back to its aggressive
    default GPU memory utilization.
    """
    metadata = profile.metadata
    kwargs: dict[str, object] = {}
    if "gpu_memory_utilization" in metadata:
        value = _metadata_float(metadata["gpu_memory_utilization"], "gpu_memory_utilization")
        if not 0.0 < value <= 1.0:
            raise InferenceBackendError(
                "engine.metadata.gpu_memory_utilization must be in (0, 1]"
            )
        kwargs["gpu_memory_utilization"] = value
    for key in ("max_num_batched_tokens", "max_num_seqs"):
        if key in metadata:
            value = _metadata_int(metadata[key], key)
            if value <= 0:
                raise InferenceBackendError(f"engine.metadata.{key} must be positive")
            kwargs[key] = value
    for key in ("enforce_eager", "disable_log_stats"):
        if key in metadata:
            kwargs[key] = _metadata_bool(metadata[key], key)
    return kwargs


def _metadata_float(value: object, name: str) -> float:
    """Parse a numeric metadata field without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceBackendError(f"engine.metadata.{name} must be numeric")
    return float(value)


def _metadata_int(value: object, name: str) -> int:
    """Parse an integer metadata field without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InferenceBackendError(f"engine.metadata.{name} must be an integer")
    return value


def _metadata_bool(value: object, name: str) -> bool:
    """Parse a boolean metadata field."""
    if not isinstance(value, bool):
        raise InferenceBackendError(f"engine.metadata.{name} must be boolean")
    return value
