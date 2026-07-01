"""Inspect whether installed vLLM exposes the hooks Tunix GRPO rollout expects."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from importlib import import_module, metadata
from typing import Any, cast

EXPECTED_TUNIX_RPC_METHODS = ("delete_kv_cache", "reinitialize_kv_cache")
VLLM_WORKER_CLASSES = (
    "vllm.v1.worker.worker_base.WorkerWrapperBase",
    "vllm.v1.worker.gpu_worker.Worker",
    "vllm.worker.worker_base.WorkerWrapperBase",
    "vllm.worker.worker.Worker",
)
VLLM_ENGINE_ARG_CLASSES = (
    "vllm.engine.arg_utils.EngineArgs",
    "vllm.engine.arg_utils.AsyncEngineArgs",
)


def build_vllm_tunix_compat_report() -> dict[str, Any]:
    """Return a JSON-serializable vLLM/Tunix rollout compatibility report."""
    version = _distribution_version("vllm")
    worker_classes = [_class_report(path) for path in VLLM_WORKER_CLASSES]
    engine_args = [_engine_args_report(path) for path in VLLM_ENGINE_ARG_CLASSES]
    missing_by_class = {
        item["path"]: item["missing_tunix_rpc_methods"]
        for item in worker_classes
        if item["available"]
    }
    any_worker_has_expected_rpc = any(
        item["available"] and not item["missing_tunix_rpc_methods"] for item in worker_classes
    )
    extension_supported = any(
        item["available"] and item["supports_worker_extension"] for item in engine_args
    )
    summary_ok = version != "not-installed" and (
        any_worker_has_expected_rpc or extension_supported
    )
    recommendation = _recommendation(
        version=version,
        any_worker_has_expected_rpc=any_worker_has_expected_rpc,
        extension_supported=extension_supported,
    )
    return {
        "schema": "tunix-craftext.vllm-tunix-compat/v1",
        "vllm_version": version,
        "expected_tunix_rpc_methods": list(EXPECTED_TUNIX_RPC_METHODS),
        "worker_classes": worker_classes,
        "engine_arg_classes": engine_args,
        "summary": {
            "ok": summary_ok,
            "any_worker_has_expected_rpc": any_worker_has_expected_rpc,
            "extension_supported": extension_supported,
            "missing_rpc_methods_by_class": missing_by_class,
            "recommendation": recommendation,
        },
    }


def _distribution_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _class_report(path: str) -> dict[str, Any]:
    cls = _resolve_class(path)
    if cls is None:
        return {
            "path": path,
            "available": False,
            "methods": [],
            "missing_tunix_rpc_methods": list(EXPECTED_TUNIX_RPC_METHODS),
        }
    methods = sorted(
        name
        for name in dir(cls)
        if any(token in name.lower() for token in ("cache", "rpc", "kv"))
    )
    missing = [name for name in EXPECTED_TUNIX_RPC_METHODS if not hasattr(cls, name)]
    return {
        "path": path,
        "available": True,
        "methods": methods,
        "missing_tunix_rpc_methods": missing,
    }


def _engine_args_report(path: str) -> dict[str, Any]:
    cls = _resolve_class(path)
    if cls is None:
        return {
            "path": path,
            "available": False,
            "signature": None,
            "supports_worker_extension": False,
            "extension_fields": [],
        }
    signature = _safe_signature(cls)
    field_names = set(_annotation_keys(cls))
    if signature is not None:
        field_names.update(str(name) for name in signature.parameters)
    extension_fields = sorted(
        name
        for name in field_names
        if "extension" in name.lower() or "worker_cls" in name.lower()
    )
    return {
        "path": path,
        "available": True,
        "signature": str(signature) if signature is not None else None,
        "supports_worker_extension": bool(extension_fields),
        "extension_fields": extension_fields,
    }


def _resolve_class(path: str) -> type[Any] | None:
    module_name, _, class_name = path.rpartition(".")
    try:
        module = import_module(module_name)
    except Exception:
        return None
    cls = getattr(module, class_name, None)
    return cls if isinstance(cls, type) else None


def _safe_signature(obj: object) -> inspect.Signature | None:
    if not callable(obj):
        return None
    try:
        return inspect.signature(cast(Callable[..., Any], obj))
    except (TypeError, ValueError):
        return None


def _annotation_keys(cls: type[Any]) -> tuple[str, ...]:
    annotations = getattr(cls, "__annotations__", None)
    if isinstance(annotations, dict):
        return tuple(str(key) for key in annotations)
    return ()


def _recommendation(
    *,
    version: str,
    any_worker_has_expected_rpc: bool,
    extension_supported: bool,
) -> str:
    if any_worker_has_expected_rpc:
        return "Installed vLLM workers expose Tunix cache RPC hooks directly."
    if extension_supported:
        return (
            "Installed vLLM likely needs a worker extension class that implements "
            "delete_kv_cache/reinitialize_kv_cache for Tunix weight sync."
        )
    if version == "not-installed":
        return "Install the target vLLM stack on the remote runner and rerun this diagnostic."
    return (
        "Installed vLLM does not expose Tunix cache RPC hooks or an obvious worker "
        "extension field; use external vLLM rollout or implement a version-specific "
        "Tunix vLLM V1 adapter."
    )
