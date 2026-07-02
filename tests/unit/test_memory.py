"""Tests for process-level memory and cache lifecycle helpers."""

from __future__ import annotations

import sys
import types

import pytest

from tunix_craftext.core.garbage_collector import clean_gpu_cache
from tunix_craftext.core.memory import (
    monolith_memory_environment,
    setup_monolith_memory,
)


def test_setup_monolith_memory_sets_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """JAX/vLLM shared-process env defaults are set before framework imports."""
    for key in (
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "VLLM_WORKER_MULTIPROC_METHOD",
    ):
        monkeypatch.delenv(key, raising=False)

    config = setup_monolith_memory(jax_fraction=0.25)

    assert config.jax_fraction == 0.25
    assert monolith_memory_environment() == {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.25",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }


def test_setup_monolith_memory_preserves_user_env_without_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

    setup_monolith_memory(jax_fraction=0.2)

    assert monolith_memory_environment()["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.8"


def test_setup_monolith_memory_force_overrides_user_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

    setup_monolith_memory(jax_fraction=0.2, force=True)

    assert monolith_memory_environment()["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.2"


def test_setup_monolith_memory_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="jax_fraction"):
        setup_monolith_memory(jax_fraction=0.0)


def test_clean_gpu_cache_is_optional_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    report = clean_gpu_cache()

    assert report.torch_available is False
    assert report.cuda_available is False
    assert report.torch_cuda_cache_cleared is False


def test_clean_gpu_cache_clears_torch_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            calls.append("empty_cache")

        @staticmethod
        def ipc_collect() -> None:
            calls.append("ipc_collect")

    fake_torch = types.SimpleNamespace(cuda=FakeCuda())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    report = clean_gpu_cache()

    assert report.torch_available is True
    assert report.cuda_available is True
    assert report.torch_cuda_cache_cleared is True
    assert report.torch_ipc_collected is True
    assert calls == ["empty_cache", "ipc_collect"]
