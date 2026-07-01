"""Diagnostics for vLLM/Tunix GRPO weight-sync compatibility."""

from __future__ import annotations

import sys
import types

from tunix_craftext.diagnostics.vllm_tunix_compat import (
    build_vllm_tunix_compat_report,
)


def test_vllm_tunix_compat_report_handles_missing_vllm() -> None:
    report = build_vllm_tunix_compat_report()

    assert report["schema"] == "tunix-craftext.vllm-tunix-compat/v1"
    assert "summary" in report
    assert "recommendation" in report["summary"]


def test_vllm_tunix_compat_detects_worker_extension_support(
    monkeypatch,
) -> None:
    worker_base = types.ModuleType("vllm.v1.worker.worker_base")
    gpu_worker = types.ModuleType("vllm.v1.worker.gpu_worker")
    arg_utils = types.ModuleType("vllm.engine.arg_utils")

    class WorkerWrapperBase:
        def collective_rpc(self) -> None:
            return None

    class Worker:
        pass

    class EngineArgs:
        __annotations__ = {"worker_extension_cls": str}

    worker_base.WorkerWrapperBase = WorkerWrapperBase
    gpu_worker.Worker = Worker
    arg_utils.EngineArgs = EngineArgs
    arg_utils.AsyncEngineArgs = EngineArgs

    for name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.engine",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.worker_base", worker_base)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", gpu_worker)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils)

    report = build_vllm_tunix_compat_report()

    assert report["summary"]["extension_supported"] is True
    assert report["summary"]["any_worker_has_expected_rpc"] is False
    assert "worker extension class" in report["summary"]["recommendation"]
