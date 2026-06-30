"""Target-platform accelerator stack diagnostics."""

from __future__ import annotations

import types
from pathlib import Path

from tunix_craftext.diagnostics import accelerator_stack


def test_accelerator_stack_report_marks_broken_torchvision_import(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
dependencies = ["numpy>=1"]

[project.optional-dependencies]
vllm = ["vllm", "torchvision"]
""",
        encoding="utf-8",
    )

    def fake_version(name: str) -> str:
        if name in {"numpy", "vllm", "torchvision"}:
            return "1.0.0"
        raise accelerator_stack.importlib_metadata.PackageNotFoundError(name)

    def fake_import(name: str) -> object:
        if name == "torchvision":
            raise RuntimeError("operator torchvision::nms does not exist")
        module = types.SimpleNamespace(__version__="1.0.0")
        return module

    monkeypatch.setattr(accelerator_stack.importlib_metadata, "version", fake_version)
    monkeypatch.setattr(accelerator_stack.importlib, "import_module", fake_import)
    monkeypatch.setattr(accelerator_stack, "_runtime_payload", lambda: {})

    report = accelerator_stack.build_accelerator_stack_report(
        tmp_path,
        extras=("vllm",),
        probes=("torchvision",),
    )

    assert report["summary"]["ok"] is False
    assert report["summary"]["broken_imports"] == ("torchvision",)
    [probe] = report["imports"]
    assert probe["import_error_type"] == "RuntimeError"
    assert "torchvision::nms" in probe["import_error"]


def test_accelerator_stack_report_records_unknown_extra(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
dependencies = []

[project.optional-dependencies]
vllm = ["vllm"]
""",
        encoding="utf-8",
    )

    report = accelerator_stack.build_accelerator_stack_report(
        tmp_path,
        extras=("missing-extra",),
        probes=(),
    )

    assert report["summary"]["ok"] is False
    assert report["unknown_extras"] == ("missing-extra",)
