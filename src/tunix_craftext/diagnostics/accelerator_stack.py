"""Inspect target-platform packages before expensive accelerator runs."""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import platform
import sys
import sysconfig
import tomllib
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:  # packaging is normally present through uv/pip tooling, but keep diagnostics robust.
    from packaging.requirements import Requirement
    from packaging.specifiers import InvalidSpecifier
    from packaging.version import InvalidVersion, Version
except ImportError:  # pragma: no cover - only for unusually stripped Python environments
    Requirement = None  # type: ignore[assignment]
    InvalidSpecifier = Exception  # type: ignore[assignment]
    InvalidVersion = Exception  # type: ignore[assignment]
    Version = None  # type: ignore[assignment]


DEFAULT_EXTRAS = ("tunix", "envs", "prompts", "vllm", "vllm-gpu-kernels")
DEFAULT_PROBES = (
    "jax",
    "jaxlib",
    "torch",
    "torchvision",
    "transformers",
    "vllm",
    "flashinfer",
    "flash_attn",
    "tunix",
    "craftext",
    "caged_craftext",
    "craftax",
    "megaprompt",
)

IMPORT_TO_DISTRIBUTION = {
    "flash_attn": "flash-attn",
    "flashinfer": "flashinfer-python",
    "jax": "jax",
    "jaxlib": "jaxlib",
    "megaprompt": "megaprompt",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "tunix": "google-tunix",
    "vllm": "vllm",
}

DISTRIBUTION_TO_IMPORT = {
    "caged-craftext": "caged_craftext",
    "flash-attn": "flash_attn",
    "flashinfer-python": "flashinfer",
    "google-tunix": "tunix",
    "pyyaml": "yaml",
}


@dataclass(frozen=True)
class RequirementProbe:
    """Installed-version status for one declared project requirement."""

    name: str
    requirement: str
    installed_version: str | None
    satisfies: bool | None
    import_name: str | None = None
    import_ok: bool | None = None
    import_error: str | None = None


@dataclass(frozen=True)
class ImportProbe:
    """Import status for one target-platform runtime module."""

    import_name: str
    distribution: str | None
    installed_version: str | None
    import_ok: bool
    import_error_type: str | None
    import_error: str | None
    traceback_tail: tuple[str, ...]


def build_accelerator_stack_report(
    project_root: Path,
    *,
    extras: Sequence[str] = DEFAULT_EXTRAS,
    probes: Sequence[str] = DEFAULT_PROBES,
) -> dict[str, Any]:
    """Build a JSON-safe target-platform package/import report.

    :param project_root: Repository root containing ``pyproject.toml``.
    :param extras: Optional-dependency groups that define the intended target stack.
    :param probes: Runtime modules to import and diagnose.
    :returns: JSON-safe report with platform, requirements and import status.
    """
    pyproject_path = project_root / "pyproject.toml"
    project = _load_pyproject(pyproject_path)
    requirement_strings, unknown_extras = _requirement_strings(project, extras)
    import_probe_map = {probe.import_name: probe for probe in _import_probes(probes)}

    requirements = tuple(
        _requirement_probe(requirement, import_probe_map=import_probe_map)
        for requirement in requirement_strings
    )
    import_probes = tuple(import_probe_map.values())
    runtime = _runtime_payload()
    summary = _summary(requirements, import_probes, unknown_extras, runtime)
    return {
        "schema": "tunix-craftext.accelerator-stack/v1",
        "project_root": str(project_root),
        "pyproject": str(pyproject_path),
        "extras": tuple(extras),
        "unknown_extras": tuple(unknown_extras),
        "platform": _platform_payload(),
        "runtime": runtime,
        "requirements": tuple(asdict(probe) for probe in requirements),
        "imports": tuple(asdict(probe) for probe in import_probes),
        "summary": summary,
        "recommendations": _recommendations(
            requirements=requirements,
            imports=import_probes,
            runtime=runtime,
            summary=summary,
        ),
    }


def _load_pyproject(path: Path) -> Mapping[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(f"cannot read pyproject.toml: {path}") from error


def _requirement_strings(
    project: Mapping[str, Any],
    extras: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    package = project.get("project", {})
    dependencies = list(_string_sequence(package.get("dependencies", ()), "project.dependencies"))
    optional = package.get("optional-dependencies", {})
    if not isinstance(optional, Mapping):
        optional = {}
    unknown: list[str] = []
    for extra in extras:
        values = optional.get(extra)
        if values is None:
            unknown.append(extra)
            continue
        dependencies.extend(_string_sequence(values, f"project.optional-dependencies.{extra}"))
    return tuple(dict.fromkeys(dependencies)), tuple(unknown)


def _requirement_probe(
    requirement: str,
    *,
    import_probe_map: Mapping[str, ImportProbe],
) -> RequirementProbe:
    name, specifier = _requirement_name_and_specifier(requirement)
    installed = _distribution_version(name)
    import_name = _distribution_to_import(name)
    import_probe = import_probe_map.get(import_name)
    return RequirementProbe(
        name=name,
        requirement=requirement,
        installed_version=installed,
        satisfies=_satisfies(installed, specifier),
        import_name=import_name,
        import_ok=None if import_probe is None else import_probe.import_ok,
        import_error=None if import_probe is None else import_probe.import_error,
    )


def _import_probes(import_names: Sequence[str]) -> tuple[ImportProbe, ...]:
    probes: list[ImportProbe] = []
    for import_name in tuple(dict.fromkeys(import_names)):
        distribution = IMPORT_TO_DISTRIBUTION.get(import_name, import_name.replace("_", "-"))
        installed = _distribution_version(distribution)
        try:
            module = importlib.import_module(import_name)
        except Exception as error:  # noqa: BLE001 - diagnostics must capture third-party failures
            probes.append(
                ImportProbe(
                    import_name=import_name,
                    distribution=distribution,
                    installed_version=installed,
                    import_ok=False,
                    import_error_type=type(error).__name__,
                    import_error=str(error),
                    traceback_tail=tuple(traceback.format_exception(error)[-8:]),
                )
            )
            continue
        probes.append(
            ImportProbe(
                import_name=import_name,
                distribution=distribution,
                installed_version=installed or _module_version(module),
                import_ok=True,
                import_error_type=None,
                import_error=None,
                traceback_tail=(),
            )
        )
    return tuple(probes)


def _platform_payload() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "sys_platform": sys.platform,
        "platform_tag": sysconfig.get_platform(),
    }


def _runtime_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        import jax

        payload["jax"] = {
            "backend": jax.default_backend(),
            "devices": tuple(str(device) for device in jax.devices()),
        }
    except Exception as error:  # noqa: BLE001 - diagnostics only
        payload["jax"] = {"error": str(error), "error_type": type(error).__name__}
    try:
        import torch

        payload["torch"] = {
            "version": getattr(torch, "__version__", None),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": tuple(
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ),
        }
    except Exception as error:  # noqa: BLE001 - diagnostics only
        payload["torch"] = {"error": str(error), "error_type": type(error).__name__}
    return payload


def _summary(
    requirements: Sequence[RequirementProbe],
    imports: Sequence[ImportProbe],
    unknown_extras: Sequence[str],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    missing = tuple(probe.name for probe in requirements if probe.installed_version is None)
    unsatisfied = tuple(
        probe.name
        for probe in requirements
        if probe.installed_version is not None and probe.satisfies is False
    )
    broken_imports = tuple(probe.import_name for probe in imports if not probe.import_ok)
    runtime_errors = tuple(
        name
        for name, payload in runtime.items()
        if isinstance(payload, Mapping) and "error" in payload
    )
    return {
        "ok": not missing
        and not unsatisfied
        and not broken_imports
        and not unknown_extras
        and not runtime_errors,
        "missing_requirements": missing,
        "unsatisfied_requirements": unsatisfied,
        "broken_imports": broken_imports,
        "runtime_errors": runtime_errors,
        "unknown_extras": tuple(unknown_extras),
    }


def _recommendations(
    *,
    requirements: Sequence[RequirementProbe],
    imports: Sequence[ImportProbe],
    runtime: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> tuple[dict[str, object], ...]:
    recommendations: list[dict[str, object]] = []
    import_by_name = {probe.import_name: probe for probe in imports}
    runtime_jax = runtime.get("jax")
    runtime_torch = runtime.get("torch")
    if isinstance(runtime_jax, Mapping):
        jax_error = str(runtime_jax.get("error", ""))
        if (
            "Unable to initialize backend 'cuda'" in jax_error
            or "no supported devices" in jax_error
        ):
            torch_cuda = (
                isinstance(runtime_torch, Mapping) and runtime_torch.get("cuda_available") is True
            )
            recommendations.append(
                {
                    "id": "jax-cuda-plugin-unusable",
                    "severity": "fail",
                    "title": "JAX CUDA backend is installed but cannot initialize a CUDA device",
                    "details": (
                        "Torch CUDA is visible, but JAX uses its own jaxlib/PJRT CUDA plugin. "
                        "A working torch CUDA stack does not prove JAX CUDA is compatible."
                        if torch_cuda
                        else "JAX could not initialize CUDA devices on this runner."
                    ),
                    "actions": (
                        "For CPU-only unit tests: run `JAX_PLATFORMS=cpu make test` or "
                        "`JAX_PLATFORMS=cpu uv run python -m pytest tests/unit`.",
                        "For GPU rollout/training: inspect "
                        "`uv pip list | grep -E 'jax|cuda|pjrt'`.",
                        "Reinstall the JAX CUDA wheel/plugin that matches the runner "
                        "driver/CUDA stack; do not assume the PyTorch CUDA wheel fixes JAX.",
                    ),
                }
            )
    torchvision = import_by_name.get("torchvision")
    if torchvision is not None and "torchvision::nms" in str(torchvision.import_error):
        recommendations.append(
            {
                "id": "torchvision-nms-mismatch",
                "severity": "fail",
                "title": "torchvision is ABI-incompatible with the installed torch build",
                "details": (
                    "vLLM imports Transformers, Transformers detects torchvision, and torchvision "
                    "fails while registering `torchvision::nms`."
                ),
                "actions": (
                    "For text-only Qwen/vLLM: remove broken torchvision from this environment.",
                    "If torchvision is required, reinstall a matching torch/torchvision "
                    "CUDA wheel pair.",
                    "Verify with `uv run python -c \"import torch, torchvision; "
                    "print(torch.__version__, torch.version.cuda, torchvision.__version__)\"`.",
                ),
            }
        )
    missing = tuple(summary.get("missing_requirements", ()))
    if missing:
        recommendations.append(
            {
                "id": "missing-requirements",
                "severity": "fail",
                "title": "Required packages from selected extras are missing",
                "details": f"Missing: {', '.join(str(item) for item in missing)}",
                "actions": (
                    "Run `uv sync --extra tunix --extra envs --extra prompts --extra vllm`.",
                    "Add `--extra vllm-gpu-kernels` when the target runner should "
                    "install flashinfer.",
                ),
            }
        )
    return tuple(recommendations)


def _string_sequence(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(item for item in value if isinstance(item, str))
    if len(result) != len(tuple(value)):
        raise TypeError(f"{name} must contain only strings")
    return result


def _requirement_name_and_specifier(requirement: str) -> tuple[str, str]:
    if Requirement is None:
        return requirement.split(";", 1)[0].split("[", 1)[0].split("=", 1)[0].strip(), ""
    parsed = Requirement(requirement)
    return parsed.name, str(parsed.specifier)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _distribution_to_import(name: str) -> str:
    normalized = name.lower().replace("_", "-")
    return DISTRIBUTION_TO_IMPORT.get(normalized, normalized.replace("-", "_"))


def _satisfies(installed: str | None, specifier: str) -> bool | None:
    if installed is None:
        return False
    if not specifier or Requirement is None or Version is None:
        return None
    try:
        return Version(installed) in Requirement(f"pkg{specifier}").specifier
    except (InvalidSpecifier, InvalidVersion):
        return None


def _module_version(module: object) -> str | None:
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None
