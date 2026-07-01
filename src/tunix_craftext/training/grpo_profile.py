"""Reproducible Agentic GRPO run profiles and evidence manifests.

The golden training path must be inspectable before it allocates accelerator
memory.  This module keeps the profile contract independent from Tunix runtime
objects: YAML is validated, static workload specs are derived, and a compact
evidence manifest records the exact profile, git revision and dependency
versions used by a run.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..artifacts.provenance import git_revision
from ..env.config import ConfigError
from ..inference import generation_config_to_manifest, load_generation_pipeline_config
from ..tunix import AgenticGrpoWorkloadSpec


class GrpoProfileError(ConfigError):
    """Raised when an Agentic GRPO profile violates the public schema."""


@dataclass(frozen=True)
class GrpoRunSpec:
    """Identity and deterministic seed for one Agentic GRPO experiment."""

    name: str
    seed: int
    goal: str


@dataclass(frozen=True)
class GrpoModelSpec:
    """Model identity, local snapshot and licence provenance."""

    model_id: str
    snapshot: Path
    revision: str
    license: str


@dataclass(frozen=True)
class GrpoEvidenceSpec:
    """Run artifact locations written beside one reproducible GRPO run."""

    root: Path
    trajectories: Path
    metrics: Path
    checkpoints: Path
    provenance: Path


@dataclass(frozen=True)
class AgenticGrpoProfile:
    """Canonical schema-versioned profile for the Tunix Agentic GRPO pipeline."""

    schema_version: int
    run: GrpoRunSpec
    environment_config: Path
    topology_config: Path
    generation_config: Path
    model: GrpoModelSpec
    workload: AgenticGrpoWorkloadSpec
    evidence: GrpoEvidenceSpec
    vendor_manifest: Path


def load_agentic_grpo_profile(path: Path) -> AgenticGrpoProfile:
    """Load and strictly validate an Agentic GRPO YAML profile.

    :param path: Path to a profile YAML.
    :returns: Parsed profile with typed workload and artifact paths.
    :raises GrpoProfileError: If the file cannot be read or violates the schema.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise GrpoProfileError(f"cannot read Agentic GRPO profile: {path}") from error
    root = _mapping(raw, "root")
    _keys(
        root,
        {
            "schema_version",
            "run",
            "environment_config",
            "topology_config",
            "generation_config",
            "model",
            "workload",
            "evidence",
            "vendor_manifest",
        },
        "root",
    )
    schema_version = _int(root["schema_version"], "schema_version")
    if schema_version != 1:
        raise GrpoProfileError(f"unsupported schema_version: {schema_version}")
    return AgenticGrpoProfile(
        schema_version=schema_version,
        run=_run(_mapping(root["run"], "run")),
        environment_config=_path(root["environment_config"], "environment_config"),
        topology_config=_path(root["topology_config"], "topology_config"),
        generation_config=_path(root["generation_config"], "generation_config"),
        model=_model(_mapping(root["model"], "model")),
        workload=_workload(_mapping(root["workload"], "workload")),
        evidence=_evidence(_mapping(root["evidence"], "evidence")),
        vendor_manifest=_path(root["vendor_manifest"], "vendor_manifest"),
    )


def profile_sha256(path: Path) -> str:
    """Return a stable SHA256 digest for a profile file."""
    return _sha256(path)


def build_grpo_evidence_manifest(
    profile: AgenticGrpoProfile,
    *,
    profile_path: Path,
    repo_root: Path | None = None,
    packages: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable run manifest before model allocation.

    :param profile: Validated Agentic GRPO profile.
    :param profile_path: Path to the profile file used for this run.
    :param repo_root: Repository root used for git provenance; defaults to cwd.
    :param packages: Distribution names recorded in the manifest.
    :returns: Dict containing source, profile, model, workload and evidence provenance.
    """
    package_names = packages or (
        "tunix-craftext",
        "jax",
        "flax",
        "optax",
        "orbax-checkpoint",
        "google-tunix",
        "flashbax",
        "qwix",
    )
    path_root = profile_path_root(profile_path, repo_root=repo_root)
    generation_config = _resolve_profile_path(profile.generation_config, path_root)
    vendor_manifest = _resolve_profile_path(profile.vendor_manifest, path_root)
    model_snapshot = _resolve_profile_path(profile.model.snapshot, path_root)
    generation = load_generation_pipeline_config(generation_config)
    return {
        "schema_version": "tunix-craftext.agentic-grpo-evidence/v1",
        "git_revision": git_revision(repo_root),
        "profile": {
            "path": str(profile_path),
            "sha256": profile_sha256(profile_path),
            "name": profile.run.name,
            "seed": profile.run.seed,
        },
        "model": {
            "model_id": profile.model.model_id,
            "snapshot": str(profile.model.snapshot),
            "snapshot_exists": model_snapshot.exists(),
            "revision": profile.model.revision,
            "license": profile.model.license,
        },
        "inputs": {
            "environment_config": str(profile.environment_config),
            "topology_config": str(profile.topology_config),
            "generation_config": str(profile.generation_config),
            "vendor_manifest": str(profile.vendor_manifest),
            "vendor_manifest_sha256": _sha256(vendor_manifest)
            if vendor_manifest.exists()
            else "missing",
        },
        "generation": generation_config_to_manifest(
            generation,
            path=profile.generation_config,
        ),
        "workload": {
            "max_steps": profile.workload.max_steps,
            "eval_every_n_steps": profile.workload.eval_every_n_steps,
            "mini_batch_size": profile.workload.mini_batch_size,
            "train_micro_batch_size": profile.workload.train_micro_batch_size,
            "rollout_micro_batch_size": profile.workload.rollout_micro_batch_size,
            "max_prompt_length": profile.workload.max_prompt_length,
            "max_new_tokens": profile.workload.max_new_tokens,
            "kv_cache_size": profile.workload.kv_cache_size,
            "learning_rate": profile.workload.learning_rate,
            "num_generations": profile.workload.num_generations,
            "max_concurrency": profile.workload.max_concurrency,
        },
        "evidence": {
            "root": str(profile.evidence.root),
            "trajectories": str(profile.evidence.trajectories),
            "metrics": str(profile.evidence.metrics),
            "checkpoints": str(profile.evidence.checkpoints),
            "provenance": str(profile.evidence.provenance),
        },
        "packages": {name: _distribution_version(name) for name in package_names},
    }


def _run(value: Mapping[str, object]) -> GrpoRunSpec:
    _keys(value, {"name", "seed", "goal"}, "run")
    return GrpoRunSpec(
        name=_string(value["name"], "run.name"),
        seed=_int(value["seed"], "run.seed"),
        goal=_string(value["goal"], "run.goal"),
    )


def _model(value: Mapping[str, object]) -> GrpoModelSpec:
    _keys(value, {"model_id", "snapshot", "revision", "license"}, "model")
    return GrpoModelSpec(
        model_id=_string(value["model_id"], "model.model_id"),
        snapshot=_path(value["snapshot"], "model.snapshot"),
        revision=_string(value["revision"], "model.revision"),
        license=_string(value["license"], "model.license"),
    )


def _workload(value: Mapping[str, object]) -> AgenticGrpoWorkloadSpec:
    _keys(
        value,
        {
            "max_steps",
            "eval_every_n_steps",
            "mini_batch_size",
            "train_micro_batch_size",
            "rollout_micro_batch_size",
            "max_prompt_length",
            "max_new_tokens",
            "kv_cache_size",
            "learning_rate",
            "num_generations",
            "max_concurrency",
        },
        "workload",
    )
    return AgenticGrpoWorkloadSpec(
        max_steps=_positive_int(value["max_steps"], "workload.max_steps"),
        eval_every_n_steps=_positive_int(
            value["eval_every_n_steps"], "workload.eval_every_n_steps"
        ),
        mini_batch_size=_positive_int(value["mini_batch_size"], "workload.mini_batch_size"),
        train_micro_batch_size=_positive_int(
            value["train_micro_batch_size"], "workload.train_micro_batch_size"
        ),
        rollout_micro_batch_size=_positive_int(
            value["rollout_micro_batch_size"], "workload.rollout_micro_batch_size"
        ),
        max_prompt_length=_positive_int(
            value["max_prompt_length"], "workload.max_prompt_length"
        ),
        max_new_tokens=_positive_int(value["max_new_tokens"], "workload.max_new_tokens"),
        kv_cache_size=_positive_int(value["kv_cache_size"], "workload.kv_cache_size"),
        learning_rate=_positive_float(value["learning_rate"], "workload.learning_rate"),
        num_generations=_positive_int(value["num_generations"], "workload.num_generations"),
        max_concurrency=_positive_int(value["max_concurrency"], "workload.max_concurrency"),
    )


def _evidence(value: Mapping[str, object]) -> GrpoEvidenceSpec:
    _keys(value, {"root", "trajectories", "metrics", "checkpoints", "provenance"}, "evidence")
    return GrpoEvidenceSpec(
        root=_path(value["root"], "evidence.root"),
        trajectories=_path(value["trajectories"], "evidence.trajectories"),
        metrics=_path(value["metrics"], "evidence.metrics"),
        checkpoints=_path(value["checkpoints"], "evidence.checkpoints"),
        provenance=_path(value["provenance"], "evidence.provenance"),
    )


def profile_path_root(profile_path: Path, *, repo_root: Path | None = None) -> Path:
    """Return the base directory used for resolving relative profile paths.

    Agentic GRPO profiles intentionally keep paths in their original
    provenance spelling, e.g. ``configs/inference/vllm/qwen25_05b_sync.yaml``.  All
    runtime consumers must resolve those paths against a stable repository root
    instead of the process cwd because notebooks, scripts and readiness checks
    are often launched from different directories.

    :param profile_path: Path to the profile file used for this run.
    :param repo_root: Optional explicit repository root.
    :returns: Repository root when detectable, otherwise current directory.
    """
    if repo_root is not None:
        return repo_root
    for candidate in (profile_path.parent, *profile_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def resolve_profile_path(
    profile_path: Path,
    path: Path,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Resolve one profile-owned path for runtime use.

    :param profile_path: Path to the profile file used for this run.
    :param path: Raw path stored inside the profile.
    :param repo_root: Optional explicit repository root.
    :returns: Absolute or repository-root-relative runtime path.
    """
    return _resolve_profile_path(path, profile_path_root(profile_path, repo_root=repo_root))


def _resolve_profile_path(path: Path, root: Path) -> Path:
    """Resolve a path from a profile without changing its provenance spelling."""
    return path if path.is_absolute() else root / path


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise GrpoProfileError(f"{name} must be a mapping with string keys")
    return value


def _keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise GrpoProfileError(
            f"{name} keys must be exactly {sorted(expected)}, got {sorted(value)}"
        )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GrpoProfileError(f"{name} must be a non-empty string")
    return value


def _path(value: object, name: str) -> Path:
    return Path(_string(value, name))


def _int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GrpoProfileError(f"{name} must be an integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _int(value, name)
    if result <= 0:
        raise GrpoProfileError(f"{name} must be positive")
    return result


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or value <= 0:
        raise GrpoProfileError(f"{name} must be a positive float")
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
