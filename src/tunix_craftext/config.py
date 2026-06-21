"""Versioned, validated configuration contract for reproducible MVP runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from .resources import ResourceConfig


class ConfigError(ValueError):
    """Raised when a run configuration violates the public schema."""


@dataclass(frozen=True)
class RunSpec:
    """Run identity and deterministic seed."""

    name: str
    seed: int


@dataclass(frozen=True)
class EnvironmentSpec:
    """Vendor environment selection and fixed rollout dimensions."""

    implementation: str
    base_environment: str
    world_preset: str
    scenario_config: str
    instruction_index: int
    batch_size: int
    horizon: int


@dataclass(frozen=True)
class PromptSpec:
    """Prompt renderer and template selection."""

    renderer: str
    template: str


@dataclass(frozen=True)
class PolicySpec:
    """Policy implementation and invalid-action behaviour."""

    implementation: str
    invalid_action: str


@dataclass(frozen=True)
class ArtifactSpec:
    """Explicit run artifacts retained for replay and documentation."""

    save_trajectory: bool
    save_rendered_prompt: bool
    save_metrics: bool


@dataclass(frozen=True)
class MvpRunConfig:
    """Canonical schema-versioned input for one reproducible MVP run."""

    schema_version: int
    run: RunSpec
    environment: EnvironmentSpec
    prompt: PromptSpec
    policy: PolicySpec
    artifacts: ArtifactSpec
    resources: ResourceConfig = ResourceConfig()


def load_mvp_config(path: Path) -> MvpRunConfig:
    """Load and strictly validate an MVP YAML config.

    :raises ConfigError: If the YAML root, key set, types, or supported values are invalid.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read config: {path}") from error
    root = _mapping(raw, "root")
    _keys(root, {"schema_version", "run", "environment", "prompt", "policy", "artifacts", "resources"}, "root", optional={"resources"})
    schema_version = _int(root.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ConfigError(f"unsupported schema_version: {schema_version}")
    run = _run(_mapping(root.get("run"), "run"))
    environment = _environment(_mapping(root.get("environment"), "environment"))
    prompt = _prompt(_mapping(root.get("prompt"), "prompt"))
    policy = _policy(_mapping(root.get("policy"), "policy"))
    artifacts = _artifacts(_mapping(root.get("artifacts"), "artifacts"))
    resources = _resources(_mapping(root["resources"], "resources")) if "resources" in root else ResourceConfig()
    return MvpRunConfig(schema_version, run, environment, prompt, policy, artifacts, resources)

def _resources(value: Mapping[str, object]) -> ResourceConfig:
    _keys(value, {"data_axis_size", "params_placement", "optimizer_placement", "trajectory_placement"}, "resources")
    return ResourceConfig(_int(value["data_axis_size"], "resources.data_axis_size"), _string(value["params_placement"], "resources.params_placement"), _string(value["optimizer_placement"], "resources.optimizer_placement"), _string(value["trajectory_placement"], "resources.trajectory_placement"))


def _run(value: Mapping[str, object]) -> RunSpec:
    _keys(value, {"name", "seed"}, "run")
    return RunSpec(_string(value.get("name"), "run.name"), _int(value.get("seed"), "run.seed"))


def _environment(value: Mapping[str, object]) -> EnvironmentSpec:
    _keys(value, {"implementation", "base_environment", "world_preset", "scenario_config", "instruction_index", "batch_size", "horizon"}, "environment")
    implementation = _string(value.get("implementation"), "environment.implementation")
    if implementation not in {"craftext", "caged-craftext"}:
        raise ConfigError("environment.implementation must be 'craftext' or 'caged-craftext'")
    batch_size = _positive_int(value.get("batch_size"), "environment.batch_size")
    horizon = _positive_int(value.get("horizon"), "environment.horizon")
    return EnvironmentSpec(implementation, _string(value.get("base_environment"), "environment.base_environment"), _string(value.get("world_preset"), "environment.world_preset"), _string(value.get("scenario_config"), "environment.scenario_config"), _int(value.get("instruction_index"), "environment.instruction_index"), batch_size, horizon)


def _prompt(value: Mapping[str, object]) -> PromptSpec:
    _keys(value, {"renderer", "template"}, "prompt")
    renderer = _string(value.get("renderer"), "prompt.renderer")
    if renderer != "megaprompts":
        raise ConfigError("prompt.renderer must be 'megaprompts'")
    return PromptSpec(renderer, _string(value.get("template"), "prompt.template"))


def _policy(value: Mapping[str, object]) -> PolicySpec:
    _keys(value, {"implementation", "invalid_action"}, "policy")
    implementation = _string(value.get("implementation"), "policy.implementation")
    invalid_action = _string(value.get("invalid_action"), "policy.invalid_action")
    if implementation not in {"scripted", "tunix"} or invalid_action not in {"error", "fallback"}:
        raise ConfigError("policy implementation or invalid_action is unsupported")
    return PolicySpec(implementation, invalid_action)


def _artifacts(value: Mapping[str, object]) -> ArtifactSpec:
    _keys(value, {"save_trajectory", "save_rendered_prompt", "save_metrics"}, "artifacts")
    return ArtifactSpec(
        _bool(value.get("save_trajectory"), "artifacts.save_trajectory"),
        _bool(value.get("save_rendered_prompt"), "artifacts.save_rendered_prompt"),
        _bool(value.get("save_metrics"), "artifacts.save_metrics"),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{name} must be a mapping with string keys")
    return value


def _keys(value: Mapping[str, object], expected: set[str], name: str, optional: set[str] | None = None) -> None:
    optional = optional or set()
    if set(value) - optional != expected - optional or set(value) - expected:
        raise ConfigError(f"{name} keys must be exactly {sorted(expected)}, got {sorted(value)}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _int(value, name)
    if result <= 0:
        raise ConfigError(f"{name} must be positive")
    return result


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be boolean")
    return value
