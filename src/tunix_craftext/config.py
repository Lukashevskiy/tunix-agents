"""Versioned, validated configuration contract for reproducible MVP runs.

This module defines the schema for MVP run configuration, validates YAML files
strictly, and exposes lightweight dataclasses for environment, prompt, policy,
artifact, and resource configuration to support deterministic reproducibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from .resources import ResourceConfig


class ConfigError(ValueError):
    """Raised when a run configuration violates the public schema.

    Example:
        >>> raise ConfigError("message")"""


@dataclass(frozen=True)
class RunSpec:
    """Run identity and deterministic seed.

    :ivar name: str
    :ivar seed: int

    Example:
        >>> obj = RunSpec(name=..., seed=...)"""

    name: str
    seed: int


@dataclass(frozen=True)
class EnvironmentSpec:
    """Vendor environment selection and fixed rollout dimensions.

    :ivar implementation: str
    :ivar base_environment: str
    :ivar world_preset: str
    :ivar scenario_config: str
    :ivar instruction_index: int
    :ivar batch_size: int
    :ivar horizon: int

    Example:
        >>> obj = EnvironmentSpec(implementation=..., base_environment=..., world_preset=...)"""

    implementation: str
    base_environment: str
    world_preset: str
    scenario_config: str
    instruction_index: int
    batch_size: int
    horizon: int


@dataclass(frozen=True)
class PromptSpec:
    """Prompt renderer and template selection.

    :ivar renderer: str
    :ivar template: str

    Example:
        >>> obj = PromptSpec(renderer=..., template=...)"""

    renderer: str
    template: str


@dataclass(frozen=True)
class PolicySpec:
    """Policy implementation and invalid-action behaviour.

    :ivar implementation: str
    :ivar invalid_action: str

    Example:
        >>> obj = PolicySpec(implementation=..., invalid_action=...)"""

    implementation: str
    invalid_action: str


@dataclass(frozen=True)
class ArtifactSpec:
    """Explicit run artifacts retained for replay and documentation.

    :ivar save_trajectory: bool
    :ivar save_rendered_prompt: bool
    :ivar save_metrics: bool

    Example:
        >>> obj = ArtifactSpec(save_trajectory=..., save_rendered_prompt=..., save_metrics=...)"""

    save_trajectory: bool
    save_rendered_prompt: bool
    save_metrics: bool


@dataclass(frozen=True)
class MvpRunConfig:
    """Canonical schema-versioned input for one reproducible MVP run.

    :ivar schema_version: int
    :ivar run: RunSpec
    :ivar environment: EnvironmentSpec
    :ivar prompt: PromptSpec
    :ivar policy: PolicySpec
    :ivar artifacts: ArtifactSpec
    :ivar resources: ResourceConfig

    Example:
        >>> obj = MvpRunConfig(schema_version=..., run=..., environment=...)"""

    schema_version: int
    run: RunSpec
    environment: EnvironmentSpec
    prompt: PromptSpec
    policy: PolicySpec
    artifacts: ArtifactSpec
    resources: ResourceConfig = ResourceConfig()


def load_mvp_config(path: Path) -> MvpRunConfig:
    """Load and strictly validate an MVP YAML config.

    :param path: Path to a YAML configuration file.
    :returns: Parsed and validated `MvpRunConfig` instance.
    :raises ConfigError: If the YAML root, key set, types, or supported values are invalid.

    Example:
        >>> config = load_mvp_config(Path('configs/my_run.yaml'))
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read config: {path}") from error
    root = _mapping(raw, "root")
    _keys(
        root,
        {"schema_version", "run", "environment", "prompt", "policy", "artifacts", "resources"},
        "root",
        optional={"resources"},
    )
    schema_version = _int(root.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ConfigError(f"unsupported schema_version: {schema_version}")
    run = _run(_mapping(root.get("run"), "run"))
    environment = _environment(_mapping(root.get("environment"), "environment"))
    prompt = _prompt(_mapping(root.get("prompt"), "prompt"))
    policy = _policy(_mapping(root.get("policy"), "policy"))
    artifacts = _artifacts(_mapping(root.get("artifacts"), "artifacts"))
    resources = (
        _resources(_mapping(root["resources"], "resources"))
        if "resources" in root
        else ResourceConfig()
    )
    return MvpRunConfig(schema_version, run, environment, prompt, policy, artifacts, resources)


def _resources(value: Mapping[str, object]) -> ResourceConfig:
    """Validate and build a `ResourceConfig` from a mapping.

    :param value: Mapping representing the `resources` section.
    :returns: A `ResourceConfig` instance.
    :raises ConfigError: If required keys are missing or types are invalid.
    """
    _keys(
        value,
        {"data_axis_size", "params_placement", "optimizer_placement", "trajectory_placement"},
        "resources",
    )
    return ResourceConfig(
        _int(value["data_axis_size"], "resources.data_axis_size"),
        _string(value["params_placement"], "resources.params_placement"),
        _string(value["optimizer_placement"], "resources.optimizer_placement"),
        _string(value["trajectory_placement"], "resources.trajectory_placement"),
    )


def _run(value: Mapping[str, object]) -> RunSpec:
    """Validate and return a `RunSpec` from the `run` mapping.

    :param value: Mapping for the `run` section.
    :returns: `RunSpec` with `name` and `seed`.
    :raises ConfigError: If required keys or types are invalid.
    """
    _keys(value, {"name", "seed"}, "run")
    return RunSpec(_string(value.get("name"), "run.name"), _int(value.get("seed"), "run.seed"))


def _environment(value: Mapping[str, object]) -> EnvironmentSpec:
    """Validate and build an `EnvironmentSpec` from the `environment` mapping.

    :param value: Mapping for the `environment` section.
    :returns: `EnvironmentSpec` with implementation and dimensions.
    :raises ConfigError: If keys, types, or supported values are invalid.
    """
    _keys(
        value,
        {
            "implementation",
            "base_environment",
            "world_preset",
            "scenario_config",
            "instruction_index",
            "batch_size",
            "horizon",
        },
        "environment",
    )
    implementation = _string(value.get("implementation"), "environment.implementation")
    if implementation not in {"craftext", "caged-craftext"}:
        raise ConfigError("environment.implementation must be 'craftext' or 'caged-craftext'")
    batch_size = _positive_int(value.get("batch_size"), "environment.batch_size")
    horizon = _positive_int(value.get("horizon"), "environment.horizon")
    return EnvironmentSpec(
        implementation,
        _string(value.get("base_environment"), "environment.base_environment"),
        _string(value.get("world_preset"), "environment.world_preset"),
        _string(value.get("scenario_config"), "environment.scenario_config"),
        _int(value.get("instruction_index"), "environment.instruction_index"),
        batch_size,
        horizon,
    )


def _prompt(value: Mapping[str, object]) -> PromptSpec:
    """Validate and return a `PromptSpec` from the `prompt` mapping.

    :param value: Mapping for the `prompt` section.
    :returns: `PromptSpec` with renderer and template.
    :raises ConfigError: If unsupported renderer is selected or types are invalid.
    """
    _keys(value, {"renderer", "template"}, "prompt")
    renderer = _string(value.get("renderer"), "prompt.renderer")
    if renderer != "megaprompts":
        raise ConfigError("prompt.renderer must be 'megaprompts'")
    return PromptSpec(renderer, _string(value.get("template"), "prompt.template"))


def _policy(value: Mapping[str, object]) -> PolicySpec:
    """Validate and return a `PolicySpec` from the `policy` mapping.

    :param value: Mapping for the `policy` section.
    :returns: `PolicySpec` describing implementation and invalid-action handling.
    :raises ConfigError: If unsupported implementation or invalid_action is provided.
    """
    _keys(value, {"implementation", "invalid_action"}, "policy")
    implementation = _string(value.get("implementation"), "policy.implementation")
    invalid_action = _string(value.get("invalid_action"), "policy.invalid_action")
    if implementation not in {"scripted", "tunix"} or invalid_action not in {"error", "fallback"}:
        raise ConfigError("policy implementation or invalid_action is unsupported")
    return PolicySpec(implementation, invalid_action)


def _artifacts(value: Mapping[str, object]) -> ArtifactSpec:
    """Validate and return an `ArtifactSpec` from the `artifacts` mapping.

    :param value: Mapping for the `artifacts` section.
    :returns: `ArtifactSpec` with boolean artifact flags.
    :raises ConfigError: If any artifact flag is missing or not boolean.
    """
    _keys(value, {"save_trajectory", "save_rendered_prompt", "save_metrics"}, "artifacts")
    return ArtifactSpec(
        _bool(value.get("save_trajectory"), "artifacts.save_trajectory"),
        _bool(value.get("save_rendered_prompt"), "artifacts.save_rendered_prompt"),
        _bool(value.get("save_metrics"), "artifacts.save_metrics"),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    """Ensure a value is a mapping with string keys.

    :param value: Object to validate as mapping.
    :param name: Logical name for error messages.
    :returns: The validated mapping.
    :raises ConfigError: If `value` is not a mapping with string keys.
    """
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{name} must be a mapping with string keys")
    return value


def _keys(
    value: Mapping[str, object], expected: set[str], name: str, optional: set[str] | None = None
) -> None:
    """Assert that a mapping has exactly the expected keys (with optional exceptions).

    :param value: Mapping to validate.
    :param expected: Set of expected keys.
    :param name: Human-friendly section name for errors.
    :param optional: Keys that may be omitted.
    :raises ConfigError: If keys differ from the expected set.
    """
    optional = optional or set()
    if set(value) - optional != expected - optional or set(value) - expected:
        raise ConfigError(f"{name} keys must be exactly {sorted(expected)}, got {sorted(value)}")


def _string(value: object, name: str) -> str:
    """Validate that a value is a non-empty string.

    :param value: Candidate value.
    :param name: Field name used for error messaging.
    :returns: The validated string.
    :raises ConfigError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _int(value: object, name: str) -> int:
    """Validate that a value is an integer (but not boolean).

    :param value: Candidate value.
    :param name: Field name used for error messaging.
    :returns: The validated integer.
    :raises ConfigError: If the value is not an integer.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    return value


def _positive_int(value: object, name: str) -> int:
    """Validate that a value is a positive integer.

    :param value: Candidate value.
    :param name: Field name used for error messaging.
    :returns: The validated positive integer.
    :raises ConfigError: If the value is not a positive integer.
    """
    result = _int(value, name)
    if result <= 0:
        raise ConfigError(f"{name} must be positive")
    return result


def _bool(value: object, name: str) -> bool:
    """Validate that a value is boolean.

    :param value: Candidate value.
    :param name: Field name used for error messaging.
    :returns: The validated boolean.
    :raises ConfigError: If the value is not a boolean.
    """
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be boolean")
    return value
