"""Strict model profile contracts for LLM actor backbones.

Model profiles describe an intended backbone without loading weights.  They are
used by CLI/profile validation, LLM actor construction and evidence manifests so
the training stack can distinguish a small Gemma smoke backbone from the Qwen
Agentic GRPO production profile before accelerator allocation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..env.config import ConfigError


class ModelProfileError(ConfigError):
    """Raised when a model profile violates the public schema."""


@dataclass(frozen=True)
class ResourceProfile:
    """Static resource declaration for one model profile."""

    accelerator: str
    mesh_axis_size: int


@dataclass(frozen=True)
class ModelProfile:
    """Versioned model/backbone declaration independent from weight loading."""

    schema_version: int
    name: str
    architecture: str
    model_id: str
    source: str
    tokenizer_source: str
    purpose: str
    weights_downloaded: bool
    license_acknowledged: bool
    resource_profile: ResourceProfile

    @property
    def is_llm_actor_candidate(self) -> bool:
        """Whether this profile is allowed to back an LLM actor."""
        return self.architecture in {"gemma3", "qwen2"} and self.tokenizer_source == "model"


def load_model_profile(path: Path) -> ModelProfile:
    """Load and strictly validate a model profile YAML.

    :param path: Path to a model profile file.
    :returns: Parsed `ModelProfile`.
    :raises ModelProfileError: If the YAML is missing, malformed or unsupported.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ModelProfileError(f"cannot read model profile: {path}") from error
    root = _mapping(raw, "root")
    _keys(
        root,
        {
            "schema_version",
            "name",
            "architecture",
            "model_id",
            "source",
            "tokenizer_source",
            "purpose",
            "weights_downloaded",
            "license_acknowledged",
            "resource_profile",
        },
        "root",
    )
    schema_version = _int(root["schema_version"], "schema_version")
    if schema_version != 1:
        raise ModelProfileError(f"unsupported schema_version: {schema_version}")
    architecture = _string(root["architecture"], "architecture")
    if architecture not in {"gemma3", "qwen2"}:
        raise ModelProfileError("architecture must be 'gemma3' or 'qwen2'")
    return ModelProfile(
        schema_version=schema_version,
        name=_string(root["name"], "name"),
        architecture=architecture,
        model_id=_string(root["model_id"], "model_id"),
        source=_string(root["source"], "source"),
        tokenizer_source=_string(root["tokenizer_source"], "tokenizer_source"),
        purpose=_string(root["purpose"], "purpose"),
        weights_downloaded=_bool(root["weights_downloaded"], "weights_downloaded"),
        license_acknowledged=_bool(root["license_acknowledged"], "license_acknowledged"),
        resource_profile=_resource_profile(
            _mapping(root["resource_profile"], "resource_profile")
        ),
    )


def _resource_profile(value: Mapping[str, object]) -> ResourceProfile:
    _keys(value, {"accelerator", "mesh_axis_size"}, "resource_profile")
    return ResourceProfile(
        accelerator=_string(value["accelerator"], "resource_profile.accelerator"),
        mesh_axis_size=_positive_int(value["mesh_axis_size"], "resource_profile.mesh_axis_size"),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ModelProfileError(f"{name} must be a mapping with string keys")
    return value


def _keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ModelProfileError(
            f"{name} keys must be exactly {sorted(expected)}, got {sorted(value)}"
        )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelProfileError(f"{name} must be a non-empty string")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelProfileError(f"{name} must be an integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _int(value, name)
    if result <= 0:
        raise ModelProfileError(f"{name} must be positive")
    return result


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ModelProfileError(f"{name} must be boolean")
    return value
