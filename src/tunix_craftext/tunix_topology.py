"""Declarative Tunix role-to-mesh topology without a project-local GPU scheduler.

The module defines topology rules that map Tunix model roles to visible JAX
device indices, materialize meshes, and validate role assignments explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import yaml

if TYPE_CHECKING:
    from jax.sharding import Mesh

_PPO_ROLES = ("actor", "rollout", "critic", "reference")
_AGENTIC_GRPO_ROLES = ("actor", "rollout", "reference")
_SUPPORTED_ROLE_SETS = (frozenset(_PPO_ROLES), frozenset(_AGENTIC_GRPO_ROLES))


class TopologyConfigError(ValueError):
    """Raised when a versioned Tunix role topology is incomplete or invalid.

    Example:
        >>> raise TopologyConfigError("message")"""


@dataclass(frozen=True)
class TunixTopology:
    """Versioned mapping of Tunix model roles to visible JAX device indices.

    Equal tuples deliberately colocate roles. Different tuples declare a
    disaggregated workload; Tunix then owns model loading, resharding and execution.
    """

    name: str
    axis_name: str
    role_to_device_indices: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.axis_name.strip():
            raise TopologyConfigError("topology name and axis_name must be non-empty")
        if frozenset(self.role_to_device_indices) not in _SUPPORTED_ROLE_SETS:
            raise TopologyConfigError(
                f"roles must be exactly {_PPO_ROLES} or {_AGENTIC_GRPO_ROLES}"
            )
        for role, indices in self.role_to_device_indices.items():
            if not indices or any(index < 0 for index in indices):
                raise TopologyConfigError(f"{role} must contain non-negative device indices")
            if len(indices) != len(set(indices)):
                raise TopologyConfigError(f"{role} must not repeat a device index")


def load_tunix_topology(path: Path) -> TunixTopology:
    """Load one strict schema-versioned role topology YAML.

    :raises TopologyConfigError: If fields or role device lists do not meet the contract.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise TopologyConfigError(f"cannot read topology config: {path}") from error
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "name", "axis_name", "roles"}:
        raise TopologyConfigError("topology keys must be schema_version, name, axis_name, roles")
    if raw["schema_version"] != 1:
        raise TopologyConfigError("unsupported topology schema_version")
    name, axis_name, roles = raw["name"], raw["axis_name"], raw["roles"]
    if (
        not isinstance(name, str)
        or not isinstance(axis_name, str)
        or not isinstance(roles, Mapping)
    ):
        raise TopologyConfigError("topology name, axis_name and roles have invalid types")
    role_indices: dict[str, tuple[int, ...]] = {}
    for role, values in roles.items():
        if not isinstance(role, str) or not isinstance(values, Sequence) or isinstance(values, str):
            raise TopologyConfigError("each role must map to an integer list")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise TopologyConfigError("role device indices must be integers")
        role_indices[role] = tuple(values)
    return TunixTopology(name, axis_name, role_indices)


def _axis_names(axis_name: str) -> tuple[str, ...]:
    """Parse one or more comma-separated JAX mesh axes from strict config text."""
    names = tuple(part.strip() for part in axis_name.split(","))
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise TopologyConfigError("axis_name must contain unique non-empty comma-separated axes")
    return names


def role_to_meshes(
    topology: TunixTopology, devices: Sequence[jax.Device] | None = None
) -> dict[str, Mesh]:
    """Materialize declared device indices as one named JAX mesh per Tunix role.

    :param topology: TunixTopology input value
    :param devices: Sequence[jax.Device] | None input value
    :returns: dict[str, Mesh]

    Example:
        >>> result = role_to_meshes(topology, devices)
    """
    visible_devices = tuple(jax.devices() if devices is None else devices)
    axis_names = _axis_names(topology.axis_name)
    meshes: dict[str, Mesh] = {}
    for role, indices in topology.role_to_device_indices.items():
        if max(indices) >= len(visible_devices):
            raise TopologyConfigError(
                f"{role} requests device {max(indices)}, only {len(visible_devices)} are visible"
            )
        selected_devices = [visible_devices[index] for index in indices]
        if len(axis_names) > 1 and len(selected_devices) != 1:
            raise TopologyConfigError(
                "multi-axis topology currently requires exactly one device per role"
            )
        meshes[role] = jax.make_mesh(
            (len(selected_devices),) if len(axis_names) == 1 else (1,) * len(axis_names),
            axis_names,
            devices=selected_devices,
        )
    return meshes


def tunix_role_to_meshes(topology: TunixTopology) -> dict[object, Mesh]:
    """Adapt a declared topology to the official Tunix `RLCluster.role_to_mesh` mapping.

    :param topology: TunixTopology input value
    :returns: dict[object, Mesh]

    Example:
    >>> result = tunix_role_to_meshes(topology)"""
    from tunix.rl.rl_cluster import Role  # type: ignore[import-untyped]

    meshes = role_to_meshes(topology)
    role_names = {
        "actor": Role.ACTOR,
        "rollout": Role.ROLLOUT,
        "critic": Role.CRITIC,
        "reference": Role.REFERENCE,
    }
    return {role_names[name]: mesh for name, mesh in meshes.items()}
