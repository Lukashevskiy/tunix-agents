"""Declarative Tunix role-to-mesh topology without a project-local GPU scheduler."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import yaml

if TYPE_CHECKING:
    from jax.sharding import Mesh

_REQUIRED_ROLES = ("actor", "rollout", "critic", "reference")


class TopologyConfigError(ValueError):
    """Raised when a versioned Tunix role topology is incomplete or invalid."""


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
        if tuple(self.role_to_device_indices) != _REQUIRED_ROLES:
            raise TopologyConfigError(f"roles must be exactly {_REQUIRED_ROLES}")
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


def role_to_meshes(
    topology: TunixTopology, devices: Sequence[jax.Device] | None = None
) -> dict[str, Mesh]:
    """Materialize declared device indices as one named JAX mesh per Tunix role."""
    visible_devices = tuple(jax.devices() if devices is None else devices)
    meshes: dict[str, Mesh] = {}
    for role, indices in topology.role_to_device_indices.items():
        if max(indices) >= len(visible_devices):
            raise TopologyConfigError(
                f"{role} requests device {max(indices)}, only {len(visible_devices)} are visible"
            )
        selected_devices = [visible_devices[index] for index in indices]
        meshes[role] = jax.make_mesh(
            (len(indices),), (topology.axis_name,), devices=selected_devices
        )
    return meshes


def tunix_role_to_meshes(topology: TunixTopology) -> dict[object, Mesh]:
    """Adapt a declared topology to the official Tunix `RLCluster.role_to_mesh` mapping."""
    from tunix.rl.rl_cluster import Role  # type: ignore[import-untyped]

    meshes = role_to_meshes(topology)
    return {
        Role.ACTOR: meshes["actor"],
        Role.ROLLOUT: meshes["rollout"],
        Role.CRITIC: meshes["critic"],
        Role.REFERENCE: meshes["reference"],
    }
