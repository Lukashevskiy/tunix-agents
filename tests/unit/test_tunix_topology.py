"""Tests for declarative role topology validation before accelerator allocation."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

import tunix_craftext.tunix as tunix_package
from tunix_craftext.tunix.topology import (
    TopologyConfigError,
    TunixTopology,
    load_tunix_topology,
    role_to_meshes,
    tunix_role_to_meshes,
)

ROOT = Path(__file__).resolve().parents[2]


def test_tunix_package_exports_match_legacy_topology_shim() -> None:
    assert tunix_package.TopologyConfigError is TopologyConfigError
    assert tunix_package.TunixTopology is TunixTopology
    assert tunix_package.load_tunix_topology is load_tunix_topology
    assert tunix_package.role_to_meshes is role_to_meshes
    assert tunix_package.tunix_role_to_meshes is tunix_role_to_meshes


def test_local_profile_colocates_all_tunix_roles_on_visible_device_zero() -> None:
    topology = load_tunix_topology(ROOT / "configs" / "topology" / "qwen_local_smoke.yaml")
    meshes = role_to_meshes(topology)

    assert tuple(meshes) == ("actor", "rollout", "critic", "reference")
    assert all(mesh.shape["data"] == 1 for mesh in meshes.values())


def test_tunix_adapter_uses_official_role_enum_for_declared_meshes() -> None:
    topology = load_tunix_topology(ROOT / "configs" / "topology" / "qwen_local_smoke.yaml")
    mapping = tunix_role_to_meshes(topology)

    assert {role.value for role in mapping} == {"actor", "rollout", "critic", "reference"}
    assert all(mesh.devices.size == 1 for mesh in mapping.values())


def test_agentic_grpo_topology_omits_the_unneeded_critic_role() -> None:
    topology = load_tunix_topology(ROOT / "configs/topology/qwen_agentic_grpo_local.yaml")
    mapping = tunix_role_to_meshes(topology)

    assert tuple(topology.role_to_device_indices) == ("actor", "rollout", "reference")
    assert {role.value for role in mapping} == {"actor", "rollout", "reference"}
    assert all(mesh.axis_names == ("fsdp", "tp") for mesh in mapping.values())


def test_topology_rejects_unknown_or_unavailable_device_indices() -> None:
    with pytest.raises(TopologyConfigError, match="roles"):
        TunixTopology("bad", "data", {"actor": (0,)})

    topology = TunixTopology(
        "bad-device",
        "data",
        {role: (len(jax.devices()),) for role in ("actor", "rollout", "critic", "reference")},
    )
    with pytest.raises(TopologyConfigError, match="only"):
        role_to_meshes(topology)
