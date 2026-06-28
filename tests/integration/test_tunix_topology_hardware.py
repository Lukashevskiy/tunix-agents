"""Hardware-gated validation for the versioned multi-device Tunix topology."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from tunix_craftext.tunix.topology import load_tunix_topology, tunix_role_to_meshes

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four visible accelerator devices")
def test_four_device_colocated_profile_materializes_all_tunix_role_meshes() -> None:
    """A real multi-device runner must preserve declared role placement exactly."""
    topology = load_tunix_topology(
        ROOT / "configs" / "topology" / "qwen_four_device_colocated.yaml"
    )
    mapping = tunix_role_to_meshes(topology)

    assert {role.value for role in mapping} == {"actor", "rollout", "critic", "reference"}
    assert all(mesh.shape["data"] == 4 for mesh in mapping.values())
    assert all(mesh.devices.size == 4 for mesh in mapping.values())
