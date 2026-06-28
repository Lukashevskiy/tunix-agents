import jax
from jax.sharding import PartitionSpec

from tunix_craftext.core.resources import (
    ResourceConfig,
    batch_sharding,
    data_mesh,
    replicated_sharding,
)


def test_default_resource_config_builds_visible_device_mesh() -> None:
    assert data_mesh(ResourceConfig()).shape["data"] == len(jax.devices())


def test_resource_shardings_expose_data_and_replication_specs() -> None:
    config = ResourceConfig()
    assert batch_sharding(config).spec == PartitionSpec("data")
    assert replicated_sharding(config).spec == PartitionSpec()
