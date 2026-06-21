import jax
from tunix_craftext.resources import ResourceConfig, data_mesh
def test_default_resource_config_builds_visible_device_mesh() -> None:
    assert data_mesh(ResourceConfig()).shape["data"] == len(jax.devices())
