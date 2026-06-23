"""JAX mesh and placement primitives with a safe single-device default.

This module exposes helper functions for constructing data-parallel meshes and
explicit sharding policies. It defaults to a safe single-device configuration
while allowing explicit multi-device layouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax.sharding import NamedSharding, PartitionSpec


@dataclass(frozen=True)
class ResourceConfig:
    """Declarative data-parallel and memory-placement policy for trainable workloads.

    :ivar data_axis_size: int
    :ivar params_placement: str
    :ivar optimizer_placement: str
    :ivar trajectory_placement: str

    Example:
        >>> obj = ResourceConfig(data_axis_size=..., params_placement=..., optimizer_placement=...)"""

    data_axis_size: int = -1
    params_placement: str = "replicated"
    optimizer_placement: str = "device"
    trajectory_placement: str = "host"

    def __post_init__(self) -> None:
        if self.data_axis_size == 0 or self.data_axis_size < -1:
            raise ValueError("data_axis_size must be -1 or positive")
        if self.params_placement != "replicated":
            raise ValueError("only replicated params are supported before model parallelism")
        if self.optimizer_placement not in {"device", "pinned_host"}:
            raise ValueError("unsupported optimizer placement")
        if self.trajectory_placement not in {"device", "host"}:
            raise ValueError("unsupported trajectory placement")


def data_mesh(config: ResourceConfig) -> jax.sharding.Mesh:
    """Return a data-parallel mesh; ``-1`` uses all visible devices.

    :param config: ResourceConfig input value
    :returns: jax.sharding.Mesh

    Example:
        >>> result = data_mesh(config)
    """
    size = len(jax.devices()) if config.data_axis_size == -1 else config.data_axis_size
    if size > len(jax.devices()):
        raise ValueError("requested data axis exceeds visible devices")
    return jax.make_mesh((size,), ("data",))


def batch_sharding(config: ResourceConfig) -> NamedSharding:
    """Return data-axis sharding for leading batch dimension tensors.

    :param config: ResourceConfig input value
    :returns: NamedSharding

    Example:
        >>> result = batch_sharding(config)
    """
    return NamedSharding(data_mesh(config), PartitionSpec("data"))


def replicated_sharding(config: ResourceConfig) -> NamedSharding:
    """Return replicated parameter sharding before model parallelism is enabled.

    :param config: ResourceConfig input value
    :returns: NamedSharding

    Example:
    >>> result = replicated_sharding(config)"""
    return NamedSharding(data_mesh(config), PartitionSpec())
