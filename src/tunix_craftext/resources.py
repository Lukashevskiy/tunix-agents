"""JAX mesh and placement primitives with a safe single-device default."""
from __future__ import annotations
from dataclasses import dataclass
import jax

@dataclass(frozen=True)
class ResourceConfig:
    data_axis_size: int = -1
    optimizer_placement: str = "device"
    trajectory_placement: str = "host"
    def __post_init__(self) -> None:
        if self.data_axis_size == 0 or self.data_axis_size < -1: raise ValueError("data_axis_size must be -1 or positive")
        if self.optimizer_placement not in {"device", "pinned_host"}: raise ValueError("unsupported optimizer placement")

def data_mesh(config: ResourceConfig) -> jax.sharding.Mesh:
    """Return a data-parallel mesh; ``-1`` uses all visible devices."""
    size = len(jax.devices()) if config.data_axis_size == -1 else config.data_axis_size
    if size > len(jax.devices()): raise ValueError("requested data axis exceeds visible devices")
    return jax.make_mesh((size,), ("data",))
