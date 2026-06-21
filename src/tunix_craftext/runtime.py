"""Construction of a real vendored CrafText runtime from the canonical MVP config."""

from __future__ import annotations

from dataclasses import dataclass

from .adapters import CrafTextAdapter
from .config import MvpRunConfig


class RuntimeError(ValueError):
    """Raised when a validated config cannot construct the selected vendor runtime."""


@dataclass(frozen=True)
class CrafTextRuntime:
    """Opaque vendor environment/params behind the training-safe adapter boundary."""

    adapter: CrafTextAdapter[object, object, object]
    env_params: object
    action_count: int


def build_craftext_runtime(config: MvpRunConfig) -> CrafTextRuntime:
    """Build a non-auto-reset CrafText world selected by one validated config.

    :raises RuntimeError: If config is not a CrafText runtime or its action cardinality is invalid.
    """
    try:
        if config.environment.implementation == "craftext":
            from craftext.environment.world_presets import (  # type: ignore[import-not-found]
                build_env_and_params,
                build_world_preset_spec,
            )
        elif config.environment.implementation == "caged-craftext":
            from caged_craftext.environment.world_presets import (  # type: ignore[import-not-found]
                build_env_and_params,
                build_world_preset_spec,
            )
        else:
            raise RuntimeError("unsupported environment implementation")
    except ImportError as error:
        raise RuntimeError("install `tunix-craftext[envs]` to build CrafText runtime") from error
    spec = build_world_preset_spec(
        env_name=config.environment.base_environment,
        preset_name=config.environment.world_preset,
        seed=config.run.seed,
    )
    environment, env_params = build_env_and_params(spec, auto_reset=False)
    action_count = getattr(environment, "num_actions", None)
    if not isinstance(action_count, int) or action_count <= 0:
        raise RuntimeError("vendor environment must expose positive integer num_actions")
    return CrafTextRuntime(
        adapter=CrafTextAdapter(environment, env_params, action_count),
        env_params=env_params,
        action_count=action_count,
    )
