"""Construction of a real vendored CrafText runtime from the canonical MVP config.

This module builds a concrete vendor runtime instance from validated MVP
configuration, while preserving a stable adapter contract for training.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adapters import CagedCrafTextAdapter, CrafTextAdapter
from .config import MvpRunConfig
from .prompts import ActionCatalog


class RuntimeError(ValueError):
    """Raised when a validated config cannot construct the selected vendor runtime.

    Example:
        >>> raise RuntimeError("message")"""


@dataclass(frozen=True)
class CrafTextRuntime:
    """Opaque vendor environment/params behind the training-safe adapter boundary.

    :ivar adapter: CrafTextAdapter[object, object, object]
    :ivar env_params: object
    :ivar action_count: int
    :ivar actions: ActionCatalog

    Example:
        >>> obj = CrafTextRuntime(adapter=..., env_params=..., action_count=...)"""

    adapter: CrafTextAdapter[object, object, object]
    env_params: object
    action_count: int
    actions: ActionCatalog


def build_craftext_runtime(config: MvpRunConfig) -> CrafTextRuntime:
    """Build a non-auto-reset CrafText world selected by one validated config.

    :param config: Validated `MvpRunConfig` selecting environment implementation and preset.
    :returns: A `CrafTextRuntime` containing an adapter, env params and action catalogue.
    :raises RuntimeError: If config is not a CrafText runtime or its action cardinality is invalid.

    Example:
        >>> runtime = build_craftext_runtime(config)
    """
    try:
        if config.environment.implementation == "craftext":
            from craftext.environment.craftext_wrapper import (  # type: ignore[import-not-found]
                RawInstructionWrapper,
            )
            from craftext.environment.scenarious.manager import (  # type: ignore[import-not-found]
                DefaultInstructionTransformer,
                DefaultJAXRepresentation,
                JaxScenarioDataHandler,
            )
            from craftext.environment.scenarious.processors import (  # type: ignore[import-not-found]
                RawProcessor,
            )
            from craftext.environment.world_presets import (  # type: ignore[import-not-found]
                build_env_and_params,
                build_world_preset_spec,
            )
        elif config.environment.implementation == "caged-craftext":
            from caged_craftext.environment.caged_craftext_wrapper import (  # type: ignore[import-not-found]
                CMDPInstructionWrapper,
            )
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
    try:
        from craftax.craftax_classic.constants import Action  # type: ignore[import-untyped]

        labels = tuple(action.name for action in Action)
    except ImportError as error:
        raise RuntimeError("Craftax Action enum is required for text-policy labels") from error
    if len(labels) != action_count:
        raise RuntimeError("vendor Action enum does not match environment action cardinality")
    if config.environment.implementation == "craftext":
        scenario_handler = JaxScenarioDataHandler(
            scenario_processor=RawProcessor,
            instruction_transformer=DefaultInstructionTransformer,
            config_name=config.environment.scenario_config,
            jax_representation_class=DefaultJAXRepresentation,
        )
        instruction_environment = RawInstructionWrapper(
            environment, scenario_handler=scenario_handler
        )
        adapter: CrafTextAdapter[object, object, object] = CrafTextAdapter(
            instruction_environment,
            env_params,
            action_count,
            world_preset=spec.name,
            instructions=tuple(scenario_handler.scenario_data.instructions_list),
            instruction_index=config.environment.instruction_index,
        )
    else:
        instruction_environment = CMDPInstructionWrapper(
            environment, config_name=config.environment.scenario_config
        )
        adapter = CagedCrafTextAdapter(
            instruction_environment,
            env_params,
            action_count,
            world_preset=spec.name,
            instructions=tuple(instruction_environment.scenario_handler.scenario_data.instructions_list),
            text_constraints=tuple(
                instruction_environment.scenario_handler.scenario_data.texutal_constraints_list
            ),
            instruction_index=config.environment.instruction_index,
        )
    return CrafTextRuntime(
        adapter=adapter,
        env_params=env_params,
        action_count=action_count,
        actions=ActionCatalog(labels),
    )
