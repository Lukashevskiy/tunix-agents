"""World preset helpers for CrafText/Craftax runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
import pathlib
from typing import Any, Callable, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from craftax.craftax_env import make_craftax_env_from_name

from craftext.environment.scenarious.loader import resolve_base_environment

WORLD_PRESET_CONFIG_DIR_NAME = "world_presets"
ROOT_PRESET_SECTION_NAMES = ("env", "map", "systems")


@dataclass(frozen=True)
class InventoryGrantSpec:
    """Probabilistic override for one named field.

    Attributes:
        item: Target field name in inventory, state, or rule mapping.
        value: Value to assign when the override is sampled.
        probability: Probability of applying the override on reset.
    """

    item: str
    value: Any
    probability: float = 1.0


@dataclass(frozen=True)
class GeneratedWorldState:
    """Container for generator-produced world state overlays.

    Attributes:
        map: Generated map tensor or per-level map tensor.
        player_position: Spawn or replacement player position.
        item_map: Optional item-layer replacement for full Craftax.
        mob_map: Optional mob-layer replacement for full Craftax.
        light_map: Optional light-layer replacement for full Craftax.
    """

    map: Any
    player_position: Any
    item_map: Any = None
    mob_map: Any = None
    light_map: Any = None


@dataclass(frozen=True)
class EnvPresetSpec:
    """Environment-level preset configuration.

    Attributes:
        env_name: Concrete Craftax environment id.
        seed: Seed used for preset construction and resets.
    """

    env_name: str
    seed: int


@dataclass(frozen=True)
class StaticEnvSpec:
    """Static environment limits and dimensions."""

    map_size: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class EnvParamsSpec:
    """Env params overrides applied before runtime starts."""

    overrides: Tuple[InventoryGrantSpec, ...] = ()


@dataclass(frozen=True)
class SpawnPolicySpec:
    """Spawn-related runtime controls."""

    disable_mob_spawns: bool = False


@dataclass(frozen=True)
class MovementPolicySpec:
    """Movement and collision policies applied after env steps."""

    behaviors: Tuple[str, ...] = ()
    rules: Tuple[InventoryGrantSpec, ...] = ()


@dataclass(frozen=True)
class ResetPolicySpec:
    """Reset-time state initialization policies."""

    behaviors: Tuple[str, ...] = ()
    starting_inventory: Tuple[InventoryGrantSpec, ...] = ()
    starting_intrinsics: Tuple[InventoryGrantSpec, ...] = ()


@dataclass(frozen=True)
class StepPolicySpec:
    """Post-step dynamics and recovery policies."""

    behaviors: Tuple[str, ...] = ()
    intrinsic_rates: Tuple[InventoryGrantSpec, ...] = ()
    intrinsic_thresholds: Tuple[InventoryGrantSpec, ...] = ()
    rules: Tuple[InventoryGrantSpec, ...] = ()


@dataclass(frozen=True)
class SystemsPresetSpec:
    """Cross-cutting systems configuration separated from world shape.

    Attributes:
        static_env: Static environment dimensions and limits.
        env_params: Direct overrides for Craftax env params.
        spawn: Spawn-related runtime controls.
        movement: Movement and collision policies.
        reset: Reset-time state initialization policies.
        step: Post-step dynamics and recovery policies.
    """

    static_env: StaticEnvSpec = field(default_factory=StaticEnvSpec)
    env_params: EnvParamsSpec = field(default_factory=EnvParamsSpec)
    spawn: SpawnPolicySpec = field(default_factory=SpawnPolicySpec)
    movement: MovementPolicySpec = field(default_factory=MovementPolicySpec)
    reset: ResetPolicySpec = field(default_factory=ResetPolicySpec)
    step: StepPolicySpec = field(default_factory=StepPolicySpec)


@dataclass(frozen=True)
class BoxGeneratorSpec:
    """Configuration for a boxed playable area generator."""

    inner_size: int
    perimeter_tree_prob: Optional[float] = None
    blocked_block: Optional[str] = None
    floor_block: Optional[str] = None
    perimeter_block: Optional[str] = None


@dataclass(frozen=True)
class RingGeneratorSpec:
    """Configuration for a ring-shaped playable area generator."""

    inner_radius: int
    outer_radius: int
    blocked_block: Optional[str] = None
    floor_block: Optional[str] = None


@dataclass(frozen=True)
class MapPresetSpec:
    """Map generation entrypoint.

    Attributes:
        generator_name: Registered generator name such as ``box`` or ``ring``.
        generator_config: Generator-specific config payload.
    """

    generator_name: Optional[str] = None
    generator_config: BoxGeneratorSpec | RingGeneratorSpec | None = None


@dataclass(frozen=True)
class WorldPresetSpec:
    """Resolved top-level world preset configuration.

    Attributes:
        name: User-facing preset name or ``inline`` for ad-hoc section input.
        env: Environment-specific preset settings.
        map: Map generation settings.
        systems: Cross-cutting runtime systems and policy settings.
    """

    name: str
    env: EnvPresetSpec
    map: MapPresetSpec
    systems: SystemsPresetSpec


def get_world_preset_config_dir() -> pathlib.Path:
    """Return the package directory that stores YAML world preset configs."""
    module = importlib.import_module("craftext")
    module_path = pathlib.Path(inspect.getmodule(module).__path__[0])
    return module_path / WORLD_PRESET_CONFIG_DIR_NAME


def _find_world_preset_config_path(preset_name: str) -> pathlib.Path:
    """Resolve one preset config path strictly relative to the preset directory."""
    config_dir = get_world_preset_config_dir()
    raw_path = pathlib.Path(preset_name)
    config_path = raw_path if raw_path.suffix == ".yaml" else raw_path.with_suffix(".yaml")
    resolved_path = (config_dir / config_path).resolve()
    if config_dir.resolve() not in resolved_path.parents:
        raise ValueError(f"World preset config path must stay inside {config_dir}: {preset_name}")
    if not resolved_path.exists():
        raise FileNotFoundError(f"World preset config {preset_name} not found at {resolved_path}")
    return resolved_path


def _load_world_preset_config(preset_name: str) -> dict[str, object]:
    config_path = _find_world_preset_config_path(preset_name)
    with open(config_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    if config_data is None:
        raise ValueError(f"World preset config {preset_name} is empty: {config_path}")
    if not isinstance(config_data, dict):
        raise TypeError(f"World preset config {preset_name} must be a YAML mapping: {config_path}")

    return dict(config_data)


def _as_section(value: Optional[Mapping[str, Any]], *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"World preset '{field_name}' section must be a mapping")
    return dict(value)


def _child_section(section: Mapping[str, Any], key: str, *, parent_name: str) -> dict[str, Any]:
    return _as_section(section.get(key), field_name=f"{parent_name}.{key}")


def _check_allowed_keys(section_name: str, data: Mapping[str, Any], allowed_keys: set[str]) -> None:
    unknown = sorted(set(data.keys()) - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown keys in world preset '{section_name}' section: {unknown}")


def _merge_sections(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    *,
    section_name: Optional[str] = None,
) -> dict[str, Any]:
    if not isinstance(base, Mapping):
        raise TypeError(f"World preset '{section_name or 'section'}' base value must be a mapping")
    if not isinstance(override, Mapping):
        raise TypeError(f"World preset '{section_name or 'section'}' override value must be a mapping")

    merged = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            child_name = f"{section_name}.{key}" if section_name else key
            merged[key] = _merge_sections(base_value, override_value, section_name=child_name)
        else:
            merged[key] = override_value
    return merged


def _merge_root_sections(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for section_name in ROOT_PRESET_SECTION_NAMES:
        base_section = base.get(section_name, {})
        override_section = override.get(section_name, {})
        if base_section or override_section:
            merged[section_name] = _merge_sections(base_section, override_section, section_name=section_name)
    return merged


def _extract_root_sections(data: Mapping[str, Any], *, parent_name: str) -> dict[str, dict[str, Any]]:
    return {
        section_name: _child_section(data, section_name, parent_name=parent_name)
        for section_name in ROOT_PRESET_SECTION_NAMES
    }


def _inline_root_sections(
    *,
    env: Optional[Mapping[str, Any]],
    map: Optional[Mapping[str, Any]],
    systems: Optional[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        "env": _as_section(env, field_name="env"),
        "map": _as_section(map, field_name="map"),
        "systems": _as_section(systems, field_name="systems"),
    }


def _registry_from_pairs(*pairs: tuple[str, Any]) -> dict[str, Any]:
    return dict(pairs)


def _registry_from_named_types(*types_: type[Any], name_attr: str) -> dict[str, type[Any]]:
    return {getattr(type_, name_attr): type_ for type_ in types_}


def _normalize_map_size(map_size: Optional[int | Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if map_size is None:
        return None
    if isinstance(map_size, int):
        return (map_size, map_size)
    return (int(map_size[0]), int(map_size[1]))


def _normalize_grants(raw_mapping: Any, *, field_name: str) -> Tuple[InventoryGrantSpec, ...]:
    if raw_mapping is None:
        return ()
    if not isinstance(raw_mapping, dict):
        raise TypeError(f"World preset '{field_name}' must be a mapping")

    grants: list[InventoryGrantSpec] = []
    for item_name, raw_value in raw_mapping.items():
        if not isinstance(item_name, str) or not item_name.strip():
            raise TypeError(f"World preset '{field_name}' item names must be non-empty strings")

        probability = 1.0
        value = raw_value
        if isinstance(raw_value, dict):
            if "value" in raw_value:
                value = raw_value["value"]
            else:
                value = raw_value.get("count")
            probability = raw_value.get("probability", 1.0)

        if value is None:
            raise ValueError(f"World preset '{field_name}' item {item_name} must define 'value' or 'count'")
        probability = float(probability)
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"World preset '{field_name}' item {item_name} probability must be in [0, 1]")

        grants.append(InventoryGrantSpec(item=item_name.strip(), value=value, probability=probability))

    return tuple(grants)


def _grants_to_mapping(grants: Tuple[InventoryGrantSpec, ...]) -> dict[str, Any]:
    return {grant.item: grant.value for grant in grants}


def _normalize_name_list(raw_value: Any, *, field_name: str) -> Tuple[str, ...]:
    if raw_value is None:
        return ()
    values = raw_value if isinstance(raw_value, (list, tuple)) else [raw_value]
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"World preset '{field_name}' entries must be non-empty strings")
        normalized.append(value.strip())
    return tuple(normalized)


def _normalize_generator_name(raw_value: Any) -> Optional[str]:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise TypeError("World preset map.generator must be a string")
    normalized = raw_value.strip().lower()
    if normalized in {"", "none", "default"}:
        return None
    if normalized not in {"box", "ring"}:
        raise ValueError(f"Unsupported world preset generator: {raw_value}")
    return normalized


def _resolve_seed(seed_value: Optional[Any]) -> int:
    if seed_value is None:
        return int(np.random.randint(0, 2**31 - 1))
    if not isinstance(seed_value, int):
        raise TypeError("World preset env.seed must be int")
    return int(seed_value)


def _validate_movement_policy_contracts(*, preset_name: str, movement_spec: MovementPolicySpec) -> None:
    if movement_spec.rules and not movement_spec.behaviors:
        raise ValueError(f"World preset {preset_name}: systems.movement.rules requires explicit movement behaviors")


def _validate_reset_policy_contracts(*, preset_name: str, reset_spec: ResetPolicySpec) -> None:
    if reset_spec.starting_inventory and "starting_inventory" not in reset_spec.behaviors:
        raise ValueError(
            f"World preset {preset_name}: systems.reset.starting_inventory requires 'starting_inventory' behavior"
        )
    if reset_spec.starting_intrinsics and "starting_intrinsics" not in reset_spec.behaviors:
        raise ValueError(
            f"World preset {preset_name}: systems.reset.starting_intrinsics requires 'starting_intrinsics' behavior"
        )


def _has_recovery_rules(step_spec: StepPolicySpec) -> bool:
    return any(
        grant.item.startswith(("instant_", "sleep_", "rest_", "wake_", "stop_"))
        for grant in step_spec.rules
    )


def _validate_step_policy_contracts(*, preset_name: str, step_spec: StepPolicySpec) -> None:
    if (step_spec.intrinsic_rates or step_spec.intrinsic_thresholds) and "intrinsic_dynamics" not in step_spec.behaviors:
        raise ValueError(
            f"World preset {preset_name}: systems.step intrinsic fields require 'intrinsic_dynamics' behavior"
        )
    if _has_recovery_rules(step_spec) and "instant_recovery" not in step_spec.behaviors:
        raise ValueError(
            f"World preset {preset_name}: systems.step.rules recovery entries require 'instant_recovery' behavior"
        )


def _validate_behavior_contracts(*, preset_name: str, systems_spec: SystemsPresetSpec) -> None:
    _validate_movement_policy_contracts(preset_name=preset_name, movement_spec=systems_spec.movement)
    _validate_reset_policy_contracts(preset_name=preset_name, reset_spec=systems_spec.reset)
    _validate_step_policy_contracts(preset_name=preset_name, step_spec=systems_spec.step)


def _validate_env_compatibility(
    *,
    preset_name: str,
    env_section: Mapping[str, Any],
    expected_env_name: str,
) -> None:
    declared_env_name = env_section.get("env_name")
    if declared_env_name is None:
        return

    declared_env_name_str = str(declared_env_name).strip()
    expected_env_name_str = expected_env_name.strip()
    if declared_env_name_str != expected_env_name_str:
        raise ValueError(
            f"World preset {preset_name} expects env.env_name={declared_env_name_str}, "
            f"but the active env is {expected_env_name_str}. "
            "Presets overlay the selected base environment and do not override it."
        )


def _build_env_spec(section: Mapping[str, Any], *, default_env_name: str, default_seed: Optional[int]) -> EnvPresetSpec:
    _check_allowed_keys("env", section, {"env_name", "seed"})
    resolved_seed = _resolve_seed(section.get("seed", default_seed))
    resolved_env_name = str(section.get("env_name", default_env_name)).strip()
    return EnvPresetSpec(env_name=resolved_env_name, seed=resolved_seed)


def _build_box_generator_config(config_section: Mapping[str, Any]) -> BoxGeneratorSpec:
    _check_allowed_keys(
        "map.generator.config",
        config_section,
        {"inner_size", "perimeter_tree_prob", "blocked_block", "floor_block", "perimeter_block"},
    )
    return BoxGeneratorSpec(
        inner_size=int(config_section["inner_size"]),
        perimeter_tree_prob=(
            None
            if config_section.get("perimeter_tree_prob") is None
            else float(config_section["perimeter_tree_prob"])
        ),
        blocked_block=(
            None if config_section.get("blocked_block") is None else str(config_section["blocked_block"])
        ),
        floor_block=(
            None if config_section.get("floor_block") is None else str(config_section["floor_block"])
        ),
        perimeter_block=(
            None if config_section.get("perimeter_block") is None else str(config_section["perimeter_block"])
        ),
    )


def _build_ring_generator_config(config_section: Mapping[str, Any]) -> RingGeneratorSpec:
    _check_allowed_keys(
        "map.generator.config",
        config_section,
        {"inner_radius", "outer_radius", "blocked_block", "floor_block"},
    )
    return RingGeneratorSpec(
        inner_radius=int(config_section["inner_radius"]),
        outer_radius=int(config_section["outer_radius"]),
        blocked_block=(
            None if config_section.get("blocked_block") is None else str(config_section["blocked_block"])
        ),
        floor_block=(
            None if config_section.get("floor_block") is None else str(config_section["floor_block"])
        ),
    )


MAP_GENERATOR_CONFIG_BUILDERS: dict[str, Callable[[Mapping[str, Any]], BoxGeneratorSpec | RingGeneratorSpec]] = _registry_from_pairs(
    ("box", _build_box_generator_config),
    ("ring", _build_ring_generator_config),
)


def _build_named_config(
    name: Optional[str],
    config_section: Mapping[str, Any],
    registry: Mapping[str, Callable[[Mapping[str, Any]], Any]],
    *,
    config_name: str,
) -> Any:
    if name is None:
        if config_section:
            raise ValueError(f"World preset {config_name} requires a matching name")
        return None
    builder = _resolve_registry_entry(registry, name, registry_name=f"{config_name} builder")
    return builder(config_section)


def _build_map_spec(section: Mapping[str, Any]) -> MapPresetSpec:
    _check_allowed_keys("map", section, {"generator"})
    generator_section = _child_section(section, "generator", parent_name="map")

    generator_name: Optional[str] = None
    generator_config: BoxGeneratorSpec | RingGeneratorSpec | None = None

    if generator_section:
        _check_allowed_keys("map.generator", generator_section, {"name", "config"})
        generator_name = _normalize_generator_name(generator_section.get("name"))
        config_section = _child_section(generator_section, "config", parent_name="map.generator")
        generator_config = _build_named_config(
            generator_name,
            config_section,
            MAP_GENERATOR_CONFIG_BUILDERS,
            config_name="map.generator.config",
        )
    return MapPresetSpec(generator_name=generator_name, generator_config=generator_config)


def _build_static_env_spec(section: Mapping[str, Any]) -> StaticEnvSpec:
    _check_allowed_keys("systems.static_env", section, {"map_size"})
    return StaticEnvSpec(map_size=_normalize_map_size(section.get("map_size")))


def _build_env_params_spec(section: Mapping[str, Any]) -> EnvParamsSpec:
    _check_allowed_keys("systems.env_params", section, {"overrides"})
    return EnvParamsSpec(
        overrides=_normalize_grants(section.get("overrides"), field_name="systems.env_params.overrides")
    )


def _build_spawn_policy_spec(section: Mapping[str, Any]) -> SpawnPolicySpec:
    _check_allowed_keys("systems.spawn", section, {"disable_mob_spawns"})
    return SpawnPolicySpec(disable_mob_spawns=bool(section.get("disable_mob_spawns", False)))


def _build_movement_policy_spec(section: Mapping[str, Any]) -> MovementPolicySpec:
    _check_allowed_keys("systems.movement", section, {"behaviors", "rules"})
    return MovementPolicySpec(
        behaviors=_normalize_name_list(section.get("behaviors"), field_name="systems.movement.behaviors"),
        rules=_normalize_grants(section.get("rules"), field_name="systems.movement.rules"),
    )


def _build_reset_policy_spec(section: Mapping[str, Any]) -> ResetPolicySpec:
    _check_allowed_keys(
        "systems.reset",
        section,
        {"behaviors", "starting_inventory", "starting_intrinsics"},
    )
    return ResetPolicySpec(
        behaviors=_normalize_name_list(section.get("behaviors"), field_name="systems.reset.behaviors"),
        starting_inventory=_normalize_grants(section.get("starting_inventory"), field_name="systems.reset.starting_inventory"),
        starting_intrinsics=_normalize_grants(
            section.get("starting_intrinsics"),
            field_name="systems.reset.starting_intrinsics",
        ),
    )


def _build_step_policy_spec(section: Mapping[str, Any]) -> StepPolicySpec:
    _check_allowed_keys(
        "systems.step",
        section,
        {"behaviors", "intrinsic_rates", "intrinsic_thresholds", "rules"},
    )
    return StepPolicySpec(
        behaviors=_normalize_name_list(section.get("behaviors"), field_name="systems.step.behaviors"),
        intrinsic_rates=_normalize_grants(section.get("intrinsic_rates"), field_name="systems.step.intrinsic_rates"),
        intrinsic_thresholds=_normalize_grants(
            section.get("intrinsic_thresholds"),
            field_name="systems.step.intrinsic_thresholds",
        ),
        rules=_normalize_grants(section.get("rules"), field_name="systems.step.rules"),
    )


def _build_systems_spec(section: Mapping[str, Any]) -> SystemsPresetSpec:
    _check_allowed_keys(
        section_name="systems",
        data=section,
        allowed_keys={"static_env", "env_params", "spawn", "movement", "reset", "step"},
    )
    static_env_section = _child_section(section, "static_env", parent_name="systems")
    env_params_section = _child_section(section, "env_params", parent_name="systems")
    spawn_section = _child_section(section, "spawn", parent_name="systems")
    movement_section = _child_section(section, "movement", parent_name="systems")
    reset_section = _child_section(section, "reset", parent_name="systems")
    step_section = _child_section(section, "step", parent_name="systems")

    return SystemsPresetSpec(
        static_env=_build_static_env_spec(static_env_section),
        env_params=_build_env_params_spec(env_params_section),
        spawn=_build_spawn_policy_spec(spawn_section),
        movement=_build_movement_policy_spec(movement_section),
        reset=_build_reset_policy_spec(reset_section),
        step=_build_step_policy_spec(step_section),
    )


def _assemble_world_preset_spec(
    *,
    preset_name: str,
    default_env_name: str,
    default_seed: Optional[int],
    env_section: Mapping[str, Any],
    map_section: Mapping[str, Any],
    systems_section: Mapping[str, Any],
) -> WorldPresetSpec:
    env_spec = _build_env_spec(env_section, default_env_name=default_env_name, default_seed=default_seed)
    map_spec = _build_map_spec(map_section)
    systems_spec = _build_systems_spec(systems_section)
    _validate_behavior_contracts(
        preset_name=preset_name,
        systems_spec=systems_spec,
    )
    return WorldPresetSpec(
        name=preset_name,
        env=env_spec,
        map=map_spec,
        systems=systems_spec,
    )


def _resolve_config_sections(preset_name: str, config_data: Mapping[str, Any]) -> dict[str, Any]:
    _check_allowed_keys("root", config_data, {"extends", *ROOT_PRESET_SECTION_NAMES})
    extends_value = config_data.get("extends")
    base_sections: dict[str, Any] = {}
    if extends_value is not None:
        if not isinstance(extends_value, str) or not extends_value.strip():
            raise TypeError(f"World preset {preset_name}: 'extends' must be a non-empty string")
        parent_name = extends_value.strip()
        base_sections = _resolve_config_sections(parent_name, _load_world_preset_config(parent_name))
    current_sections = _extract_root_sections(config_data, parent_name="root")
    return _merge_root_sections(base_sections, current_sections)


def build_world_preset_spec(
    *,
    env_name: str,
    preset_name: Optional[str],
    seed: Optional[int],
    env: Optional[Mapping[str, Any]] = None,
    map: Optional[Mapping[str, Any]] = None,
    systems: Optional[Mapping[str, Any]] = None,
) -> WorldPresetSpec:
    """Build a world preset spec from strict sectioned inputs."""
    inline_sections = _inline_root_sections(env=env, map=map, systems=systems)
    requested_name = (preset_name or "").strip()
    if requested_name:
        merged_sections = _merge_root_sections(
            _resolve_config_sections(requested_name, _load_world_preset_config(requested_name)),
            inline_sections,
        )
    else:
        merged_sections = inline_sections
    resolved_sections = _extract_root_sections(merged_sections, parent_name="root")
    env_section = resolved_sections["env"]
    _validate_env_compatibility(
        preset_name=requested_name or "inline",
        env_section=env_section,
        expected_env_name=env_name,
    )
    return _assemble_world_preset_spec(
        preset_name=requested_name or "inline",
        default_env_name=env_name,
        default_seed=seed,
        env_section=env_section,
        map_section=resolved_sections["map"],
        systems_section=resolved_sections["systems"],
    )


def _build_static_env_params(env_name: str, map_size: Optional[Tuple[int, int]]) -> Optional[Any]:
    if map_size is None:
        return None

    resolved_family = resolve_base_environment(env_name).family
    if resolved_family == "classic":
        from craftax.craftax_classic.envs.craftax_state import StaticEnvParams
    else:
        from craftax.craftax.craftax_state import StaticEnvParams

    return StaticEnvParams(map_size=map_size)


def _build_static_env_instance(env_name: str, static_env_params: Any, *, auto_reset: bool) -> Any:
    if env_name == "Craftax-Classic-Pixels-v1":
        from craftax.craftax_classic.envs.craftax_pixels_env import (
            CraftaxClassicPixelsEnv,
            CraftaxClassicPixelsEnvNoAutoReset,
        )

        return CraftaxClassicPixelsEnv(static_env_params) if auto_reset else CraftaxClassicPixelsEnvNoAutoReset(static_env_params)
    if env_name == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_classic.envs.craftax_symbolic_env import (
            CraftaxClassicSymbolicEnv,
            CraftaxClassicSymbolicEnvNoAutoReset,
        )

        return (
            CraftaxClassicSymbolicEnv(static_env_params)
            if auto_reset
            else CraftaxClassicSymbolicEnvNoAutoReset(static_env_params)
        )
    if env_name == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv, CraftaxPixelsEnvNoAutoReset

        return CraftaxPixelsEnv(static_env_params) if auto_reset else CraftaxPixelsEnvNoAutoReset(static_env_params)
    if env_name == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv, CraftaxSymbolicEnvNoAutoReset

        return CraftaxSymbolicEnv(static_env_params) if auto_reset else CraftaxSymbolicEnvNoAutoReset(static_env_params)
    raise ValueError(f"Unsupported Craftax environment for world preset: {env_name}")


def _build_env(spec: WorldPresetSpec, *, auto_reset: bool) -> Any:
    static_env_params = _build_static_env_params(spec.env.env_name, spec.systems.static_env.map_size)
    if static_env_params is None:
        return make_craftax_env_from_name(spec.env.env_name, auto_reset=auto_reset)
    return _build_static_env_instance(spec.env.env_name, static_env_params, auto_reset=auto_reset)


def _spawn_disable_overrides(env_params: Any) -> dict[str, Any]:
    replace_kwargs = {"mob_despawn_distance": 0}
    if hasattr(env_params, "spawn_cow_chance"):
        replace_kwargs["spawn_cow_chance"] = 0.0
    if hasattr(env_params, "spawn_zombie_base_chance"):
        replace_kwargs["spawn_zombie_base_chance"] = 0.0
    if hasattr(env_params, "spawn_zombie_night_chance"):
        replace_kwargs["spawn_zombie_night_chance"] = 0.0
    if hasattr(env_params, "spawn_skeleton_chance"):
        replace_kwargs["spawn_skeleton_chance"] = 0.0
    return replace_kwargs


def _apply_env_params_overrides(env_params: Any, spec: WorldPresetSpec) -> Any:
    replace_kwargs: dict[str, Any] = {}
    if spec.systems.spawn.disable_mob_spawns:
        replace_kwargs.update(_spawn_disable_overrides(env_params))
    for override in spec.systems.env_params.overrides:
        if not hasattr(env_params, override.item):
            raise ValueError(
                f"World preset env_params override {override.item!r} does not exist for env {spec.env.env_name}"
            )
        replace_kwargs[override.item] = override.value
    return env_params.replace(**replace_kwargs) if replace_kwargs else env_params


def _uses_runtime_pipeline(spec: WorldPresetSpec) -> bool:
    return any(
        (
            spec.map.generator_name is not None,
            bool(spec.systems.movement.behaviors),
            bool(spec.systems.reset.behaviors),
            bool(spec.systems.step.behaviors),
        )
    )


def build_env_and_params(
    spec: WorldPresetSpec,
    *,
    auto_reset: bool = False,
    adapter_cls: Optional[type[Any]] = None,
) -> Tuple[Any, Any]:
    """Construct the runtime environment and env params for a preset.

    Args:
        spec: Resolved world preset specification.
        auto_reset: Whether to build an auto-resetting Craftax env variant.
        adapter_cls: Optional runtime adapter override. Defaults to ``CompositePresetAdapter``.

    Returns:
        A tuple of ``(env, env_params)`` ready for reset/step calls.
    """
    env = _build_env(spec, auto_reset=auto_reset)
    env_params = _apply_env_params_overrides(env.default_params, spec)
    if _uses_runtime_pipeline(spec):
        runtime_adapter_cls = CompositePresetAdapter if adapter_cls is None else adapter_cls
        env = runtime_adapter_cls(env, spec)

    return env, env_params


def _distance_mask(height: int, width: int, center_x: int, center_y: int, inner_radius: int, outer_radius: int) -> jnp.ndarray:
    xs = jnp.arange(height)[:, None]
    ys = jnp.arange(width)[None, :]
    dist2 = (xs - center_x) ** 2 + (ys - center_y) ** 2
    return jnp.logical_and(dist2 >= inner_radius**2, dist2 <= outer_radius**2)


def _apply_ring_to_level(
    level_map: jnp.ndarray,
    *,
    player_position: jnp.ndarray,
    blocked_value: int,
    spawn_value: int,
    inner_radius: int,
    outer_radius: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    height, width = level_map.shape
    center_x = height // 2
    center_y = width // 2
    ring_mask = _distance_mask(height, width, center_x, center_y, inner_radius, outer_radius)

    new_map = jnp.where(ring_mask, level_map, blocked_value)
    spawn_x = center_x
    spawn_y = center_y + min(outer_radius, width // 2 - 2)
    if inner_radius > 0:
        spawn_y = center_y + min(max(inner_radius + 1, 1), width // 2 - 2)
    spawn_position = jnp.array([spawn_x, spawn_y], dtype=jnp.int32)
    new_map = new_map.at[spawn_position[0], spawn_position[1]].set(spawn_value)
    return new_map, spawn_position


def _apply_box_to_level(
    level_map: jnp.ndarray,
    *,
    key: Any,
    blocked_value: int,
    floor_value: int,
    tree_value: int,
    inner_size: int,
    perimeter_tree_prob: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    height, width = level_map.shape
    center_x = height // 2
    center_y = width // 2
    half = inner_size // 2

    inner_x0 = center_x - half
    inner_x1 = inner_x0 + inner_size
    inner_y0 = center_y - half
    inner_y1 = inner_y0 + inner_size

    base_map = jnp.full_like(level_map, blocked_value)
    base_map = base_map.at[inner_x0:inner_x1, inner_y0:inner_y1].set(floor_value)

    inner_height = inner_x1 - inner_x0
    inner_width = inner_y1 - inner_y0
    xs = jnp.arange(inner_height)[:, None]
    ys = jnp.arange(inner_width)[None, :]
    inner_perimeter_mask = jnp.logical_or(
        jnp.logical_or(xs == 0, xs == inner_height - 1),
        jnp.logical_or(ys == 0, ys == inner_width - 1),
    )

    random_values = jax.random.uniform(key, shape=(inner_height, inner_width))
    tree_mask = jnp.logical_and(inner_perimeter_mask, random_values < perimeter_tree_prob)
    inner_slice = base_map[inner_x0:inner_x1, inner_y0:inner_y1]
    inner_slice = jnp.where(tree_mask, tree_value, inner_slice)
    base_map = base_map.at[inner_x0:inner_x1, inner_y0:inner_y1].set(inner_slice)

    spawn_position = jnp.array([center_x, center_y], dtype=jnp.int32)
    base_map = base_map.at[spawn_position[0], spawn_position[1]].set(floor_value)
    return base_map, spawn_position


def _resolve_classic_block_value(block_name: Optional[str], default_value: int) -> int:
    if not block_name:
        return default_value
    from craftax.craftax_classic.constants import BlockType

    normalized = str(block_name).strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return BlockType[normalized].value
    except KeyError as exc:
        raise ValueError(f"Unknown classic block name for world preset: {block_name}") from exc


def _resolve_full_block_value(block_name: Optional[str], default_value: int) -> int:
    if not block_name:
        return default_value
    from craftax.craftax.constants import BlockType

    normalized = str(block_name).strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return BlockType[normalized].value
    except KeyError as exc:
        raise ValueError(f"Unknown full block name for world preset: {block_name}") from exc


def _coerce_record_value(current_value: Any, desired_value: Any):
    if isinstance(current_value, jnp.ndarray):
        if isinstance(desired_value, (list, tuple)):
            array_value = jnp.asarray(desired_value, dtype=current_value.dtype)
        else:
            array_value = jnp.full(current_value.shape, desired_value, dtype=current_value.dtype)
        if array_value.shape != current_value.shape:
            raise ValueError(
                f"World preset array shape mismatch: expected {current_value.shape}, got {array_value.shape}"
            )
        return array_value

    if isinstance(desired_value, (list, tuple, dict)):
        raise TypeError("Scalar fields must use scalar values")
    return type(current_value)(desired_value)


def _apply_grants_to_record(
    *,
    record: Any,
    grants: Tuple[InventoryGrantSpec, ...],
    key: Any,
    context_name: str,
):
    updated_record = record
    rng = key
    for grant in grants:
        if not hasattr(updated_record, grant.item):
            raise ValueError(f"World preset item {grant.item!r} does not exist in {context_name}")

        rng, sample_rng = jax.random.split(rng)
        include_item = jax.random.uniform(sample_rng) < float(grant.probability)
        current_value = getattr(updated_record, grant.item)
        desired_value = _coerce_record_value(current_value=current_value, desired_value=grant.value)
        final_value = jax.tree_util.tree_map(
            lambda desired, current: jax.lax.select(include_item, desired, current),
            desired_value,
            current_value,
        )
        updated_record = updated_record.replace(**{grant.item: final_value})
    return updated_record


def _apply_state_grants_to_state(
    *,
    state: Any,
    grants: Tuple[InventoryGrantSpec, ...],
    key: Any,
    label: str,
    env_name: str,
):
    rng = key
    updates: dict[str, Any] = {}
    for grant in grants:
        if not hasattr(state, grant.item):
            raise ValueError(
                f"World preset {label} item {grant.item!r} does not exist for env {env_name}"
            )
        rng, sample_rng = jax.random.split(rng)
        include_item = jax.random.uniform(sample_rng) < float(grant.probability)
        current_value = getattr(state, grant.item)
        desired_value = _coerce_record_value(current_value=current_value, desired_value=grant.value)
        final_value = jax.tree_util.tree_map(
            lambda desired, current: jax.lax.select(include_item, desired, current),
            desired_value,
            current_value,
        )
        updates[grant.item] = final_value
    return state.replace(**updates) if updates else state


class BaseWorldGenerator:
    """Extension point for custom world generation overlays.

    Subclasses translate ``MapPresetSpec`` into a generated world-state overlay
    that is applied during ``reset`` before gameplay begins.
    """

    generator_name = "base"

    def __init__(self, spec: WorldPresetSpec, resolved_family: str) -> None:
        self.spec = spec
        self.resolved_family = resolved_family

    def apply(self, state: Any, key: Any) -> GeneratedWorldState:
        raise NotImplementedError


class BoxWorldGenerator(BaseWorldGenerator):
    generator_name = "box"

    def apply(self, state: Any, key: Any) -> GeneratedWorldState:
        if not isinstance(self.spec.map.generator_config, BoxGeneratorSpec):
            raise TypeError("BoxWorldGenerator requires BoxGeneratorSpec")
        config = self.spec.map.generator_config
        if self.resolved_family == "classic":
            from craftax.craftax_classic.constants import BlockType

            blocked_value = _resolve_classic_block_value(config.blocked_block, BlockType.OUT_OF_BOUNDS.value)
            floor_value = _resolve_classic_block_value(config.floor_block, BlockType.GRASS.value)
            perimeter_value = _resolve_classic_block_value(config.perimeter_block, BlockType.TREE.value)
            updated_map, spawn_position = _apply_box_to_level(
                state.map,
                key=key,
                blocked_value=blocked_value,
                floor_value=floor_value,
                tree_value=perimeter_value,
                inner_size=int(config.inner_size),
                perimeter_tree_prob=float(config.perimeter_tree_prob or 0.7),
            )
            return GeneratedWorldState(map=updated_map, player_position=spawn_position)

        from craftax.craftax.constants import BlockType

        current_level = int(state.player_level)
        current_map = state.map[current_level]
        default_floor_value = BlockType.GRASS.value if current_level == 0 else BlockType.PATH.value
        default_perimeter_value = BlockType.TREE.value if current_level == 0 else BlockType.WALL.value
        blocked_value = _resolve_full_block_value(config.blocked_block, BlockType.OUT_OF_BOUNDS.value)
        floor_value = _resolve_full_block_value(config.floor_block, default_floor_value)
        perimeter_value = _resolve_full_block_value(config.perimeter_block, default_perimeter_value)
        updated_level_map, spawn_position = _apply_box_to_level(
            current_map,
            key=key,
            blocked_value=blocked_value,
            floor_value=floor_value,
            tree_value=perimeter_value,
            inner_size=int(config.inner_size),
            perimeter_tree_prob=float(config.perimeter_tree_prob or 0.7),
        )
        updated_light_level = jnp.where(
            updated_level_map == BlockType.OUT_OF_BOUNDS.value,
            0.0,
            state.light_map[current_level],
        )
        return GeneratedWorldState(
            map=state.map.at[current_level].set(updated_level_map),
            player_position=spawn_position,
            item_map=state.item_map.at[current_level].set(jnp.zeros_like(state.item_map[current_level])),
            mob_map=state.mob_map.at[current_level].set(jnp.zeros_like(state.mob_map[current_level])),
            light_map=state.light_map.at[current_level].set(updated_light_level),
        )


class RingWorldGenerator(BaseWorldGenerator):
    generator_name = "ring"

    def apply(self, state: Any, key: Any) -> GeneratedWorldState:
        if not isinstance(self.spec.map.generator_config, RingGeneratorSpec):
            raise TypeError("RingWorldGenerator requires RingGeneratorSpec")
        config = self.spec.map.generator_config
        if self.resolved_family == "classic":
            from craftax.craftax_classic.constants import BlockType

            blocked_value = _resolve_classic_block_value(config.blocked_block, BlockType.WATER.value)
            floor_value = _resolve_classic_block_value(config.floor_block, BlockType.GRASS.value)
            updated_map, spawn_position = _apply_ring_to_level(
                state.map,
                player_position=state.player_position,
                blocked_value=blocked_value,
                spawn_value=floor_value,
                inner_radius=int(config.inner_radius),
                outer_radius=int(config.outer_radius),
            )
            return GeneratedWorldState(map=updated_map, player_position=spawn_position)

        from craftax.craftax.constants import BlockType, ItemType

        current_level = int(state.player_level)
        current_map = state.map[current_level]
        blocked_value = _resolve_full_block_value(
            config.blocked_block,
            BlockType.WATER.value if current_level == 0 else BlockType.WALL.value,
        )
        floor_value = _resolve_full_block_value(
            config.floor_block,
            BlockType.GRASS.value if current_level == 0 else BlockType.PATH.value,
        )
        updated_level_map, spawn_position = _apply_ring_to_level(
            current_map,
            player_position=state.player_position,
            blocked_value=blocked_value,
            spawn_value=floor_value,
            inner_radius=int(config.inner_radius),
            outer_radius=int(config.outer_radius),
        )
        ring_mask = _distance_mask(
            current_map.shape[0],
            current_map.shape[1],
            current_map.shape[0] // 2,
            current_map.shape[1] // 2,
            int(config.inner_radius),
            int(config.outer_radius),
        )
        return GeneratedWorldState(
            map=state.map.at[current_level].set(updated_level_map),
            player_position=spawn_position,
            item_map=state.item_map.at[current_level].set(
                jnp.where(ring_mask, state.item_map[current_level], ItemType.NONE.value)
            ),
            mob_map=state.mob_map.at[current_level].set(jnp.where(ring_mask, state.mob_map[current_level], 0)),
            light_map=state.light_map.at[current_level].set(
                jnp.where(ring_mask, state.light_map[current_level], 0.0)
            ),
        )


WORLD_GENERATOR_REGISTRY: dict[str, type[BaseWorldGenerator]] = _registry_from_named_types(
    BoxWorldGenerator,
    RingWorldGenerator,
    name_attr="generator_name",
)


class BaseMovementBehavior:
    """Extension point for movement and collision policies.

    Movement behaviors run after generation and may adjust reset or step-time
    state while remaining independent from the map generator itself.
    """

    behavior_name = "base_movement"

    def __init__(self, spec: WorldPresetSpec, resolved_family: str) -> None:
        self.spec = spec
        self.resolved_family = resolved_family
        self.rules = _grants_to_mapping(spec.systems.movement.rules)

    def apply_reset(self, state: Any, key: Any) -> Any:
        return state

    def apply_step(self, previous_state: Any, new_state: Any) -> Any:
        return new_state


class SolidBlocksBehavior(BaseMovementBehavior):
    behavior_name = "solid_blocks"

    def __init__(self, spec: WorldPresetSpec, resolved_family: str) -> None:
        super().__init__(spec, resolved_family)
        self.solid_block_values = self._resolve_solid_block_values()

    def apply_step(self, previous_state: Any, new_state: Any) -> Any:
        if not self.solid_block_values:
            return new_state

        if self.resolved_family == "classic":
            pos = new_state.player_position
            current_block = int(new_state.map[pos[0], pos[1]])
        else:
            pos = new_state.player_position
            level = int(new_state.player_level)
            current_block = int(new_state.map[level, pos[0], pos[1]])

        if current_block not in self.solid_block_values:
            return new_state
        return new_state.replace(player_position=previous_state.player_position)

    def _resolve_solid_block_values(self) -> set[int]:
        solid_values: set[int] = set()
        if bool(self.rules.get("solid_out_of_bounds", False)):
            if self.resolved_family == "classic":
                from craftax.craftax_classic.constants import BlockType
            else:
                from craftax.craftax.constants import BlockType
            solid_values.add(int(BlockType.OUT_OF_BOUNDS.value))

        extra_blocks = self.rules.get("solid_blocks", [])
        if isinstance(extra_blocks, str):
            extra_blocks = [extra_blocks]
        if extra_blocks:
            resolver = _resolve_classic_block_value if self.resolved_family == "classic" else _resolve_full_block_value
            for block_name in extra_blocks:
                solid_values.add(int(resolver(str(block_name), 0)))
        return solid_values


MOVEMENT_BEHAVIOR_REGISTRY: dict[str, type[BaseMovementBehavior]] = _registry_from_named_types(
    SolidBlocksBehavior,
    name_attr="behavior_name",
)


class BaseResetBehavior:
    """Extension point for reset-time state initialization."""

    behavior_name = "base_reset"

    def __init__(self, spec: WorldPresetSpec) -> None:
        self.spec = spec

    def apply_reset(self, state: Any, key: Any) -> Any:
        return state


class StartingInventoryBehavior(BaseResetBehavior):
    behavior_name = "starting_inventory"

    def apply_reset(self, state: Any, key: Any) -> Any:
        inventory = _apply_grants_to_record(
            record=state.inventory,
            grants=self.spec.systems.reset.starting_inventory,
            key=key,
            context_name=f"inventory for env {self.spec.env.env_name}",
        )
        return state.replace(inventory=inventory)


class StartingIntrinsicsBehavior(BaseResetBehavior):
    behavior_name = "starting_intrinsics"

    def apply_reset(self, state: Any, key: Any) -> Any:
        return _apply_state_grants_to_state(
            state=state,
            grants=self.spec.systems.reset.starting_intrinsics,
            key=key,
            label="starting_intrinsics",
            env_name=self.spec.env.env_name,
        )


RESET_BEHAVIOR_REGISTRY: dict[str, type[BaseResetBehavior]] = _registry_from_named_types(
    StartingInventoryBehavior,
    StartingIntrinsicsBehavior,
    name_attr="behavior_name",
)


class BaseStepBehavior:
    """Extension point for post-step state dynamics and recovery."""

    behavior_name = "base_step"

    def __init__(self, spec: WorldPresetSpec) -> None:
        self.spec = spec

    def apply_step(self, previous_state: Any, new_state: Any) -> Any:
        return new_state


class IntrinsicDynamicsBehavior(BaseStepBehavior):
    behavior_name = "intrinsic_dynamics"

    def __init__(self, spec: WorldPresetSpec) -> None:
        super().__init__(spec)
        self.rate_values = _grants_to_mapping(spec.systems.step.intrinsic_rates)
        self.threshold_values = _grants_to_mapping(spec.systems.step.intrinsic_thresholds)

    def apply_step(self, previous_state: Any, new_state: Any) -> Any:
        state = new_state
        updates: dict[str, Any] = {}

        def _value(name: str, default: float) -> float:
            return float(self.rate_values.get(name, default))

        def _threshold(name: str, default: float) -> float:
            return float(self.threshold_values.get(name, default))

        def _process_meter(counter_name: str, resource_name: str, rate_name: str, threshold_name: str, default_threshold: float):
            if not hasattr(state, counter_name) or not hasattr(state, resource_name):
                return
            rate = _value(rate_name, 0.0)
            threshold = max(_threshold(threshold_name, default_threshold), 1e-6)
            if rate == 0.0:
                return
            counter = getattr(state, counter_name) + rate
            ticks = jnp.floor(jnp.maximum(counter, 0.0) / threshold).astype(jnp.int32)
            new_counter = counter - ticks.astype(counter.dtype) * threshold
            resource = jnp.maximum(getattr(state, resource_name) - ticks, 0)
            updates[counter_name] = new_counter
            updates[resource_name] = resource

        _process_meter("player_hunger", "player_food", "player_hunger_rate", "player_hunger_threshold", 25.0)
        _process_meter("player_thirst", "player_drink", "player_thirst_rate", "player_thirst_threshold", 20.0)

        if hasattr(state, "player_fatigue") and hasattr(state, "player_energy"):
            awake_rate = _value("player_fatigue_rate", 0.0)
            sleep_rate = _value("player_fatigue_sleep_rate", -abs(awake_rate) if awake_rate != 0.0 else 0.0)
            fatigue_delta = jax.lax.select(getattr(state, "is_sleeping", False), sleep_rate, awake_rate)
            if fatigue_delta != 0.0:
                fatigue = state.player_fatigue + fatigue_delta
                drop_threshold = max(_threshold("player_fatigue_threshold", 30.0), 1e-6)
                recover_threshold = min(_threshold("player_fatigue_recover_threshold", -10.0), -1e-6)
                drain_ticks = jnp.floor(jnp.maximum(fatigue, 0.0) / drop_threshold).astype(jnp.int32)
                fatigue = fatigue - drain_ticks.astype(fatigue.dtype) * drop_threshold
                energy = jnp.maximum(state.player_energy - drain_ticks, 0)
                recover_ticks = jnp.floor(jnp.maximum(-fatigue, 0.0) / abs(recover_threshold)).astype(jnp.int32)
                fatigue = fatigue + recover_ticks.astype(fatigue.dtype) * abs(recover_threshold)
                energy = jnp.minimum(energy + recover_ticks, 9)
                updates["player_fatigue"] = fatigue
                updates["player_energy"] = energy

        if hasattr(state, "player_recover") and hasattr(state, "player_health"):
            positive_rate = _value("player_recover_rate", 0.0)
            negative_rate = _value("player_recover_penalty_rate", -abs(positive_rate) if positive_rate != 0.0 else 0.0)
            has_food = getattr(state, "player_food", 1) > 0
            has_drink = getattr(state, "player_drink", 1) > 0
            has_energy = jnp.logical_or(getattr(state, "player_energy", 1) > 0, getattr(state, "is_sleeping", False))
            all_necessities = jnp.logical_and(jnp.logical_and(has_food, has_drink), has_energy)
            recover_delta = jax.lax.select(all_necessities, positive_rate, negative_rate)
            if recover_delta != 0.0:
                recover = state.player_recover + recover_delta
                pos_threshold = max(_threshold("player_recover_positive_threshold", 25.0), 1e-6)
                neg_threshold = min(_threshold("player_recover_negative_threshold", -15.0), -1e-6)
                heal_ticks = jnp.floor(jnp.maximum(recover, 0.0) / pos_threshold).astype(jnp.int32)
                recover = recover - heal_ticks.astype(recover.dtype) * pos_threshold
                health = jnp.minimum(state.player_health + heal_ticks, 9)
                hurt_ticks = jnp.floor(jnp.maximum(-recover, 0.0) / abs(neg_threshold)).astype(jnp.int32)
                recover = recover + hurt_ticks.astype(recover.dtype) * abs(neg_threshold)
                health = jnp.maximum(health - hurt_ticks, 0)
                updates["player_recover"] = recover
                updates["player_health"] = health

        return state.replace(**updates) if updates else state


class InstantRecoveryBehavior(BaseStepBehavior):
    behavior_name = "instant_recovery"

    def __init__(self, spec: WorldPresetSpec) -> None:
        super().__init__(spec)
        self.rules = _grants_to_mapping(spec.systems.step.rules)

    def apply_step(self, previous_state: Any, new_state: Any) -> Any:
        state = new_state
        updates: dict[str, Any] = {}
        instant_sleep_enabled = bool(self.rules.get("instant_sleep_recovery", False))
        instant_rest_enabled = bool(self.rules.get("instant_rest_recovery", False))

        if instant_sleep_enabled and getattr(state, "is_sleeping", False):
            self._apply_recovery_mode_updates(state=state, updates=updates, prefix="sleep")
            if bool(self.rules.get("wake_after_sleep_recovery", True)) and hasattr(state, "is_sleeping"):
                updates["is_sleeping"] = False

        if instant_rest_enabled and getattr(state, "is_resting", False):
            self._apply_recovery_mode_updates(state=state, updates=updates, prefix="rest")
            if bool(self.rules.get("stop_rest_after_recovery", True)) and hasattr(state, "is_resting"):
                updates["is_resting"] = False

        return state.replace(**updates) if updates else state

    def _apply_recovery_mode_updates(self, *, state: Any, updates: dict[str, Any], prefix: str):
        if hasattr(state, "player_energy"):
            energy_target = self.rules.get(f"{prefix}_energy_value", 9)
            updates["player_energy"] = jnp.asarray(energy_target, dtype=getattr(state.player_energy, "dtype", None))
        if hasattr(state, "player_fatigue"):
            fatigue_target = self.rules.get(f"{prefix}_fatigue_value", 0.0)
            updates["player_fatigue"] = jnp.asarray(fatigue_target, dtype=getattr(state.player_fatigue, "dtype", None))
        if hasattr(state, "player_mana") and f"{prefix}_mana_value" in self.rules:
            updates["player_mana"] = jnp.asarray(
                self.rules[f"{prefix}_mana_value"],
                dtype=getattr(state.player_mana, "dtype", None),
            )
        if hasattr(state, "player_recover") and f"{prefix}_recover_value" in self.rules:
            updates["player_recover"] = jnp.asarray(
                self.rules[f"{prefix}_recover_value"],
                dtype=getattr(state.player_recover, "dtype", None),
            )
        if hasattr(state, "player_health") and f"{prefix}_health_value" in self.rules:
            updates["player_health"] = jnp.asarray(
                self.rules[f"{prefix}_health_value"],
                dtype=getattr(state.player_health, "dtype", None),
            )


STEP_BEHAVIOR_REGISTRY: dict[str, type[BaseStepBehavior]] = _registry_from_named_types(
    IntrinsicDynamicsBehavior,
    InstantRecoveryBehavior,
    name_attr="behavior_name",
)


def _resolve_registry_entry(
    registry: Mapping[str, Any],
    name: str,
    *,
    registry_name: str,
) -> Any:
    entry = registry.get(name)
    if entry is None:
        raise ValueError(f"Unknown {registry_name}: {name}")
    return entry


def _build_components(
    names: Tuple[str, ...],
    registry: Mapping[str, type[Any]],
    factory: Callable[[type[Any]], Any],
    *,
    registry_name: str,
) -> tuple[Any, ...]:
    return tuple(
        factory(_resolve_registry_entry(registry, name, registry_name=registry_name))
        for name in names
    )


class CompositePresetAdapter:
    """Single runtime wrapper that executes all preset pipelines.

    The adapter applies reset-time and step-time transforms in a fixed order and
    recomputes observations once after all transforms complete.

    Args:
        env: Wrapped Craftax environment.
        spec: Resolved world preset specification.
    """

    def __init__(self, env: Any, spec: WorldPresetSpec) -> None:
        self.env = env
        self.spec = spec
        self.resolved_family = resolve_base_environment(spec.env.env_name).family
        self.world_generator = self._build_world_generator()
        self.movement_behaviors = self._build_movement_behaviors()
        self.reset_behaviors = self._build_reset_behaviors()
        self.step_behaviors = self._build_step_behaviors()
        self.reset_transforms = self._build_reset_pipeline()
        self.step_transforms = self._build_step_pipeline()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, key: Any, params: Any):
        obs, state = self.env.reset(key, params)
        for transform in self.reset_transforms:
            state = transform(state, key)
        return self._finalize(obs=obs, state=state)

    def step(self, key: Any, state: Any, action: int, params: Any):
        obs, new_state, reward, done, info = self.env.step(key, state, action, params)
        for transform in self.step_transforms:
            new_state = transform(state, new_state)
        obs, new_state = self._finalize(obs=obs, state=new_state)
        return obs, new_state, reward, done, info

    def _finalize(self, *, obs: Any, state: Any):
        if hasattr(self.env, "get_obs"):
            obs = self.env.get_obs(state)
        return obs, state

    def _build_reset_pipeline(self) -> tuple[Callable[[Any, Any], Any], ...]:
        transforms: list[Callable[[Any, Any], Any]] = []
        if self.spec.map.generator_name is not None:
            transforms.append(self._apply_map_overlay)
        for behavior in self.movement_behaviors:
            if behavior.apply_reset.__func__ is not BaseMovementBehavior.apply_reset:
                transforms.append(behavior.apply_reset)
        for behavior in self.reset_behaviors:
            if behavior.apply_reset.__func__ is not BaseResetBehavior.apply_reset:
                transforms.append(behavior.apply_reset)
        return tuple(transforms)

    def _build_step_pipeline(self) -> tuple[Callable[[Any, Any], Any], ...]:
        transforms: list[Callable[[Any, Any], Any]] = []
        for behavior in self.movement_behaviors:
            if behavior.apply_step.__func__ is not BaseMovementBehavior.apply_step:
                transforms.append(behavior.apply_step)
        for behavior in self.step_behaviors:
            if behavior.apply_step.__func__ is not BaseStepBehavior.apply_step:
                transforms.append(behavior.apply_step)
        return tuple(transforms)

    def _build_movement_behaviors(self) -> tuple[BaseMovementBehavior, ...]:
        return _build_components(
            self.spec.systems.movement.behaviors,
            MOVEMENT_BEHAVIOR_REGISTRY,
            lambda behavior_cls: behavior_cls(self.spec, self.resolved_family),
            registry_name="movement behavior for world preset",
        )

    def _build_reset_behaviors(self) -> tuple[BaseResetBehavior, ...]:
        return _build_components(
            self.spec.systems.reset.behaviors,
            RESET_BEHAVIOR_REGISTRY,
            lambda behavior_cls: behavior_cls(self.spec),
            registry_name="reset behavior for world preset",
        )

    def _build_step_behaviors(self) -> tuple[BaseStepBehavior, ...]:
        return _build_components(
            self.spec.systems.step.behaviors,
            STEP_BEHAVIOR_REGISTRY,
            lambda behavior_cls: behavior_cls(self.spec),
            registry_name="step behavior for world preset",
        )

    def _apply_map_overlay(self, state: Any, key: Any):
        if self.world_generator is None:
            return state
        generated = self.world_generator.apply(state, key)
        updates = {
            "map": generated.map,
            "player_position": generated.player_position,
        }
        if generated.item_map is not None:
            updates["item_map"] = generated.item_map
        if generated.mob_map is not None:
            updates["mob_map"] = generated.mob_map
        if generated.light_map is not None:
            updates["light_map"] = generated.light_map
        return state.replace(**updates)

    def _build_world_generator(self) -> Optional[BaseWorldGenerator]:
        if self.spec.map.generator_name is None:
            return None
        generator_cls = _resolve_registry_entry(
            WORLD_GENERATOR_REGISTRY,
            self.spec.map.generator_name,
            registry_name="world generator for preset",
        )
        return generator_cls(self.spec, self.resolved_family)
