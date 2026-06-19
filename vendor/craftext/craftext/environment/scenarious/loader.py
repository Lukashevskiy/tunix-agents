"""Scenario config schema, validation, and module loading for CrafText."""

import os
import importlib
from types import ModuleType
import flax.struct
import yaml
from dataclasses import dataclass

import pathlib
import inspect
import flax
from typing import Dict, Iterable, Mapping, Optional, Tuple, Type

import logging

from craftext.environment.craftext_constants import Scenarios
from craftext.environment.scenarious.checkers.target_state import TargetState
from craftext.environment.scenarious.scenario_types import ScenarioEntry, ScenarioMap

logger = logging.getLogger(__name__)

CONFIG_DIR_NAME = "configs"


@flax.struct.dataclass
class ScenariosConfig:
    """Parsed scenario configuration values.

    Attributes:
        dataset_key: Scenario dataset namespace used for module import.
        subset_key: Scenario subset key (e.g., split or difficulty key).
        base_environment: Craftax base environment descriptor.
        use_parafrases: Whether instruction paraphrases are enabled.
        test: Whether test instructions should be loaded.
        use_constraints_parafrases: Whether constrained paraphrases are enabled.
        world_preset: Optional world preset identifier associated with the config.
        dataset_package: Root package name where datasets are stored.
        config_path: Absolute path to the YAML config file.
    """
    dataset_key: str
    subset_key: str
    base_environment: str
    use_parafrases: bool
    test: bool
    use_constraints_parafrases: bool
    world_preset: Optional[str]
    dataset_package: str
    config_path: str

    @property
    def environment_family(self) -> str:
        """Return normalized environment family name."""
        return resolve_base_environment(self.base_environment).family

    @property
    def environment_key(self) -> int:
        """Return checker-state adapter key for the configured environment."""
        return resolve_base_environment(self.base_environment).environment_key


@dataclass(frozen=True)
class ResolvedBaseEnvironment:
    """Normalized description of a configured base environment."""

    raw_value: str
    normalized_value: str
    family: str
    environment_key: int


@dataclass(frozen=True)
class ConfigFieldSchema:
    """Validation schema for one config field.

    Attributes:
        expected_type: Expected Python type for the value.
        required: Whether the field is mandatory.
        default: Default value to use when field is optional and absent.
        non_empty: Whether a non-empty string is required.
    """

    expected_type: Type[object]
    required: bool = False
    default: Optional[object] = None
    non_empty: bool = False


CONFIG_SCHEMA: Dict[str, ConfigFieldSchema] = {
    "dataset_key": ConfigFieldSchema(str, required=True, non_empty=True),
    "subset_key": ConfigFieldSchema(str, required=True, non_empty=True),
    "base_environment": ConfigFieldSchema(str, required=False, default="Classic", non_empty=True),
    "use_parafrases": ConfigFieldSchema(bool, required=False, default=False),
    "test": ConfigFieldSchema(bool, required=False, default=False),
    "use_constraints_parafrases": ConfigFieldSchema(bool, required=False, default=False),
    "world_preset": ConfigFieldSchema(str, required=False, default=None, non_empty=True),
}

CONFIG_KEY_ALIASES = {
    "world_presets": "world_preset",
}


def resolve_base_environment(base_environment: str) -> ResolvedBaseEnvironment:
    """Normalize base-environment identifiers into stable family metadata."""
    normalized_value = base_environment.strip()
    lowered = normalized_value.lower()

    classic_markers = (
        "classic",
        "craftax-classic-",
    )
    is_classic = any(marker in lowered for marker in classic_markers)

    family = "classic" if is_classic else "full"
    environment_key = 0 if is_classic else 1
    return ResolvedBaseEnvironment(
        raw_value=base_environment,
        normalized_value=normalized_value,
        family=family,
        environment_key=environment_key,
    )


def _validate_and_normalize_config_data(
    config_data: Mapping[str, object],
    *,
    config_name: str,
    config_path: pathlib.Path,
) -> Dict[str, object]:
    """Validate raw YAML config mapping and normalize defaults.

    Args:
        config_data: Raw mapping loaded from YAML.
        config_name: Logical config name requested by caller.
        config_path: Absolute path to the YAML file.

    Returns:
        Dict[str, object]: Normalized config dictionary aligned with ``CONFIG_SCHEMA``.

    Raises:
        ValueError: If unknown keys are present, required keys are missing,
            or non-empty string constraints are violated.
        TypeError: If any field has invalid type.
    """
    remapped_config = {
        CONFIG_KEY_ALIASES.get(key, key): value for key, value in config_data.items()
    }

    unknown_keys = set(remapped_config.keys()) - set(CONFIG_SCHEMA.keys())
    if unknown_keys:
        raise ValueError(
            f"Configuration file {config_name} has unknown keys {sorted(unknown_keys)} "
            f"at {config_path}. Allowed keys: {sorted(CONFIG_SCHEMA.keys())}"
        )

    normalized: Dict[str, object] = {}
    for field_name, schema in CONFIG_SCHEMA.items():
        if field_name not in remapped_config:
            if schema.required:
                raise ValueError(
                    f"Configuration file {config_name} is missing required key '{field_name}' at {config_path}."
                )
            normalized[field_name] = schema.default
            continue

        value = remapped_config[field_name]

        if value is None:
            if schema.non_empty:
                raise ValueError(
                    f"Configuration file {config_name}: key '{field_name}' cannot be empty at {config_path}."
                )
            normalized[field_name] = None
            continue

        if schema.expected_type is bool:
            if not isinstance(value, bool):
                raise TypeError(
                    f"Configuration file {config_name}: key '{field_name}' must be bool, "
                    f"got {type(value).__name__} at {config_path}."
                )
        elif not isinstance(value, schema.expected_type):
            raise TypeError(
                f"Configuration file {config_name}: key '{field_name}' must be {schema.expected_type.__name__}, "
                f"got {type(value).__name__} at {config_path}."
            )

        if schema.non_empty and isinstance(value, str) and not value.strip():
            raise ValueError(
                f"Configuration file {config_name}: key '{field_name}' must be a non-empty string at {config_path}."
            )

        normalized[field_name] = value

    return normalized


def _find_config_path(module_name: str, config_name: str) -> pathlib.Path:
    """Resolve config name into concrete YAML path.

    Resolution strategy supports direct paths, underscore conventions,
    dotted paths, and unique filename fallback.

    Args:
        module_name: Dataset module path containing ``configs`` directory.
        config_name: User-provided config identifier.

    Returns:
        pathlib.Path: Resolved YAML config path.

    Raises:
        FileExistsError: If fallback filename search is ambiguous.
        FileNotFoundError: If no matching config file is found.
    """
    config_parts = config_name.split("_")
    module = importlib.import_module(module_name)
    module_path = pathlib.Path(module.__path__[0])
    configs_root = module_path.joinpath(CONFIG_DIR_NAME)

    # 0) direct relative path (allows slashes)
    if "/" in config_name:
        config_path = configs_root.joinpath(f"{config_name}.yaml")
        if config_path.exists():
            return config_path

    # 1) legacy underscore -> folder path
    config_path = configs_root.joinpath(f'{"/".join(config_parts)}.yaml')
    if config_path.exists():
        return config_path

    # 2) direct filename under configs root
    config_path = configs_root.joinpath(f'{config_name}.yaml')
    if config_path.exists():
        return config_path

    # 3) dot path support
    config_path = configs_root.joinpath(f'{config_name.replace(".", "/")}.yaml')
    if config_path.exists():
        return config_path

    # 4) fallback search by filename (unique)
    matches = list(configs_root.glob(f"**/{config_name}.yaml"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileExistsError(f"Multiple configs named {config_name}.yaml found: {matches}")

    raise FileNotFoundError(f"Configuration file {config_name} not found under {configs_root}.")


def _load_config_from_path(config_path: pathlib.Path, dataset_package: str, config_name: str) -> ScenariosConfig:
    """Load and validate one scenario config file.

    Args:
        config_path: Absolute path to YAML config file.
        dataset_package: Dataset package alias (e.g., ``craftext``).
        config_name: Logical config name requested by caller.

    Returns:
        ScenariosConfig: Structured and validated config object.

    Raises:
        ValueError: If config is empty or not a mapping.
        TypeError: If field types violate schema.
    """
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)

    if config_data is None:
        raise ValueError(f"Configuration file {config_name} is empty or not found.")
    elif not isinstance(config_data, dict):
        raise ValueError(f"Configuration file {config_name} is not a valid YAML dictionary.")
    normalized = _validate_and_normalize_config_data(
        config_data,
        config_name=config_name,
        config_path=config_path,
    )

    return ScenariosConfig(
        dataset_key=normalized["dataset_key"],
        subset_key=normalized["subset_key"],
        base_environment=normalized["base_environment"],
        use_parafrases=normalized["use_parafrases"],
        test=normalized["test"],
        use_constraints_parafrases=normalized["use_constraints_parafrases"],
        world_preset=normalized["world_preset"],
        dataset_package=dataset_package,
        config_path=str(config_path),
    )


class CraftextScenariosConfigLoader:
    """Public loader facade for CrafText scenario configs."""

    @staticmethod
    def get_config_path(config_name: str) -> pathlib.Path:
        """Resolve config name to path within CrafText dataset package.

        Args:
            config_name: User-visible config name.

        Returns:
            pathlib.Path: Absolute YAML path.
        """
        return _find_config_path("craftext.dataset", config_name)

    @staticmethod
    def load_config(config_name: str) -> ScenariosConfig:
        """Load and validate CrafText scenario config by name.

        Args:
            config_name: User-visible config name.

        Returns:
            ScenariosConfig: Parsed config dataclass.
        """
        config_path = CraftextScenariosConfigLoader.get_config_path(config_name)
        return _load_config_from_path(config_path, "craftext", config_name)



def get_default_scenario_path(dataset_package: str = "craftext"):
    """Return absolute scenarios directory for a dataset package.

    Args:
        dataset_package: Importable dataset package root.

    Returns:
        str: Absolute path to ``dataset/scenarious`` directory.
    """
    module = importlib.import_module(f"{dataset_package}.dataset")
    module_path = inspect.getmodule(module).__path__[0]
    return os.path.join(module_path, 'scenarious')


def _candidate_dataset_modules(dataset_key: str) -> Iterable[str]:
    # Support both naming styles without hardcoding concrete task keys:
    # - underscore style: building_line
    # - dotted package style: building.line
    candidates = [dataset_key]
    if "_" in dataset_key:
        candidates.append(dataset_key.replace("_", "."))
    if "." in dataset_key:
        candidates.append(dataset_key.replace(".", "_"))

    seen = set()
    for name in candidates:
        if name not in seen:
            seen.add(name)
            yield name


def _resolve_subset_key(module: ModuleType, data_key: str) -> str:
    if hasattr(module, data_key):
        return data_key

    raise AttributeError(
        f"Scenario module {module.__name__} has no key '{data_key}'."
    )


def _import_scenario_module(dataset_package: str, dataset_key: str, test: bool) -> Tuple[str, ModuleType]:
    module_candidates = []
    for mode in _candidate_dataset_modules(dataset_key):
        if test:
            module_candidates.append(f"{dataset_package}.dataset.scenarious.{mode}.test")
        module_candidates.append(f"{dataset_package}.dataset.scenarious.{mode}.instructions")

    attempted = []
    for module_name in module_candidates:
        try:
            return module_name, importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if not module_name.startswith(exc.name):
                raise
            attempted.append(module_name)

    raise ModuleNotFoundError(
        f"Could not resolve dataset_key '{dataset_key}'. Tried modules: {attempted}"
    )


def _validate_scenario_entry(name: str, entry: Mapping[str, object], module_name: str) -> ScenarioEntry:
    if "instruction" not in entry:
        raise ValueError(f"Scenario '{name}' in {module_name} is missing required key 'instruction'.")
    if "scenario_checker" not in entry:
        raise ValueError(f"Scenario '{name}' in {module_name} is missing required key 'scenario_checker'.")
    if "arguments" not in entry:
        raise ValueError(f"Scenario '{name}' in {module_name} is missing required key 'arguments'.")

    instruction = entry["instruction"]
    scenario_checker = entry["scenario_checker"]
    arguments = entry["arguments"]

    if not isinstance(instruction, str):
        raise TypeError(
            f"Scenario '{name}' in {module_name}: 'instruction' must be str, got {type(instruction).__name__}."
        )
    if not isinstance(scenario_checker, Scenarios):
        raise TypeError(
            f"Scenario '{name}' in {module_name}: 'scenario_checker' must be Scenarios, got {type(scenario_checker).__name__}."
        )
    if not isinstance(arguments, TargetState):
        raise TypeError(
            f"Scenario '{name}' in {module_name}: 'arguments' must be TargetState, got {type(arguments).__name__}."
        )

    raw_paraphrases = entry.get("instruction_paraphrases", [])
    if not isinstance(raw_paraphrases, list) or not all(isinstance(x, str) for x in raw_paraphrases):
        raise TypeError(
            f"Scenario '{name}' in {module_name}: 'instruction_paraphrases' must be list[str]."
        )

    typed_entry: ScenarioEntry = {
        "instruction": instruction,
        "scenario_checker": scenario_checker,
        "arguments": arguments,
        "instruction_paraphrases": raw_paraphrases,
    }

    str_check_lambda = entry.get("str_check_lambda")
    if str_check_lambda is not None:
        if not isinstance(str_check_lambda, str):
            raise TypeError(
                f"Scenario '{name}' in {module_name}: 'str_check_lambda' must be str when provided."
            )
        typed_entry["str_check_lambda"] = str_check_lambda

    return typed_entry


def _validate_scenario_map(raw_scenarios: object, module_name: str, subset_key: str) -> ScenarioMap:
    if not isinstance(raw_scenarios, dict):
        raise TypeError(
            f"Scenario subset '{subset_key}' in {module_name} must be dict[str, Mapping[str, object]]."
        )

    validated: ScenarioMap = {}
    for scenario_name, scenario_entry in raw_scenarios.items():
        if not isinstance(scenario_name, str):
            raise TypeError(
                f"Scenario key in {module_name}.{subset_key} must be str, got {type(scenario_name).__name__}."
            )
        if not isinstance(scenario_entry, Mapping):
            raise TypeError(
                f"Scenario '{scenario_name}' in {module_name}.{subset_key} must be mapping, got {type(scenario_entry).__name__}."
            )
        validated[scenario_name] = _validate_scenario_entry(scenario_name, scenario_entry, module_name)

    return validated


def load_scenarios(scenarious_config: ScenariosConfig) -> ScenarioMap:
    """Load scenario dictionary by validated configuration.

    Args:
        scenarious_config: Validated scenario config dataclass.

    Returns:
        ScenarioMap: Scenario dictionary from imported module keyed by subset.

    Raises:
        ValueError: If scenario root path cannot be resolved.
        AttributeError: If subset key is missing in scenario module.
    """
    scenarios: ScenarioMap = {}
    scenarios_dir = get_default_scenario_path(scenarious_config.dataset_package)

    data_key = scenarious_config.subset_key
    logger.info(
        "Loading scenarios from %s with dataset key '%s' and data key '%s'",
        scenarios_dir,
        scenarious_config.dataset_key,
        data_key,
    )

    if scenarios_dir is None:
        raise ValueError("Scenario path could not be determined.")

    scenario_module_name, scenario_module = _import_scenario_module(
        scenarious_config.dataset_package,
        scenarious_config.dataset_key,
        scenarious_config.test,
    )
    resolved_data_key = _resolve_subset_key(scenario_module, data_key)
    raw_subset = getattr(scenario_module, resolved_data_key)
    scenarios.update(_validate_scenario_map(raw_subset, scenario_module_name, resolved_data_key))

    return scenarios
