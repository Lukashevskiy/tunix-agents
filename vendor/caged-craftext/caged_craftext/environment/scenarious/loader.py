"""Config resolution and scenario loading helpers for Caged CrafText."""

import importlib
import pathlib
import yaml

from craftext.environment.scenarious.loader import ScenariosConfig, _load_config_from_path, CONFIG_DIR_NAME


def _find_caged_config_path(config_name: str) -> pathlib.Path:
    """Resolve caged config name into concrete YAML file path.

    Args:
        config_name: User-facing config identifier.

    Returns:
        pathlib.Path: Absolute path to the matched YAML file.

    Raises:
        FileExistsError: If fallback filename search is ambiguous.
        FileNotFoundError: If no config file matches.
    """
    # Support underscore-structured names: a/b/c/name -> a_b_c_name
    if "/" not in config_name and "_" in config_name:
        parts = config_name.split("_")
        if len(parts) >= 4:
            module = importlib.import_module("caged_craftext.dataset")
            module_path = pathlib.Path(module.__path__[0])
            configs_root = module_path.joinpath(CONFIG_DIR_NAME)
            rel_dir = "/".join(parts[:3])
            filename = "_".join(parts[3:])
            candidate = configs_root.joinpath(rel_dir, f"{filename}.yaml")
            if candidate.exists():
                return candidate

    module = importlib.import_module("caged_craftext.dataset")
    module_path = pathlib.Path(module.__path__[0])
    configs_root = module_path.joinpath(CONFIG_DIR_NAME)

    # direct relative path (allows slashes)
    if "/" in config_name:
        config_path = configs_root.joinpath(f"{config_name}.yaml")
        if config_path.exists():
            return config_path

    # legacy underscore -> folder path
    config_parts = config_name.split("_")
    config_path = configs_root.joinpath(f'{"/".join(config_parts)}.yaml')
    if config_path.exists():
        return config_path

    # direct filename under configs root
    config_path = configs_root.joinpath(f'{config_name}.yaml')
    if config_path.exists():
        return config_path

    # dot path support
    config_path = configs_root.joinpath(f'{config_name.replace(".", "/")}.yaml')
    if config_path.exists():
        return config_path

    # fallback search by filename (unique)
    matches = list(configs_root.glob(f"**/{config_name}.yaml"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileExistsError(f"Multiple configs named {config_name}.yaml found: {matches}")

    raise FileNotFoundError(f"Configuration file {config_name} not found under {configs_root}.")


class CagedCraftextScenariosConfigLoader:
    """Public loader facade for Caged CrafText config files."""

    @staticmethod
    def get_config_path(config_name: str) -> pathlib.Path:
        """Resolve caged config name into YAML path.

        Args:
            config_name: User-facing config identifier.

        Returns:
            pathlib.Path: Absolute path to config file.
        """
        return _find_caged_config_path(config_name)

    @staticmethod
    def load_config(config_name: str) -> ScenariosConfig:
        """Load and validate caged config by name.

        Args:
            config_name: User-facing config identifier.

        Returns:
            ScenariosConfig: Parsed validated configuration.
        """
        config_path = CagedCraftextScenariosConfigLoader.get_config_path(config_name)
        return _load_config_from_path(config_path, "caged_craftext", config_name)


def load_caged_scenarios(scenarious_config: ScenariosConfig) -> dict:
    """Load caged scenarios strictly from config path and dataset key.

    Args:
        scenarious_config: Validated scenario configuration.

    Returns:
        dict: Scenario dictionary for configured subset.

    Raises:
        AttributeError: If requested subset key is absent in scenario module.
    """
    scenarios = {}
    module = "test" if scenarious_config.test else "instructions"
    data_key = scenarious_config.subset_key

    config_path = pathlib.Path(scenarious_config.config_path)
    if CONFIG_DIR_NAME in config_path.parts:
        rel_parts = config_path.parts[config_path.parts.index(CONFIG_DIR_NAME) + 1: -1]
        rel_dir = ".".join(rel_parts) if rel_parts else ""
    else:
        rel_dir = ""

    if rel_dir:
        module_path = f"{rel_dir}.{scenarious_config.dataset_key}"
    else:
        module_path = scenarious_config.dataset_key

    scenario_module_name = f"{scenarious_config.dataset_package}.dataset.scenarious.{module_path}.{module}"
    scenario_module = importlib.import_module(scenario_module_name)

    if hasattr(scenario_module, data_key):
        scenarios.update(getattr(scenario_module, data_key))
    else:
        raise AttributeError(f"Scenario module {scenario_module_name} has no key '{data_key}'.")

    return scenarios
