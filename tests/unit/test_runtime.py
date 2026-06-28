"""Unit tests for CrafText runtime construction without vendor dependencies."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from tunix_craftext.adapters import CagedCrafTextAdapter, CrafTextAdapter
from tunix_craftext.env.runtime import RuntimeError, build_craftext_runtime


@dataclass
class _FakeSpec:
    name: str


class _FakeEnvironment:
    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions


def _config(
    implementation: str,
    *,
    world_preset: str = "tiny",
    instruction_index: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        environment=SimpleNamespace(
            implementation=implementation,
            base_environment="craftax",
            world_preset=world_preset,
            scenario_config="wood_achievements_with_energy",
            instruction_index=instruction_index,
        ),
        run=SimpleNamespace(seed=13),
    )


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_craftax_actions(monkeypatch: pytest.MonkeyPatch, labels: tuple[str, ...]) -> None:
    _install_module(monkeypatch, "craftax")
    _install_module(monkeypatch, "craftax.craftax_classic")
    _install_module(
        monkeypatch,
        "craftax.craftax_classic.constants",
        Action=tuple(SimpleNamespace(name=label) for label in labels),
    )


def _install_craftext_vendor(monkeypatch: pytest.MonkeyPatch, *, action_count: int = 3) -> None:
    class RawInstructionWrapper:
        def __init__(self, environment: object, *, scenario_handler: object) -> None:
            self.environment = environment
            self.scenario_handler = scenario_handler

    class JaxScenarioDataHandler:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.scenario_data = SimpleNamespace(
                instructions_list=("collect wood", "avoid monster")
            )

    def build_world_preset_spec(
        *, env_name: str, preset_name: str, seed: int
    ) -> _FakeSpec:
        assert (env_name, seed) == ("craftax", 13)
        return _FakeSpec(preset_name)

    def build_env_and_params(spec: _FakeSpec, *, auto_reset: bool) -> tuple[object, object]:
        assert spec.name == "tiny"
        assert auto_reset is False
        return _FakeEnvironment(action_count), {"preset": spec.name}

    _install_module(monkeypatch, "craftext")
    _install_module(monkeypatch, "craftext.environment")
    _install_module(
        monkeypatch,
        "craftext.environment.craftext_wrapper",
        RawInstructionWrapper=RawInstructionWrapper,
    )
    _install_module(monkeypatch, "craftext.environment.scenarious")
    _install_module(
        monkeypatch,
        "craftext.environment.scenarious.manager",
        DefaultInstructionTransformer=object(),
        DefaultJAXRepresentation=object(),
        JaxScenarioDataHandler=JaxScenarioDataHandler,
    )
    _install_module(
        monkeypatch,
        "craftext.environment.scenarious.processors",
        RawProcessor=object(),
    )
    _install_module(
        monkeypatch,
        "craftext.environment.world_presets",
        build_env_and_params=build_env_and_params,
        build_world_preset_spec=build_world_preset_spec,
    )


def _install_caged_vendor(monkeypatch: pytest.MonkeyPatch, *, action_count: int = 3) -> None:
    class CMDPInstructionWrapper:
        def __init__(self, environment: object, *, config_name: str) -> None:
            self.environment = environment
            self.config_name = config_name
            self.scenario_handler = SimpleNamespace(
                scenario_data=SimpleNamespace(
                    instructions_list=("collect wood",),
                    texutal_constraints_list=("keep energy positive",),
                )
            )

    def build_world_preset_spec(
        *, env_name: str, preset_name: str, seed: int
    ) -> _FakeSpec:
        assert (env_name, seed) == ("craftax", 13)
        return _FakeSpec(preset_name)

    def build_env_and_params(spec: _FakeSpec, *, auto_reset: bool) -> tuple[object, object]:
        assert spec.name == "tiny"
        assert auto_reset is False
        return _FakeEnvironment(action_count), {"preset": spec.name}

    _install_module(monkeypatch, "caged_craftext")
    _install_module(monkeypatch, "caged_craftext.environment")
    _install_module(
        monkeypatch,
        "caged_craftext.environment.caged_craftext_wrapper",
        CMDPInstructionWrapper=CMDPInstructionWrapper,
    )
    _install_module(
        monkeypatch,
        "caged_craftext.environment.world_presets",
        build_env_and_params=build_env_and_params,
        build_world_preset_spec=build_world_preset_spec,
    )


def test_build_craftext_runtime_uses_raw_instruction_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_craftext_vendor(monkeypatch)
    _install_craftax_actions(monkeypatch, ("LEFT", "RIGHT", "DO"))

    runtime = build_craftext_runtime(_config("craftext"))

    assert isinstance(runtime.adapter, CrafTextAdapter)
    assert not isinstance(runtime.adapter, CagedCrafTextAdapter)
    assert runtime.action_count == 3
    assert runtime.actions.labels == ("LEFT", "RIGHT", "DO")
    assert runtime.adapter.world_preset == "tiny"
    assert runtime.adapter.has_instruction_context


def test_build_craftext_runtime_uses_caged_constraint_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_caged_vendor(monkeypatch)
    _install_craftax_actions(monkeypatch, ("LEFT", "RIGHT", "DO"))

    runtime = build_craftext_runtime(_config("caged-craftext"))

    assert isinstance(runtime.adapter, CagedCrafTextAdapter)
    assert runtime.action_count == 3
    assert runtime.actions.labels == ("LEFT", "RIGHT", "DO")


def test_build_craftext_runtime_rejects_unsupported_implementation() -> None:
    with pytest.raises(RuntimeError, match="unsupported environment"):
        build_craftext_runtime(_config("craftax"))


@pytest.mark.parametrize("action_count", [0, "three"])
def test_build_craftext_runtime_rejects_invalid_vendor_action_count(
    monkeypatch: pytest.MonkeyPatch, action_count: object
) -> None:
    _install_craftext_vendor(monkeypatch, action_count=action_count)  # type: ignore[arg-type]
    _install_craftax_actions(monkeypatch, ("LEFT", "RIGHT", "DO"))

    with pytest.raises(RuntimeError, match="positive integer num_actions"):
        build_craftext_runtime(_config("craftext"))


def test_build_craftext_runtime_rejects_action_enum_cardinality_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_craftext_vendor(monkeypatch, action_count=3)
    _install_craftax_actions(monkeypatch, ("LEFT", "RIGHT"))

    with pytest.raises(RuntimeError, match="Action enum"):
        build_craftext_runtime(_config("craftext"))
