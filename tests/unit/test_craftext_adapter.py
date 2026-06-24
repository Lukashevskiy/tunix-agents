from types import SimpleNamespace

import jax
import numpy as np
import pytest

from tests.fixtures.tiny_craftext import TinyCrafText
from tunix_craftext.adapters import (
    AdapterContractError,
    CagedCrafTextAdapter,
    CraftaxAdapter,
    CrafTextAdapter,
)


def test_reset_and_step_normalize_craftext_done_and_static_action_mask() -> None:
    adapter = CrafTextAdapter(TinyCrafText(terminal_timestep=2), params=object(), action_count=3)

    reset = adapter.reset(key=7)
    first = adapter.step(key=8, state=reset.state, action=2)
    second = adapter.step(key=9, state=first.state, action=1)

    np.testing.assert_array_equal(reset.observation, [7, 0])
    assert isinstance(reset.action_mask, jax.Array)
    assert isinstance(first.reward, jax.Array)
    assert isinstance(first.terminated, jax.Array)
    np.testing.assert_array_equal(reset.action_mask, [True, True, True])
    np.testing.assert_array_equal(first.terminated, False)
    np.testing.assert_array_equal(first.truncated, False)
    np.testing.assert_array_equal(first.action_mask, [True, True, True])
    np.testing.assert_array_equal(second.terminated, True)
    np.testing.assert_array_equal(second.truncated, False)


def test_caged_adapter_has_identical_transition_contract() -> None:
    adapter = CagedCrafTextAdapter(
        TinyCrafText(terminal_timestep=1, caged=True), params=object(), action_count=3
    )

    reset = adapter.reset(key=0)
    transition = adapter.step(key=1, state=reset.state, action=0)

    np.testing.assert_array_equal(transition.reward, 1.0)
    np.testing.assert_array_equal(transition.terminated, True)
    np.testing.assert_array_equal(transition.action_mask, [True, True, False])


def test_adapter_rejects_a_vendor_action_mask_with_dynamic_wrong_shape() -> None:
    class BadMaskEnvironment(TinyCrafText):
        def step(self, key: int, state, action: int, params: object):
            observation, next_state, reward, done, info = super().step(key, state, action, params)
            return observation, next_state, reward, done, {"action_mask": np.ones(2, dtype=bool)}

    adapter = CrafTextAdapter(
        BadMaskEnvironment(terminal_timestep=1), params=object(), action_count=3
    )
    with pytest.raises(AdapterContractError, match="shape"):
        adapter.step(key=0, state=adapter.reset(key=0).state, action=0)


def test_craftext_context_preserves_instruction_world_and_underlying_env_state() -> None:
    adapter = CrafTextAdapter(
        TinyCrafText(terminal_timestep=1),
        params=object(),
        action_count=3,
        world_preset="tiny_box_oob_no_mobs",
        instructions=("collect wood", "find water"),
        instruction_index=0,
    )
    state = SimpleNamespace(idx=np.asarray(1), env_state={"inventory": "wood"})

    context = adapter.episode_context(state)

    assert adapter.has_instruction_context
    assert context.world_preset == "tiny_box_oob_no_mobs"
    assert context.instruction == "find water"
    assert context.env_state == {"inventory": "wood"}
    assert context.text_constraint == ""


def test_caged_context_aligns_text_constraint_with_selected_instruction() -> None:
    adapter = CagedCrafTextAdapter(
        TinyCrafText(terminal_timestep=1, caged=True),
        params=object(),
        action_count=3,
        world_preset="default",
        instructions=("survive", "collect wood"),
        text_constraints=("do not lose health", "do not attack"),
    )

    context = adapter.episode_context(SimpleNamespace(idx=np.asarray(0), env_state="state"))

    assert context.instruction == "survive"
    assert context.text_constraint == "do not lose health"


def test_craftax_adapter_stays_free_of_craftext_instruction_metadata() -> None:
    adapter = CraftaxAdapter(TinyCrafText(terminal_timestep=1), params=object(), action_count=3)

    assert not hasattr(adapter, "episode_context")
