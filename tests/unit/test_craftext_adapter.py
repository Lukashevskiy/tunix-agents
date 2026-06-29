from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.fixtures.tiny_craftext import TinyCrafText
from tunix_craftext.adapters import (
    AdapterContractError,
    CagedCrafTextAdapter,
    CraftaxAdapter,
    CrafTextAdapter,
)


class InstructionTinyCrafText(TinyCrafText):
    def __init__(self) -> None:
        super().__init__(terminal_timestep=1)
        self.last_instruction_idx: int | None = None

    def reset(
        self, key: int, params: object, *, instruction_idx: int | None = None
    ) -> tuple[np.ndarray, SimpleNamespace]:
        self.last_instruction_idx = instruction_idx
        observation, state = super().reset(key, params)
        return observation, SimpleNamespace(idx=np.asarray(instruction_idx or 0), env_state=state)


class JaxTinyCrafText:
    def reset(self, key: jax.Array, params: object) -> tuple[jax.Array, jax.Array]:
        del params
        return jnp.asarray([key, 0], dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)

    def step(
        self, key: jax.Array, state: jax.Array, action: jax.Array, params: object
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key, params
        next_state = state + 1
        return (
            jnp.asarray([next_state, action], dtype=jnp.int32),
            next_state,
            action.astype(jnp.float32),
            next_state >= 2,
            {"action_mask": jnp.asarray([True, True, False])},
        )


def test_craftax_adapter_rejects_invalid_action_count() -> None:
    with pytest.raises(AdapterContractError, match="action_count"):
        CraftaxAdapter(TinyCrafText(terminal_timestep=1), params=object(), action_count=0)


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


def test_craftax_adapter_supports_custom_action_mask_key_and_fallback_mask() -> None:
    class CustomMaskEnvironment(TinyCrafText):
        def step(self, key: int, state, action: int, params: object):
            observation, next_state, reward, done, _ = super().step(key, state, action, params)
            return observation, next_state, reward, done, {"legal": np.asarray([False, True])}

    adapter = CraftaxAdapter(
        CustomMaskEnvironment(terminal_timestep=1),
        params=object(),
        action_count=2,
        action_mask_key="legal",
    )

    reset = adapter.reset(key=1)
    transition = adapter.step(key=2, state=reset.state, action=1)

    np.testing.assert_array_equal(reset.action_mask, [True, True])
    np.testing.assert_array_equal(transition.action_mask, [False, True])


def test_craftax_adapter_step_is_vmap_compatible_for_jax_env() -> None:
    adapter = CraftaxAdapter(JaxTinyCrafText(), params=object(), action_count=3)

    keys = jnp.asarray([1, 2], dtype=jnp.int32)
    reset = jax.vmap(adapter.reset)(keys)
    transition = jax.vmap(adapter.step)(
        keys,
        reset.state,
        jnp.asarray([0, 1], dtype=jnp.int32),
    )

    assert reset.observation.shape == (2, 2)
    assert transition.observation.shape == (2, 2)
    np.testing.assert_array_equal(
        transition.action_mask,
        [[True, True, False], [True, True, False]],
    )


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


def test_craftext_context_requires_configured_instruction_metadata() -> None:
    adapter = CrafTextAdapter(TinyCrafText(terminal_timestep=1), params=object(), action_count=3)

    assert not adapter.has_instruction_context
    with pytest.raises(AdapterContractError, match="instruction metadata"):
        adapter.episode_context(SimpleNamespace(idx=np.asarray(0), env_state="state"))


def test_craftext_reset_binds_configured_instruction_index() -> None:
    environment = InstructionTinyCrafText()
    adapter = CrafTextAdapter(
        environment,
        params=object(),
        action_count=3,
        instructions=("collect wood", "find water"),
        instruction_index=1,
    )

    reset = adapter.reset(key=5)
    context = adapter.episode_context(reset.state)

    assert environment.last_instruction_idx == 1
    assert context.instruction == "find water"
    assert adapter.prompt_state(reset.state).timestep == 0


def test_craftext_reset_with_instruction_overrides_runtime_default() -> None:
    environment = InstructionTinyCrafText()
    adapter = CrafTextAdapter(
        environment,
        params=object(),
        action_count=3,
        instructions=("collect wood", "find water"),
        instruction_index=0,
    )

    reset = adapter.reset_with_instruction(key=5, instruction_index=1)
    context = adapter.episode_context(reset.state)

    assert adapter.instructions == ("collect wood", "find water")
    assert environment.last_instruction_idx == 1
    assert context.instruction == "find water"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"instruction_index": -1}, "non-negative"),
        (
            {"instructions": ("only",), "instruction_index": 1},
            "one configured instruction",
        ),
    ],
)
def test_craftext_adapter_rejects_invalid_instruction_index(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(AdapterContractError, match=message):
        CrafTextAdapter(
            TinyCrafText(terminal_timestep=1),
            params=object(),
            action_count=3,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (SimpleNamespace(env_state="state"), "instruction idx"),
        (SimpleNamespace(idx=np.asarray(3), env_state="state"), "outside configured rows"),
    ],
)
def test_craftext_context_rejects_missing_or_out_of_range_state_idx(
    state: object, message: str
) -> None:
    adapter = CrafTextAdapter(
        TinyCrafText(terminal_timestep=1),
        params=object(),
        action_count=3,
        instructions=("survive",),
    )

    with pytest.raises(AdapterContractError, match=message):
        adapter.episode_context(state)


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


def test_caged_adapter_rejects_misaligned_text_constraints() -> None:
    with pytest.raises(AdapterContractError, match="one-to-one"):
        CagedCrafTextAdapter(
            TinyCrafText(terminal_timestep=1, caged=True),
            params=object(),
            action_count=3,
            instructions=("survive", "collect wood"),
            text_constraints=("do not lose health",),
        )


def test_craftax_adapter_stays_free_of_craftext_instruction_metadata() -> None:
    adapter = CraftaxAdapter(TinyCrafText(terminal_timestep=1), params=object(), action_count=3)

    reset = adapter.reset(key=0)

    assert adapter.world_preset == ""
    assert not adapter.has_instruction_context
    assert adapter.prompt_state(reset.state) is reset.state
    with pytest.raises(AdapterContractError, match="instruction metadata"):
        adapter.episode_context(reset.state)
