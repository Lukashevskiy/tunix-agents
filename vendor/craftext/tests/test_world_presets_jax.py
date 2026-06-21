"""Regression tests for JAX transformations of vendored world-preset behaviors."""

import jax
import jax.numpy as jnp

from craftext.environment.world_presets import build_env_and_params, build_world_preset_spec


def test_tiny_preset_step_supports_vmap_over_two_independent_envs() -> None:
    """Preset behaviors must not use Python scalar conversion or branches on JAX tracers."""
    spec = build_world_preset_spec(
        env_name="Craftax-Classic-Pixels-v1",
        preset_name="tiny_box_oob_no_mobs",
        seed=7,
    )
    environment, params = build_env_and_params(spec, auto_reset=False)
    keys = jax.random.split(jax.random.PRNGKey(7), 2)
    _, states = jax.vmap(environment.reset, in_axes=(0, None))(keys, params)
    step_keys = jax.random.split(jax.random.PRNGKey(8), 2)

    _, _, reward, done, _ = jax.vmap(environment.step, in_axes=(0, 0, 0, None))(
        step_keys, states, jnp.zeros(2, dtype=jnp.int32), params
    )

    assert reward.shape == (2,)
    assert done.shape == (2,)
