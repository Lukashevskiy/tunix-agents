import jax
import jax.numpy as jnp
from tunix_craftext.learner import create_state, ppo_update
def test_ppo_update_is_finite_and_changes_parameters() -> None:
    state=create_state(jax.random.PRNGKey(0),3,2)
    updated,metrics=ppo_update(state,jnp.ones((4,3)),jnp.zeros(4,dtype=jnp.int32),jnp.zeros(4),jnp.ones(4),jnp.ones(4))
    assert bool(jnp.isfinite(metrics["loss"]))
    assert not bool(jnp.allclose(state.params["Dense_0"]["kernel"],updated.params["Dense_0"]["kernel"]))
