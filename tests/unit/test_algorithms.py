import jax.numpy as jnp
from tunix_craftext.algorithms import generalized_advantage_estimation
from tunix_craftext.algorithms import ppo_loss
def test_gae_matches_hand_computed_one_step_returns() -> None:
    advantages, returns = generalized_advantage_estimation(jnp.array([[1.],[2.]]), jnp.zeros((2,1)), jnp.array([3.]), jnp.zeros((2,1)), 1.0, 1.0)
    assert advantages[:,0].tolist() == [6.0,5.0]
    assert returns[:,0].tolist() == [6.0,5.0]

def test_ppo_loss_is_finite_for_hand_computed_minibatch() -> None:
    loss, metrics = ppo_loss(jnp.array([0.,0.]), jnp.array([0.,0.]), jnp.array([1.,-1.]), jnp.zeros(2), jnp.zeros(2), jnp.array([1.,-1.]), .2, .5, jnp.ones(2), .01)
    assert float(loss) > 0
    assert float(metrics["approx_kl"]) == 0
