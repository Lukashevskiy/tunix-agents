import jax.numpy as jnp

from tunix_craftext.algorithms import (
    generalized_advantage_estimation,
    masked_token_ppo_loss,
    masked_token_returns,
    ppo_loss,
)


def test_gae_matches_hand_computed_one_step_returns() -> None:
    advantages, returns = generalized_advantage_estimation(
        jnp.array([[1.0], [2.0]]), jnp.zeros((2, 1)), jnp.array([3.0]), jnp.zeros((2, 1)), 1.0, 1.0
    )
    assert advantages[:, 0].tolist() == [6.0, 5.0]
    assert returns[:, 0].tolist() == [6.0, 5.0]


def test_ppo_loss_is_finite_for_hand_computed_minibatch() -> None:
    loss, metrics = ppo_loss(
        jnp.array([0.0, 0.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([1.0, -1.0]),
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.array([1.0, -1.0]),
        0.2,
        0.5,
        jnp.ones(2),
        0.01,
    )
    assert float(loss) > 0
    assert float(metrics["approx_kl"]) == 0


def test_masked_token_returns_puts_terminal_reward_on_all_valid_completion_tokens() -> None:
    rewards = jnp.array([[0.0, 2.0, 0.0], [0.0, -1.0, 0.0]])
    mask = jnp.array([[True, True, False], [True, True, False]])

    returns = masked_token_returns(rewards, mask, gamma=0.5)

    assert jnp.array_equal(returns, jnp.array([[1.0, 2.0, 0.0], [-0.5, -1.0, 0.0]]))


def test_masked_token_ppo_loss_ignores_padding() -> None:
    valid = jnp.array([[True, True, False]])
    loss, metrics = masked_token_ppo_loss(
        new_log_prob=jnp.array([[-0.5, -0.5, 100.0]]),
        old_log_prob=jnp.array([[-0.5, -0.5, -100.0]]),
        advantages=jnp.array([[1.0, 1.0, 999.0]]),
        new_value=jnp.zeros((1, 3)),
        old_value=jnp.zeros((1, 3)),
        returns=jnp.zeros((1, 3)),
        token_mask=valid,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy=jnp.ones((1, 3)),
        entropy_coefficient=0.01,
    )

    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["policy_loss"])
