import jax
import jax.numpy as jnp

from craftext.environment.states.state_classic import GameDataClassic

from caged_craftext.environment.scenarious.checkers.constrained_target_state import StepOnBlock

def checker_step_on_block(game_data: GameDataClassic,  target_state: StepOnBlock) -> jax.Array:
    block_type  = target_state.block_type
    
    return is_on_block(game_data, block_type)

def is_on_block(gd: GameDataClassic, block_type):
    x, y = gd.states[0].variables.player_position
    game_map = gd.states[0].map.game_map
    return jnp.array_equal(game_map[x][y], block_type)


