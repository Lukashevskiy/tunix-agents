import jax

from typing import Union
from craftext.environment.states.state_classic import GameDataClassic


def checker_budget_drink_level(game_data: GameDataClassic,  level) -> jax.Array:
    # raise NotImplementedError("checker_budget_build_collect is not implemented yet")
    return level_status(game_data, level)

def level_status(game_data: GameDataClassic, level: int):
    current_level = game_data.states[0].variables.player_drink

    # def get_item(index, inventory):
    #     leaves, _ = tree_util.tree_flatten(inventory)
    #     leaves = jnp.stack(leaves)
    #     return leaves[index]
    
    
    
    
    # curr = get_item(block_type, game_data.states[0].inventory)
    # prev = get_item(block_type, game_data.states[1].inventory)
    return current_level < level


