import jax

from jax import (
    numpy as jnp,
    lax
)
from jax import tree_util

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from caged_craftext.environment.scenarious.checkers.constrained_target_state import BuildBudgetState
from caged_craftext.environment.scenarious.checkers.constrained_target_state import ConstrainedTargetState

CMDPTargetState = ConstrainedTargetState

def checker_budget_build_collect(game_data: Union[GameDataClassic, GameData],  target_state: BuildBudgetState) -> jax.Array:
    # raise NotImplementedError("checker_budget_build_collect is not implemented yet")
    block_type = target_state.block_type
    return new_item(game_data, block_type)

def new_item(game_data: Union[GameDataClassic, GameData], block_type: int):
    block_type  = block_type

    def get_item(index, inventory):
        leaves, _ = tree_util.tree_flatten(inventory)
        leaves = jnp.stack(leaves)
        return leaves[index]
    
    
    
    
    curr = get_item(block_type, game_data.states[0].inventory)
    prev = get_item(block_type, game_data.states[1].inventory)
    return (curr - prev) > 0

