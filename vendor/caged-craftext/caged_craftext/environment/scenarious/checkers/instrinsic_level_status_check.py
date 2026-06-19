from craftext.environment.states.state_classic import GameDataClassic



def level_status(game_data: GameDataClassic, level: int):
    current_level = game_data.states[0].variables.player_health

    # def get_item(index, inventory):
    #     leaves, _ = tree_util.tree_flatten(inventory)
    #     leaves = jnp.stack(leaves)
    #     return leaves[index]
    
    
    
    
    # curr = get_item(block_type, game_data.states[0].inventory)
    # prev = get_item(block_type, game_data.states[1].inventory)
    return current_level < level


