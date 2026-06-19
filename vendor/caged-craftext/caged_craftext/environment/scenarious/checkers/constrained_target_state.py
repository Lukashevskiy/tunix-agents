from craftext.environment.craftext_constants import BlockType, MobType
from craftext.environment.scenarious.checkers.target_state import TargetState
from craftext.environment.states.state_classic import Action
from enum import Enum

import jax
from jax import numpy as jnp
from flax import struct

@struct.dataclass
class StepOnBlock:
    block_type: int = BlockType.INVALID.value

@struct.dataclass
class IntrinsicState:
    level: int = 100

@struct.dataclass
class EnergyLevelState:
    level: int = 100

@struct.dataclass
class LightLevelState:
    level: float = 0

@struct.dataclass
class BuildBudgetState:
    block_type: int = BlockType.INVALID.value
    inventory_block_capacity: int = 0

class TargetAction(Enum):
    FOOD = 0
    WOOD = 1

@struct.dataclass
class BudgetByAction:
    target_action: int = 0
    budget: int = 100
    delimiter_actions: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros((len(Action.__dict__.keys()),), dtype=jnp.float32)
    )

@struct.dataclass
class AvoidMobDistance:
    mob: int = struct.field(default_factory=lambda: MobType.ZOMBIE.value)
    distance: int = struct.field(default_factory=int)
    
class TypeOfInterest(Enum):
    FOOD = 0
    WATER = 1
    TREE = 2
    STONE = 3
    
@struct.dataclass
class TargetOfInterest:
    object_of_interest: int = struct.field(default_factory=lambda: TypeOfInterest.FOOD.value)
    far_from_agent: int = struct.field(default_factory=lambda: 5)
    last_visible_target_position: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros(shape=(2,), dtype=jnp.int32)
    )

@struct.dataclass
class SeeInView:
    block_type: int = struct.field(default_factory=lambda: BlockType.GRASS.value)
        

@struct.dataclass
class ConstrainedTargetState(TargetState):
    step_on_block: StepOnBlock =  struct.field(default_factory=StepOnBlock)
    
    drink_level_state:  IntrinsicState = struct.field(default_factory=IntrinsicState)
    energy_level_state: IntrinsicState = struct.field(default_factory=IntrinsicState)
    hp_level_state:     IntrinsicState = struct.field(default_factory=IntrinsicState)
    hungry_level_state: IntrinsicState = struct.field(default_factory=IntrinsicState)
    away_monsters_hp_level_state: IntrinsicState = struct.field(default_factory=IntrinsicState)
    
    build_budget_state: BuildBudgetState = struct.field(default_factory=BuildBudgetState)
    budget_by_action:   BudgetByAction   = struct.field(default_factory=BudgetByAction) 
    budget_food_by_action: BudgetByAction = struct.field(default_factory=BudgetByAction)
    budget_wood_by_action: BudgetByAction = struct.field(default_factory=BudgetByAction)
    
    night_constraint_level: IntrinsicState = struct.field(default_factory=IntrinsicState)
    day_constraint_level:   IntrinsicState = struct.field(default_factory=IntrinsicState)

    target_of_interest: TargetOfInterest = struct.field(default_factory=TargetOfInterest)
    target_of_interest_food: TargetOfInterest = struct.field(default_factory=TargetOfInterest)
    target_of_interest_water: TargetOfInterest = struct.field(default_factory=TargetOfInterest)
    avoid_mob_distance: AvoidMobDistance = struct.field(default_factory=AvoidMobDistance)

    see_in_fov: SeeInView = struct.field(default_factory=SeeInView)
