"""Constraint-aware instruction wrapper for Caged CrafText episodes."""

from typing import Any, Dict, Generic, Optional, Tuple, Type
import jax
from jax import Array
from flax import struct

from craftext.environment.craftext_wrapper import (
    BaseInstructionWrapper,
    EnvStateT,
    JaxEnvProtocol,
    ObsT,
    TextEnvState,
)
from craftext.environment.scenarious.processors import RawProcessor

from .scenarious.manager import create_scenarios_with_dataset
from .scenarious.checkers.step_on_block    import checker_step_on_block
from .scenarious.checkers.budget_build_collect import checker_budget_build_collect
from .scenarious.checkers.dont_move import checker_moveing_at_night_level
from .scenarious.checkers.sleep_at_night import checker_sleep_at_night
from .scenarious.checkers.monster_is_attacked_without_sword import checker_monster_is_attacked_without_sword
from .scenarious.checkers.away_from_monsters_when_hp_low import checker_away_from_monsters_when_hp_low
from .scenarious.checkers.dont_sleep_near_monsters import checker_dont_sleep_near_monsters
from .scenarious.checkers.drink_level import checker_budget_drink_level
from .scenarious.checkers.hp_level import checker_budget_hp_level
from .scenarious.checkers.hungry_level import checker_budget_hungry_level
from .scenarious.checkers.energy_level import checker_budget_energy_level
from .scenarious.checkers.avoid_mob_distance import checker_relactional_avoid_mob_distance
from .scenarious.checkers.budget_by_action import checker_budget_by_action
from .scenarious.checkers.last_visible_target import checker_last_visible_target
from .scenarious.checkers.constrained_target_state import ConstrainedTargetState 

CMDPScenarioHandler = create_scenarios_with_dataset(use_plans_gpt=False)

@struct.dataclass
class TextEnvStateWithConstraint(TextEnvState[EnvStateT], Generic[EnvStateT]):
    cost_type: int
    cost: float
    episode_cost: float
    target_state: ConstrainedTargetState

class CMDPInstructionWrapper(BaseInstructionWrapper[ObsT, TextEnvStateWithConstraint, Dict[str, Any]]):
    def __init__(
        self,
        env: JaxEnvProtocol[ObsT, EnvStateT, Dict[str, Any]],
        config_name: Optional[str] = None,
        scenario_handler_class: Type = CMDPScenarioHandler,
        scenario_handler: Optional[Any] = None,
        scenario_processor: Type = RawProcessor,
    ) -> None:
        if scenario_handler is None:
            if config_name is None:
                config_name = "achievements_wood"
            scenario_handler = scenario_handler_class(scenario_processor, config_name)
        super().__init__(env, scenario_handler=scenario_handler)
        # self.encoded_textual_constraint = self.scenario_handler.scenario_data_jax.constraints_embeddings_list[0]

    def _get_instruction(self, idx: int) -> Tuple[Optional[Array], str]:
        instruction_text = self.scenario_handler.scenario_data.instructions_list[idx]
        return None, instruction_text

    def reset(
        self,
        _rng: Array,
        env_params: Any,
        instruction_idx: int = -1,
    ) -> Tuple[ObsT, TextEnvStateWithConstraint[EnvStateT]]:
        obs, state = super().reset(_rng, env_params, instruction_idx=instruction_idx)

        cost_type = self.scenario_handler.scenario_data_jax.cost_types[state.idx]
        state = TextEnvStateWithConstraint(
            env_state=state.env_state,
            timestep=state.timestep,
            cost_type=cost_type,
            idx=state.idx,
            success_rate=state.success_rate,
            episode_cost=0.,
            cost=0.,
            total_success_rate=state.total_success_rate,
            rng=state.rng,
            instruction_done=state.instruction_done,
            checker_id=state.checker_id,
            target_state=state.target_state
        )
        return obs, state

    def step(
        self,
        _rng: Array,
        env_state: TextEnvStateWithConstraint[EnvStateT],
        action: int,
        env_params: Any,
    ) -> Tuple[ObsT, TextEnvStateWithConstraint[EnvStateT], Array, Array, Dict[str, Any]]:
        obs, state, reward, done, info = super().step(_rng, env_state, action, env_params)

        game_data_vector = self.StateStructure.from_state(env_state.env_state, state.env_state, action)
        ts = env_state.target_state

        def get_cost_and_ts(cost_type: int, g: Any, ts: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
            
            def fn_budget_hp(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                c = checker_budget_hp_level(g, ts_.hp_level_state.level).astype(float)
                return c, ts_
            def fn_budget_drink(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                c = checker_budget_drink_level(g, ts_.drink_level_state.level).astype(float)
                return c, ts_
            def fn_budget_energy(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                c = checker_budget_energy_level(g, ts_.energy_level_state.level).astype(float)
                return c, ts_
            def fn_budget_hungry(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                c = checker_budget_hungry_level(g, ts_.hungry_level_state.level).astype(float)
                return c, ts_
            def fn_seq_away_monsters(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                c = checker_away_from_monsters_when_hp_low(g, ts_.away_monsters_hp_level_state).astype(float)
                return c, ts_
            def fn_seq_dont_sleep_near_monsters(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                c = checker_dont_sleep_near_monsters(g).astype(float)
                return c, ts_
            def fn_seq_defeat_monster(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                c = checker_monster_is_attacked_without_sword(g).astype(float)
                return c, ts_
            def fn_rel_last_food(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                new_ts, c = checker_last_visible_target(g, ts_.target_of_interest_food)
                return c.astype(float), ts_.replace(target_of_interest_food=new_ts, target_of_interest=new_ts)
            def fn_rel_last_water(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                new_ts, c = checker_last_visible_target(g, ts_.target_of_interest_water)
                return c.astype(float), ts_.replace(target_of_interest_water=new_ts, target_of_interest=new_ts)
            def fn_rel_avoid_enemy(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                c = checker_relactional_avoid_mob_distance(g, ts_.avoid_mob_distance).astype(float)
                return c, ts_
            def fn_math_food_budget(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                new_ts, c = checker_budget_by_action(g, ts_.budget_food_by_action)
                return c.astype(float), ts_.replace(budget_food_by_action=new_ts, budget_by_action=new_ts)
            def fn_math_wood_budget(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                new_ts, c = checker_budget_by_action(g, ts_.budget_wood_by_action)
                return c.astype(float), ts_.replace(budget_wood_by_action=new_ts, budget_by_action=new_ts)
            def fn_step_on_block(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                c = checker_step_on_block(g, ts_.step_on_block).astype(float)
                return c, ts_
            def fn_dont_move_at_night(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                c = checker_moveing_at_night_level(g, ts_.night_constraint_level).astype(float)
                return c, ts_
            def fn_budget_build_collect(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                c = checker_budget_build_collect(g, ts_.build_budget_state).astype(float)
                return c, ts_
            def fn_sleep_at_night(ts_: ConstrainedTargetState) -> Tuple[Array, ConstrainedTargetState]:
                c = checker_sleep_at_night(g, ts_.night_constraint_level, ts_.day_constraint_level).astype(float)
                return c, ts_
            
            functions = [
                fn_budget_hp, fn_budget_drink, fn_budget_energy, fn_budget_hungry,
                fn_seq_away_monsters, fn_seq_dont_sleep_near_monsters, fn_seq_defeat_monster,
                fn_rel_last_food, fn_rel_last_water, fn_rel_avoid_enemy,
                fn_math_food_budget, fn_math_wood_budget,
                fn_step_on_block, fn_dont_move_at_night, fn_budget_build_collect,
                fn_sleep_at_night
            ]
            
            return jax.lax.switch(cost_type, functions, ts)

        cost, updated_ts = get_cost_and_ts(env_state.cost_type, game_data_vector, ts)
        
        state = TextEnvStateWithConstraint(
            env_state=state.env_state,
            timestep=state.timestep,
            cost_type=env_state.cost_type,
            idx=state.idx,
            success_rate=state.success_rate,
            episode_cost=env_state.episode_cost + cost,
            cost=cost,
            total_success_rate=state.total_success_rate,
            rng=state.rng,
            instruction_done=state.instruction_done,
            checker_id=state.checker_id,
            target_state=updated_ts
        )

        return obs, state, reward, done, info
