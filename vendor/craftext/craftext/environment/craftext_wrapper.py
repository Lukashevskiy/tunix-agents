"""Instruction-aware wrappers and typed state containers for CrafText env steps."""

from abc import ABC, abstractmethod
import logging
from typing import Callable, Dict, Generic, MutableMapping, Optional, Protocol, Sequence, Tuple, TypeVar, Union, Type, cast
import jax
import jax.numpy as jnp
from jax import Array
from jax import lax

from flax import struct


from craftext.environment.scenarious.manager import  JaxScenarioDataHandler

from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic
from craftext.environment.craftext_constants import Scenarios 
from craftext.environment.scenarious.checkers.target_state  import TargetState
from craftext.environment.scenarious.checkers.registry import CHECKER_FUNCTIONS

logger = logging.getLogger(__name__)

ObsT = TypeVar("ObsT", covariant=True)
EnvStateT = TypeVar("EnvStateT")
InfoT = TypeVar("InfoT", bound=MutableMapping[str, object], covariant=True)

CheckerFn = Callable[[Union[GameData, GameDataClassic], TargetState], Array]


class StateStructureProtocol(Protocol):
    """Protocol for state adapters used by checker functions."""

    @classmethod
    def from_state(cls, previos_state: object, current_state: object, action: int) -> Union[GameData, GameDataClassic]:
        """Build checker-friendly game data from two env states.

        Args:
            previos_state: Previous environment state.
            current_state: Current environment state after step.
            action: Action applied between the two states.

        Returns:
            Union[GameData, GameDataClassic]: State payload used by checkers.
        """
        ...


class JaxEnvProtocol(Protocol[ObsT, EnvStateT, InfoT]):
    """Minimal environment protocol required by instruction wrappers."""

    def reset(self, key: Array, params: object) -> Tuple[ObsT, EnvStateT]:
        """Reset environment state.

        Args:
            key: JAX RNG key.
            params: Environment parameters object.

        Returns:
            Tuple[ObsT, EnvStateT]: Initial observation and state.
        """
        ...

    def step(
        self,
        key: Array,
        state: EnvStateT,
        action: int,
        params: object,
    ) -> Tuple[ObsT, EnvStateT, Array, Array, InfoT]:
        """Apply one environment step.

        Args:
            key: JAX RNG key.
            state: Current environment state.
            action: Discrete action id.
            params: Environment parameters object.

        Returns:
            Tuple[ObsT, EnvStateT, Array, Array, InfoT]:
                Observation, next state, reward, done flag, and info mapping.
        """
        ...

@struct.dataclass
class TextEnvState(Generic[EnvStateT]):
    """Wrapper state that augments env state with instruction metadata."""

    env_state: EnvStateT
    timestep: Array
    idx: Array
    success_rate: Array
    total_success_rate: Array
    rng: Array
    instruction_done: Array
    checker_id: Array
    target_state: TargetState   


def generic_check(
    game_data: Union[GameData, GameDataClassic],
    target_state: TargetState,
    idx: Array,
    fns: Sequence[CheckerFn],
) -> jnp.ndarray:
    """Dispatch checker function by index.

    Args:
        game_data: Extracted state payload for checker evaluation.
        target_state: Target state aligned with current instruction.
        idx: Checker index from `Scenarios` enum values.
        fns: Tuple/list of checker callables.

    Returns:
        jnp.ndarray: Boolean-like JAX scalar indicating task completion.
    """
    return lax.switch(idx, fns, game_data, target_state)


class BaseInstructionWrapper(Generic[ObsT, EnvStateT, InfoT], ABC):
    """Base wrapper that injects instruction-check logic into env transitions."""

    def __init__(self, env: JaxEnvProtocol[ObsT, EnvStateT, InfoT], scenario_handler: JaxScenarioDataHandler) -> None:
        """Initialize instruction-aware wrapper.

        Args:
            env: Environment implementing `JaxEnvProtocol`.
            scenario_handler: Prepared scenario handler with JAX payloads.
        """
        self.scenario_handler = scenario_handler

        self.env = env
        self.steps = 0

        # Determine the environment key and state structure
        self.environment_key = self.scenario_handler.environment_key
        self.StateStructure: Type[StateStructureProtocol] = cast(
            Type[StateStructureProtocol],
            GameData if self.environment_key == 1 else GameDataClassic,
        )

        logger.info("Initialized Instruction Wrapper with environment key: %s", self.environment_key)
        # print(self.StateStructure)
        self.n_instructions = len(self.scenario_handler.scenario_data.instructions_list)

    @abstractmethod
    def _get_instruction(self, idx: int) -> Tuple[Optional[Array], str]:
        """Fetch instruction representation and text for an index.

        Args:
            idx: Instruction index.

        Returns:
            Tuple[Optional[Array], str]:
                Optional instruction embedding/token payload and raw text.
        """
        pass

    def reset(self, _rng: Array, env_params: object, instruction_idx: int = -1) -> Tuple[ObsT, TextEnvState[EnvStateT]]:
        """Reset env and bind one instruction to the new episode.

        Args:
            _rng: JAX RNG key.
            env_params: Environment parameter object.
            instruction_idx: Fixed instruction index. If `-1`, index is sampled.

        Returns:
            Tuple[ObsT, TextEnvState[EnvStateT]]:
                Initial observation and augmented wrapper state.
        """

        obs, base_state = self.env.reset(_rng, env_params)
        
        idx = cast(
            Array,
            jax.lax.cond(
                instruction_idx == -1,
                lambda: jax.random.randint(_rng, shape=(), minval=0, maxval=self.n_instructions),
                lambda: jnp.asarray(instruction_idx, dtype=jnp.int32),
            ),
        )
        
        # Initialize the state with the selected instruction embedding/token and set success rates to zero
        text_state = TextEnvState(
            env_state=base_state,
            timestep=jnp.asarray(base_state.timestep),
            idx=idx,
            success_rate=jnp.asarray(0.0, dtype=jnp.float32),
            total_success_rate=jnp.asarray(0.0, dtype=jnp.float32),
            rng=_rng,
            instruction_done=jnp.asarray(False),
            checker_id=self.scenario_handler.scenario_data_jax.scenario_checker[idx],
            target_state=self.scenario_handler.scenario_data_jax.arguments.select(idx)
        )
        return obs, text_state

    def step(
        self,
        _rng: Array,
        env_state: TextEnvState[EnvStateT],
        action: int,
        env_params: object,
    ) -> Tuple[ObsT, TextEnvState[EnvStateT], Array, Array, InfoT]:
        """Run one environment transition with instruction completion check.

        Args:
            _rng: JAX RNG key.
            env_state: Current wrapper state.
            action: Discrete action id.
            env_params: Environment parameter object.

        Returns:
            Tuple[ObsT, TextEnvState[EnvStateT], Array, Array, InfoT]:
                Observation, next wrapper state, reward, done flag, and info.
        """
        obs, next_env_state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        
        # Obtain the game data vector for the current state and check instruction completion
        game_data_vector = self.StateStructure.from_state(env_state.env_state, next_env_state, action)
                    
        ts = self.scenario_handler.scenario_data_jax.arguments.select(env_state.idx)

        instruction_done = generic_check(game_data_vector, ts, env_state.checker_id, CHECKER_FUNCTIONS)
        
        # If EXPLORE mode - give craftAx reward
        reward = lax.cond(
                    env_state.checker_id != Scenarios.EXPLORE,
                    lambda: reward / 50,
                    lambda: reward
                )
        reward = jax.lax.cond(instruction_done, lambda _: reward + 1, lambda _: reward, operand=None)
        done = instruction_done | done
   
        new_episode_sr = env_state.success_rate + jnp.float32(instruction_done)

        # Update state with the new success rates
        state = TextEnvState(
            env_state=next_env_state,
            timestep=jnp.asarray(next_env_state.timestep),
            idx=env_state.idx,
            success_rate=new_episode_sr * (1 - done),
            total_success_rate=env_state.total_success_rate * (1 - done) + new_episode_sr * done,
            rng=env_state.rng,
            instruction_done=instruction_done,
            checker_id=env_state.checker_id,
            target_state=env_state.target_state
        )
        
        # Update step information in info dictionary
        info.update({"SR": state.total_success_rate, "steps": self.steps})
        info.update({"Cheker_id": env_state.checker_id})
        self.steps += 1
        return obs, state, reward, done, info

class EncodedInstructionWrapper(BaseInstructionWrapper[ObsT, EnvStateT, Dict[str, object]]):
    """Instruction wrapper variant that exposes encoded instruction vectors."""

    def __init__(self, env: JaxEnvProtocol[ObsT, EnvStateT, Dict[str, object]], scenario_handler: JaxScenarioDataHandler) -> None:
        """Initialize encoded-instruction wrapper.

        Args:
            env: Environment implementing `JaxEnvProtocol`.
            scenario_handler: Scenario handler configured for encoded payload.

        Raises:
            ValueError: If handler JAX payload does not contain embeddings.
        """
        super().__init__(env, scenario_handler)
        if not hasattr(self.scenario_handler.scenario_data_jax, 'embeddings_list'):
            raise ValueError("EncodedInstructionWrapper requires a scenario handler that produces embeddings.")

    def _get_instruction(self, idx: int) -> Tuple[Optional[Array], str]:
        """Return encoded instruction vector and source text.

        Args:
            idx: Instruction index.

        Returns:
            Tuple[Optional[Array], str]: Embedding tensor and instruction text.
        """
        instructions_emb = self.scenario_handler.scenario_data_jax.embeddings_list[idx]
        instruction_text = self.scenario_handler.scenario_data.instructions_list[idx]
        return instructions_emb, instruction_text

class RawInstructionWrapper(BaseInstructionWrapper[ObsT, EnvStateT, Dict[str, object]]):
    """Instruction wrapper variant that exposes only raw text instructions."""

    def _get_instruction(self, idx: int) -> Tuple[Optional[Array], str]:
        """Return raw instruction text without embedding payload.

        Args:
            idx: Instruction index.

        Returns:
            Tuple[Optional[Array], str]: `None` embedding and instruction text.
        """
        instruction_text = self.scenario_handler.scenario_data.instructions_list[idx]
        return None, instruction_text

 
 
