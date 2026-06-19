"""Encoded-scenario data structures and JAX conversion utilities."""

from dataclasses import dataclass
from typing import List

import jax
from jax import numpy as jnp
import numpy as np

from craftext.environment.craftext_constants import Scenarios
from craftext.environment.scenarious.checkers.target_state import TargetState

import logging

logger = logging.getLogger(__name__)


@dataclass
class EncodedScenarioData:
    instructions_list: List[str]
    scenario_checker: List[Scenarios]
    arguments: List[TargetState]
    scenario_names: List[str]
    embeddings_list: np.ndarray


@dataclass
class EncodedScenarioDataJAX:
    scenario_checker: jax.Array
    arguments: TargetState
    embeddings_list: jax.Array


class EncodedJAXRepresentation:
    def convert(self, scenario_data: EncodedScenarioData) -> EncodedScenarioDataJAX:
        """Converts encoded scenario data to JAX-compatible structures."""
        scenario_checker_jax = jnp.array([checker.value for checker in scenario_data.scenario_checker])
        embeddings_jax = jnp.array(scenario_data.embeddings_list)

        logger.info("Final number of instructions: %s", len(scenario_data.embeddings_list))

        return EncodedScenarioDataJAX(
            scenario_checker=scenario_checker_jax,
            arguments=TargetState().stack(scenario_data.arguments),
            embeddings_list=embeddings_jax,
        )
