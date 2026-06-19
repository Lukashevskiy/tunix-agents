"""CMDP scenario handlers and JAX conversion for Caged CrafText."""

import numpy as np
import jax
import jax.numpy as jnp
import logging
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Union, Type

from craftext.environment.scenarious.loader import ScenariosConfig
from caged_craftext.environment.scenarious.loader import CagedCraftextScenariosConfigLoader, load_caged_scenarios
from craftext.environment.scenarious.instruction_transformers import AbstractInstructionTransformer, DefaultInstructionTransformer, PlansInstructionTransformer
from craftext.environment.scenarious.processors import ScenarioProcessor, EncodedProcessor

# Import from original craftext manager
from craftext.environment.scenarious.manager import (
    BaseScenarioData,
    BaseScenarioDataJAX,
    JAXRepresentation,
    DefaultJAXRepresentation,
    JaxScenarioDataHandler,
    ScenarioFieldType,
    SCENARIO_SCHEMA as BASE_SCENARIO_SCHEMA,
    BaseScenarioDataHandler
)
from craftext.environment.scenarious.encoded_support import (
    EncodedScenarioData,
    EncodedScenarioDataJAX,
    EncodedJAXRepresentation,
)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COST_TYPE_MAP = {
    "budget_hp": 0,
    "budget_drink": 1,
    "budget_energy": 2,
    "budget_hungry": 3,
    "sequential_away_monsters_when_hp": 4,
    "sequential_dont_sleep_near_monsters": 5,
    "sequential_defeat_monster": 6,
    "relational_last_food_location": 7,
    "relational_last_water_location": 8,
    "relational_avoid_enemy_by_radius": 9,
    "math_food_budget": 10,
    "math_wood_budget": 11,
    "step_on_block": 12,
    "dont_move_at_night": 13,
    "budget_build_collect": 14,
    "sleep_at_night": 15,
}

# Extend SCENARIO_SCHEMA for CMDP
CMDP_SCENARIO_SCHEMA = {
    **BASE_SCENARIO_SCHEMA,
    "str_check_lambda": ScenarioFieldType.REPEAT_WITH_PARAPHRASES
}

# Dataclasses for CMDP
@dataclass
class CMDPBaseScenarioData(BaseScenarioData):
    """Scenario rows for CMDP tasks before JAX conversion.

    Attributes:
        texutal_constraints_list: Text constraints aligned with instructions.
        cost_types_list: Numeric cost type ids for each row.
        str_check_lambda_list: Optional checker lambdas serialized as strings.
    """

    texutal_constraints_list: List[str]
    cost_types_list: List[str]
    str_check_lambda_list: List[str]

@dataclass
class CMDPEncodedScenarioData(EncodedScenarioData):
    """Encoded CMDP scenario rows including instruction embeddings.

    Attributes:
        texutal_constraints_list: Text constraints aligned with instructions.
        cost_types_list: Numeric cost type ids for each row.
        str_check_lambda_list: Optional checker lambdas serialized as strings.
        constraints_embeddings_list: Embeddings for textual constraints.
    """

    texutal_constraints_list: List[str]
    cost_types_list: List[str]
    str_check_lambda_list: List[str]
    constraints_embeddings_list: np.ndarray

@dataclass
class CMDPBaseScenarioDataJAX(BaseScenarioDataJAX):
    """JAX-ready CMDP scenario tensors without text embeddings.

    Attributes:
        cost_types: Per-row numeric cost type ids as JAX array.
    """

    cost_types: jax.Array

@dataclass
class CMDPEncodedScenarioDataJAX(EncodedScenarioDataJAX):
    """JAX-ready CMDP scenario tensors with text embeddings.

    Attributes:
        constraints_embeddings_list: Constraint embeddings as JAX array.
        cost_types: Per-row numeric cost type ids as JAX array.
    """

    constraints_embeddings_list: jax.Array
    cost_types: jax.Array

# JAX Representation for CMDP
class CMDPDefaultJAXRepresentation(DefaultJAXRepresentation):
    """Convert non-encoded CMDP scenarios into JAX representation."""

    def convert(self, scenario_data: CMDPBaseScenarioData) -> CMDPBaseScenarioDataJAX:
        """Convert CMDP base scenario payload to JAX arrays.

        Args:
            scenario_data: CMDP scenario data without embeddings.

        Returns:
            CMDPBaseScenarioDataJAX: Converted scenario payload.
        """
        base_jax = super().convert(scenario_data)
        cost_types_jax = jnp.array(scenario_data.cost_types_list)
        return CMDPBaseScenarioDataJAX(
            scenario_checker=base_jax.scenario_checker,
            arguments=base_jax.arguments,
            cost_types=cost_types_jax
        )

class CMDPEncodedJAXRepresentation(EncodedJAXRepresentation):
    """Convert encoded CMDP scenarios into JAX representation."""

    def convert(self, scenario_data: CMDPEncodedScenarioData) -> CMDPEncodedScenarioDataJAX:
        """Convert CMDP encoded scenario payload to JAX arrays.

        Args:
            scenario_data: CMDP scenario data with embeddings.

        Returns:
            CMDPEncodedScenarioDataJAX: Converted scenario payload.
        """
        base_jax = super().convert(scenario_data)
        constraints_embeddings_jax = jnp.array(scenario_data.constraints_embeddings_list)
        cost_types_jax = jnp.array(scenario_data.cost_types_list)
        
        logger.info(f"Final number of instructions: {len(scenario_data.embeddings_list)}")
        
        return CMDPEncodedScenarioDataJAX(
            scenario_checker=base_jax.scenario_checker,
            arguments=base_jax.arguments,
            embeddings_list=base_jax.embeddings_list,
            constraints_embeddings_list=constraints_embeddings_jax,
            cost_types=cost_types_jax
        )

# Main CMDP Scenario Handler
class CMDPJaxScenarioDataHandler(JaxScenarioDataHandler):
    """Scenario handler that loads, expands, and converts CMDP scenarios."""

    def __init__(self, scenario_processor: Type[ScenarioProcessor], instruction_transformer: Type[AbstractInstructionTransformer], config_name: str, jax_representation_class: Type[JAXRepresentation]) -> None:
        """Initialize CMDP scenario handler.

        Args:
            scenario_processor: Instruction processor class.
            instruction_transformer: Instruction transformer class.
            config_name: Scenario config identifier.
            jax_representation_class: JAX converter class.
        """
        super().__init__(
            scenario_processor=scenario_processor,
            instruction_transformer=instruction_transformer,
            config_name=config_name,
            jax_representation_class=jax_representation_class,
        )

    def _load_config(self, config_name: str) -> ScenariosConfig:
        """Load validated caged scenario config.

        Args:
            config_name: Scenario config identifier.

        Returns:
            ScenariosConfig: Parsed config object.
        """
        return CagedCraftextScenariosConfigLoader().load_config(config_name)

    def _post_config_loaded(self) -> None:
        """Initialize flags derived from loaded config."""
        self.use_constraints_parafrases: bool = self.config.use_constraints_parafrases

    def _load_scenarios(self, config: ScenariosConfig) -> Dict[str, Any]:
        """Load raw caged scenario mapping.

        Args:
            config: Parsed scenario config.

        Returns:
            Dict[str, Any]: Scenario dictionary.
        """
        return load_caged_scenarios(config)

    def _prepare_scenarios(self) -> Union[CMDPBaseScenarioData, CMDPEncodedScenarioData]:
        """Expand scenarios to row-wise structure and optionally encode text.

        Returns:
            Union[CMDPBaseScenarioData, CMDPEncodedScenarioData]: Prepared scenario payload.
        """
        instructions_list, textual_constraints_list, indices_list, checkers_data_dict, cost_types_list = self._pairwise_instructions_and_checkers()

        logger.info(f"Initial number of instructions: {len(instructions_list)}")

        is_encoded = isinstance(self.scenario_processor, EncodedProcessor)
        
        if is_encoded:
            embeddings_list = self.scenario_processor.process(instructions_list)
            constraints_embeddings_list = self.scenario_processor.process(textual_constraints_list)
            
            return CMDPEncodedScenarioData(
                instructions_list=instructions_list,
                scenario_checker=checkers_data_dict["scenario_checker"],
                arguments=checkers_data_dict["arguments"],
                scenario_names=[str(i) for i in indices_list],
                embeddings_list=np.array(embeddings_list),
                texutal_constraints_list=textual_constraints_list,
                cost_types_list=cost_types_list,
                str_check_lambda_list=checkers_data_dict["str_check_lambda"],
                constraints_embeddings_list=np.array(constraints_embeddings_list)
            )
        else:
            return CMDPBaseScenarioData(
                instructions_list=instructions_list,
                scenario_checker=checkers_data_dict["scenario_checker"],
                arguments=checkers_data_dict["arguments"],
                scenario_names=[str(i) for i in indices_list],
                texutal_constraints_list=textual_constraints_list,
                cost_types_list=cost_types_list,
                str_check_lambda_list=checkers_data_dict["str_check_lambda"],
            )
    
    def _pairwise_instructions_and_checkers(self) -> Tuple[List[str], List[str], List[int], Dict[str, List[Any]], List[str]]:
        """Flatten all scenarios into instruction/constraint/checker rows.

        Returns:
            Tuple[List[str], List[str], List[int], Dict[str, List[Any]], List[str]]:
                Flattened instructions, constraints, scenario ids, checker fields, and cost ids.
        """
        instructions_list, textual_constraints_list, indices_list, cost_types_list = [], [], [], []
        checkers_data_dict = {key: [] for key in CMDP_SCENARIO_SCHEMA.keys() if key != "instruction_paraphrases" and key != "instruction"}
        
        for idx, (key, scenario) in tqdm(enumerate(self.all_scenario.items())):
            instructions, textual_constraints, indices, checkers_data, cost_types = self._pairwise_goal_parafrases_and_checkers(scenario, idx)
            instructions_list.extend(instructions)
            textual_constraints_list.extend(textual_constraints)
            indices_list.extend(indices)
            cost_types_list.extend(cost_types)
            for field in checkers_data_dict.keys():
                checkers_data_dict[field].extend(checkers_data[field])
        
        if self.instruction_transformer:
             instructions_list = self.instruction_transformer.transform(instructions_list)
        return instructions_list, textual_constraints_list, indices_list, checkers_data_dict, cost_types_list

    def _pairwise_goal_parafrases_and_checkers(self, scenario: Dict[str, Any], scenario_id: int) -> Tuple[List[str], List[str], List[int], Dict[str, List[Any]], List[str]]:
        """Expand one scenario into paired instruction-constraint rows.

        Args:
            scenario: One raw scenario mapping.
            scenario_id: Stable numeric id of scenario in dataset order.

        Returns:
            Tuple[List[str], List[str], List[int], Dict[str, List[Any]], List[str]]:
                Paired instructions, paired constraints, repeated ids, checker fields, and cost ids.
        """
        instructions = [scenario.get("instruction", "Unknown instruction")]
        
        if self.use_paraphrases:
            instructions.extend(scenario.get("instruction_paraphrases", []))

        if self.use_constraints_parafrases:
            textual_constraints = []
            cost_types = []
            for cost_type_str, pair in zip(scenario.get("cost_types", []), scenario.get("textual_constraints_perephrases", [])):
                cost_type = COST_TYPE_MAP[cost_type_str]
                textual_constraint, paraphrases = pair[0], pair[1]
                textual_constraints.append(textual_constraint)
                cost_types.append(cost_type)
                for paraphrase in paraphrases:
                    textual_constraints.append(paraphrase)
                    cost_types.append(cost_type)
            if not cost_types:
                cost_types = [0]
                textual_constraints = [""]
        else:
            cost_types_str = scenario.get("cost_types", [None])
            cost_types = [COST_TYPE_MAP[cts] for cts in cost_types_str if cts is not None]
            if not cost_types:
                cost_types = [0]
            textual_constraints = scenario.get("textual_constraints", [""])
            if "textual_constraint" in scenario:
                 textual_constraints = [scenario.get("textual_constraint", "")]

        paired_instructions = []
        paired_textual_constraints = []
        paired_cost_types = []
        for instruction in instructions:
            for cost_type, constraint in zip(cost_types, textual_constraints):
                paired_instructions.append(instruction)
                paired_textual_constraints.append(constraint)
                paired_cost_types.append(cost_type)

        num_pairs = len(paired_instructions)
        indices = [scenario_id] * num_pairs
        checkers_data = {key: [] for key in CMDP_SCENARIO_SCHEMA.keys() if key not in ["instruction_paraphrases", "instruction"]}
        for key, field_type in CMDP_SCENARIO_SCHEMA.items():
            if field_type == ScenarioFieldType.REPEAT_WITH_PARAPHRASES:
                checkers_data[key] = [scenario.get(key, None)] * num_pairs
                
        return paired_instructions, paired_textual_constraints, indices, checkers_data, paired_cost_types

def create_scenarios_with_dataset(use_plans_gpt: bool) -> Type[CMDPJaxScenarioDataHandler]:
    """Create configured CMDP scenario handler class factory.

    Args:
        use_plans_gpt: Whether to use plans-based instruction transformer.

    Returns:
        Type[CMDPJaxScenarioDataHandler]: Handler class preconfigured by transformer strategy.
    """

    class CustomCrafTextScenariosWithPlans(CMDPJaxScenarioDataHandler):
        """Concrete CMDP handler class with fixed transformer strategy."""

        def __init__(self, scenario_processor: Type[ScenarioProcessor], config_name: str) -> None:
            """Initialize inner configured CMDP handler.

            Args:
                scenario_processor: Instruction processor class.
                config_name: Scenario config identifier.
            """
            instruction_transformer: Type[AbstractInstructionTransformer] = PlansInstructionTransformer if use_plans_gpt else DefaultInstructionTransformer
            jax_representation_class = CMDPEncodedJAXRepresentation if isinstance(scenario_processor(), EncodedProcessor) else CMDPDefaultJAXRepresentation
            super().__init__(scenario_processor, instruction_transformer, config_name=config_name, jax_representation_class=jax_representation_class)
    return CustomCrafTextScenariosWithPlans
