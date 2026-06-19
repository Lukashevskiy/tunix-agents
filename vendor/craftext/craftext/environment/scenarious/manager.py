"""Scenario handlers and JAX-ready scenario assembly for core CrafText."""

from typing import List, Type, TypeVar, Generic
from tqdm import tqdm
from dataclasses import dataclass
from enum import Enum

from craftext.environment.scenarious.loader import (
    CraftextScenariosConfigLoader as ScenariosConfigLoader,
    load_scenarios,
    ScenariosConfig,
)

from craftext.environment.scenarious.base import AbstractScenarioHandler
from craftext.environment.scenarious.processors import ScenarioProcessor, EncodedProcessor
from craftext.environment.scenarious.scenario_data_pipeline import (
    BaseScenarioData,
    ScenarioDataPayload,
    ScenarioRows,
    ScenarioDataFactory,
    create_raw_scenario_data_factory,
    create_encoded_scenario_data_factory,
)
from craftext.environment.scenarious.scenario_types import ScenarioMap

from craftext.environment.scenarious.instruction_transformers import(
    AbstractInstructionTransformer, 
    DefaultInstructionTransformer, 
    PlansInstructionTransformer
)

from craftext.environment.craftext_constants import plans_path
from craftext.environment.craftext_constants import Scenarios
from craftext.environment.scenarious.checkers.target_state import TargetState

import logging
from jax import numpy as jnp
import jax

from abc import ABC, abstractmethod

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ScenarioFieldType(Enum):
    """Schema behavior for scenario fields during expansion."""

    SINGLE_VALUE = "single_value"  # The base instruction (not copied)
    PARAPHRASE_LIST = "paraphrase_list"  # A list of paraphrases (added to the base instruction)
    REPEAT_WITH_PARAPHRASES = "repeat_with_paraphrases"  # Repeated for each instruction and its paraphrases

SCENARIO_SCHEMA = {
    "instruction": ScenarioFieldType.SINGLE_VALUE,  
    "instruction_paraphrases": ScenarioFieldType.PARAPHRASE_LIST,  
    "scenario_checker": ScenarioFieldType.REPEAT_WITH_PARAPHRASES,  
    "arguments": ScenarioFieldType.REPEAT_WITH_PARAPHRASES,  
}

@dataclass
class BaseScenarioDataJAX:
    """JAX-serializable scenario tensors used at runtime."""

    scenario_checker: jax.Array
    arguments: TargetState


scenario_data_type = TypeVar('scenario_data_type')
scenario_data_jax_type = TypeVar("scenario_data_jax_type")

class JAXRepresentation(ABC, Generic[scenario_data_type, scenario_data_jax_type]):
    """Abstract converter from Python scenario payloads to JAX payloads."""

    @abstractmethod
    def convert(self, scenarios_data: scenario_data_type) -> scenario_data_jax_type:
        """Convert scenario data object into JAX-ready representation.

        Args:
            scenarios_data: Scenario data in Python/native structures.

        Returns:
            BaseScenarioDataJAX: JAX-friendly scenario representation.
        """
        ...


class DefaultJAXRepresentation(JAXRepresentation[BaseScenarioData, BaseScenarioDataJAX]):
    """Default converter for non-encoded scenario data."""

    def convert(self, scenario_data: BaseScenarioData) -> BaseScenarioDataJAX:
        """Convert base scenario data to JAX tensors.

        Args:
            scenario_data: Scenario data without instruction embeddings.

        Returns:
            BaseScenarioDataJAX: JAX-ready checkers and arguments.
        """
        scenario_checker_jax = self._prepare_jax_checkers(scenario_data.scenario_checker)
        
        data = BaseScenarioDataJAX(
            scenario_checker=scenario_checker_jax,
            arguments=TargetState().stack(scenario_data.arguments),
        )
        
        return data

    def _prepare_jax_checkers(self, checkers_list: List[Scenarios]) -> jax.Array:
        """Convert checker enums into integer JAX array.

        Args:
            checkers_list: Sequence of checker enum values.

        Returns:
            jax.Array: Integer checker ids.
        """
        logger.info("Preparing JAX checkers: %s", len(checkers_list))
        return jnp.array(list(map(lambda x: x.value, checkers_list)))

class BaseScenarioDataHandler(AbstractScenarioHandler):
    """Load, expand, and preprocess raw scenario definitions."""

    def __init__(self, scenario_processor: Type[ScenarioProcessor[object]], instruction_transformer: Type[AbstractInstructionTransformer], config_name: str) -> None:
        """Build scenario data from config for one environment family.

        Args:
            scenario_processor: Processor class for instruction preprocessing.
            instruction_transformer: Transformer class for instruction rewriting.
            config_name: Scenario config name to load.
        """
        super().__init__()
        self.scenario_processor: ScenarioProcessor[object] = scenario_processor()
        self.instruction_transformer = instruction_transformer()
        self.config: ScenariosConfig = self._load_config(config_name)
        self.use_paraphrases: bool = self.config.use_parafrases
        self.environment_key: int = self.config.environment_key
        self.n_instructions: int = 0
        self.instruction_to_update_file: str = plans_path  # This remains as a constant path for the plans
        self._post_config_loaded()
        self.all_scenario: ScenarioMap = self._load_scenarios(self.config)
        self.scenario_data_factory: ScenarioDataFactory[ScenarioDataPayload] = self._resolve_scenario_data_factory()
        self.scenario_data: ScenarioDataPayload = self._prepare_scenarios()

    def _load_config(self, config_name: str) -> ScenariosConfig:
        """Load validated scenario config for current handler.

        Args:
            config_name: Scenario config identifier.

        Returns:
            ScenariosConfig: Parsed config object.
        """
        return ScenariosConfigLoader().load_config(config_name)

    def _post_config_loaded(self) -> None:
        """Hook for subclasses to initialize extra config-derived fields."""
        return

    def _resolve_scenario_data_factory(self) -> ScenarioDataFactory[ScenarioDataPayload]:
        """Select payload factory based on processor capabilities."""
        if isinstance(self.scenario_processor, EncodedProcessor):
            from craftext.environment.scenarious.encoded_support import EncodedScenarioData

            return create_encoded_scenario_data_factory(
                processor_provider=lambda: self.scenario_processor,
                encoded_payload_cls=EncodedScenarioData,
            )
        return create_raw_scenario_data_factory(
            processor_provider=lambda: self.scenario_processor,
        )

    @property
    def initial_instruction(self) -> object:
        """Return one default processed instruction sample.

        Returns:
            object: Single-item processed instruction payload.
        """
        processed_items = self.scenario_processor.process(["None"])
        return processed_items[:1]
    

    def castom_initial_instruction(self, instruction: str) -> object:
        """Return one processed sample for a custom instruction.

        Args:
            instruction: Raw instruction string.

        Returns:
            object: Single-item processed instruction payload.
        """
        processed_items = self.scenario_processor.process([instruction])
        return processed_items[:1]

    def _load_scenarios(self, config: ScenariosConfig) -> ScenarioMap:
        """Load raw scenarios dictionary for config.

        Args:
            config: Parsed scenarios config.

        Returns:
            ScenarioMap: Raw scenario mapping.
        """
        return load_scenarios(config)

    def get_scenarios(self) -> ScenarioDataPayload:
        """Return processed scenario data structure.

        Returns:
            ScenarioDataPayload: Materialized scenario data contract.
        """
        
        return self.scenario_data

    
    def _collect_scenario_rows(self) -> ScenarioRows:
        """Expand scenario config entries into flat aligned row lists."""
        instructions_list: List[str] = []
        checker_indecies:  List[Scenarios] = []
        arguments:         List[TargetState] = []
        names:             List[str] = []
        keys = list(self.all_scenario.keys())
        
        for name in tqdm(keys):
            entry = self.all_scenario[name]
            current_instr = entry["instruction"]
            current_checker_index = entry["scenario_checker"]
            current_paraphrases = entry.get("instruction_paraphrases", [])
            current_arguments = entry["arguments"]
            instructions_list.append(current_instr)
            checker_indecies.append(current_checker_index)
            arguments.append(current_arguments)
            names.append(name)
            if self.use_paraphrases:
                for para in current_paraphrases:
                    names.append(f"{name}_PARA")
                    instructions_list.append(para)
                    checker_indecies.append(current_checker_index)
                    arguments.append(current_arguments)

        return ScenarioRows(
            instructions_list=instructions_list,
            checker_indices=checker_indecies,
            arguments=arguments,
            scenario_names=names,
        )

    def _prepare_scenarios(self) -> ScenarioDataPayload:
        """Materialize scenario rows via selected payload factory.

        Returns:
            ScenarioDataPayload: Processed scenario data contract.
        """
        rows = self._collect_scenario_rows()
        scenario_data = self.scenario_data_factory.build(rows)
        is_encoded = isinstance(self.scenario_processor, EncodedProcessor)
        logger.info("Prepared %s instructions (Encoded: %s)", len(rows.instructions_list), is_encoded)
        return scenario_data

class  JaxScenarioDataHandler(BaseScenarioDataHandler, Generic[scenario_data_jax_type]):
    """Scenario handler with additional JAX conversion stage."""

    def __init__(self, scenario_processor: Type[ScenarioProcessor[object]], instruction_transformer: Type[AbstractInstructionTransformer], config_name: str, jax_representation_class: Type[JAXRepresentation[ScenarioDataPayload, scenario_data_jax_type]]) -> None:
        """Create scenario handler with JAX representation stage.

        Args:
            scenario_processor: Processor class for instruction preprocessing.
            instruction_transformer: Transformer class for instruction rewriting.
            config_name: Scenario config name.
            jax_representation_class: Converter class to JAX representation.
        """
        super().__init__(scenario_processor, instruction_transformer, config_name)
        
        self.jax_representation_converter: JAXRepresentation[ScenarioDataPayload, scenario_data_jax_type] = jax_representation_class()
        self.scenario_data_jax: scenario_data_jax_type = self.scenarios_to_jax()

    def scenarios_to_jax(self) -> scenario_data_jax_type:
        """Convert loaded scenario data to JAX-friendly structures.

        Returns:
            scenario_data_jax_type: JAX scenario payload.
        """
        return self.jax_representation_converter.convert(self.scenario_data)

def create_scenarios_with_dataset(use_plans_gpt: bool) -> Type[JaxScenarioDataHandler[BaseScenarioDataJAX]]:
    """Factory producing a scenario handler class with selected transformer.

    Args:
        use_plans_gpt: If ``True``, use plans transformer; otherwise default.

    Returns:
        Type[JaxScenarioDataHandler]: Configured handler class.
    """

    class CustomCrafTextScenariosWithPlans(JaxScenarioDataHandler):
        """Concrete scenario handler class with fixed transformer strategy."""

        def __init__(self, scenario_processor: Type[ScenarioProcessor[object]], config_name: str) -> None:
            """Initialize configured inner scenario handler.

            Args:
                scenario_processor: Processor class for instructions.
                config_name: Scenario config name.
            """
            instruction_transformer: Type[AbstractInstructionTransformer] = PlansInstructionTransformer if use_plans_gpt else DefaultInstructionTransformer
            super().__init__(scenario_processor, instruction_transformer, jax_representation_class=DefaultJAXRepresentation, config_name=config_name)
    return CustomCrafTextScenariosWithPlans
