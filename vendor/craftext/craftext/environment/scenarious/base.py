"""Abstract interfaces for scenario handlers used by CrafText environments."""

from abc import ABC, abstractmethod
from craftext.environment.scenarious.loader import ScenariosConfig
from craftext.environment.scenarious.scenario_data_pipeline import ScenarioDataPayload
from craftext.environment.scenarious.scenario_types import ScenarioMap


class AbstractScenarioHandler(ABC):
    
    @property
    @abstractmethod
    def initial_instruction(self) -> object:
        ...

    @abstractmethod
    def castom_initial_instruction(self, instruction: str) -> object:
        ...

    @abstractmethod
    def _load_scenarios(self, config: ScenariosConfig) -> ScenarioMap:
        ...

    @abstractmethod
    def get_scenarios(self) -> ScenarioDataPayload:
        ...

    @abstractmethod
    def _prepare_scenarios(self) -> ScenarioDataPayload:
        ...
        
