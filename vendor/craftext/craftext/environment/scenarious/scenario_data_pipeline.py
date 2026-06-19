"""Pipeline-based scenario data factories and reusable build components."""

from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, List, Optional, Protocol, Type, TypeVar, cast
from abc import ABC, abstractmethod

from craftext.environment.craftext_constants import Scenarios
from craftext.environment.scenarious.checkers.target_state import TargetState
from craftext.environment.scenarious.processors import ScenarioProcessor


@dataclass
class BaseScenarioData:
    """Materialized scenario rows before JAX conversion."""

    instructions_list: List[str]
    scenario_checker: List[Scenarios]
    arguments: List[TargetState]
    scenario_names: List[str]


class ScenarioDataPayload(Protocol):
    """Shared contract for scenario data used by wrappers/JAX converters."""

    instructions_list: List[str]
    scenario_checker: List[Scenarios]
    arguments: List[TargetState]
    scenario_names: List[str]


PayloadT_co = TypeVar("PayloadT_co", bound=ScenarioDataPayload, covariant=True)
PayloadT = TypeVar("PayloadT", bound=ScenarioDataPayload)


@dataclass
class ScenarioRows:
    """Expanded scenario rows before materialization into concrete payloads."""

    instructions_list: List[str]
    checker_indices: List[Scenarios]
    arguments: List[TargetState]
    scenario_names: List[str]


class ScenarioDataFactory(Generic[PayloadT_co], ABC):
    """Factory that materializes collected rows into scenario payload objects."""

    @abstractmethod
    def build(self, rows: ScenarioRows) -> PayloadT_co:
        ...


@dataclass
class ScenarioBuildContext(Generic[PayloadT]):
    """Mutable context passed through pipeline factory components."""

    rows: ScenarioRows
    processor: ScenarioProcessor
    base_payload: Optional[BaseScenarioData] = None
    embeddings: Optional[object] = None
    payload: Optional[PayloadT] = None


class ScenarioBuildComponent(Generic[PayloadT], ABC):
    """Single processing step used inside scenario-data factory pipeline."""

    @abstractmethod
    def run(self, context: ScenarioBuildContext[PayloadT]) -> None:
        ...


class BuildBasePayloadComponent(ScenarioBuildComponent[PayloadT]):
    """Create base payload from flattened rows."""

    def run(self, context: ScenarioBuildContext[PayloadT]) -> None:
        context.base_payload = BaseScenarioData(
            instructions_list=context.rows.instructions_list,
            scenario_checker=context.rows.checker_indices,
            arguments=context.rows.arguments,
            scenario_names=context.rows.scenario_names,
        )


class ComputeEmbeddingsComponent(ScenarioBuildComponent[PayloadT]):
    """Compute embeddings using current processor."""

    def run(self, context: ScenarioBuildContext[PayloadT]) -> None:
        context.embeddings = context.processor.process(context.rows.instructions_list)


class FinalizeRawPayloadComponent(ScenarioBuildComponent[BaseScenarioData]):
    """Select base payload as final payload."""

    def run(self, context: ScenarioBuildContext[BaseScenarioData]) -> None:
        if context.base_payload is None:
            raise ValueError("BuildBasePayloadComponent must run before FinalizeRawPayloadComponent.")
        context.payload = context.base_payload


EncodedPayloadT = TypeVar("EncodedPayloadT", bound=ScenarioDataPayload)


class EncodedPayloadClass(Protocol[EncodedPayloadT]):
    def __call__(
        self,
        *,
        instructions_list: List[str],
        scenario_checker: List[Scenarios],
        arguments: List[TargetState],
        scenario_names: List[str],
        embeddings_list: object,
    ) -> EncodedPayloadT:
        ...


class FinalizeEncodedPayloadComponent(ScenarioBuildComponent[EncodedPayloadT]):
    """Compose final encoded payload from base fields and computed embeddings."""

    def __init__(self, encoded_payload_cls: EncodedPayloadClass[EncodedPayloadT]) -> None:
        self.encoded_payload_cls = encoded_payload_cls

    def run(self, context: ScenarioBuildContext[EncodedPayloadT]) -> None:
        if context.base_payload is None:
            raise ValueError("BuildBasePayloadComponent must run before FinalizeEncodedPayloadComponent.")
        if context.embeddings is None:
            raise ValueError("ComputeEmbeddingsComponent must run before FinalizeEncodedPayloadComponent.")

        context.payload = self.encoded_payload_cls(
            instructions_list=context.base_payload.instructions_list,
            scenario_checker=context.base_payload.scenario_checker,
            arguments=context.base_payload.arguments,
            scenario_names=context.base_payload.scenario_names,
            embeddings_list=context.embeddings,
        )


@dataclass
class ComponentSpec:
    """Component declaration used by pipeline factory."""

    name: str
    kwargs: Dict[str, object] = field(default_factory=dict)


class PipelineScenarioDataFactory(ScenarioDataFactory[PayloadT]):
    """Factory that executes a registered sequence of components."""

    _component_registry: Dict[str, Type[ScenarioBuildComponent[ScenarioDataPayload]]] = {}

    @classmethod
    def register_component(cls, name: str, component_cls: Type[ScenarioBuildComponent[ScenarioDataPayload]]) -> None:
        cls._component_registry[name] = component_cls

    def __init__(
        self,
        component_sequence: List[ComponentSpec],
        processor_provider: Callable[[], ScenarioProcessor],
    ) -> None:
        if len(component_sequence) == 0:
            raise ValueError("Scenario build pipeline must contain at least one component.")
        self.components = self._build_components(component_sequence)
        self.processor_provider = processor_provider

    def _build_components(self, sequence: List[ComponentSpec]) -> List[ScenarioBuildComponent[ScenarioDataPayload]]:
        components: List[ScenarioBuildComponent[ScenarioDataPayload]] = []
        for spec in sequence:
            component_cls = self._component_registry.get(spec.name)
            if component_cls is None:
                raise ValueError(f"Unknown scenario build component: {spec.name}")
            components.append(component_cls(**spec.kwargs))
        return components

    def build(self, rows: ScenarioRows) -> PayloadT:
        context = ScenarioBuildContext[PayloadT](rows=rows, processor=self.processor_provider())
        for component in self.components:
            cast(ScenarioBuildComponent[PayloadT], component).run(context)

        if context.payload is None:
            raise ValueError("Scenario build pipeline finished without a final payload.")
        return context.payload


PipelineScenarioDataFactory.register_component("build_base_payload", BuildBasePayloadComponent)
PipelineScenarioDataFactory.register_component("compute_embeddings", ComputeEmbeddingsComponent)
PipelineScenarioDataFactory.register_component("finalize_raw_payload", FinalizeRawPayloadComponent)
PipelineScenarioDataFactory.register_component("finalize_encoded_payload", FinalizeEncodedPayloadComponent)


def create_raw_scenario_data_factory(
    processor_provider: Callable[[], ScenarioProcessor],
) -> ScenarioDataFactory[BaseScenarioData]:
    return PipelineScenarioDataFactory[BaseScenarioData](
        component_sequence=[
            ComponentSpec(name="build_base_payload"),
            ComponentSpec(name="finalize_raw_payload"),
        ],
        processor_provider=processor_provider,
    )


def create_encoded_scenario_data_factory(
    processor_provider: Callable[[], ScenarioProcessor],
    encoded_payload_cls: EncodedPayloadClass[EncodedPayloadT],
) -> ScenarioDataFactory[EncodedPayloadT]:
    return PipelineScenarioDataFactory[EncodedPayloadT](
        component_sequence=[
            ComponentSpec(name="build_base_payload"),
            ComponentSpec(name="compute_embeddings"),
            ComponentSpec(
                name="finalize_encoded_payload",
                kwargs={"encoded_payload_cls": encoded_payload_cls},
            ),
        ],
        processor_provider=processor_provider,
    )
