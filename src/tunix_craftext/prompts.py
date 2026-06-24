"""Typed prompt assembly at the CrafText environment-to-model boundary.

This module defines prompt rendering contracts, stable action catalogues, and
wrapper logic for vendored MegaPrompts templates. It keeps prompt generation
explicit and verifiable for downstream action decoding.

Example:
    >>> renderer = MegaPromptRenderer('my_template')
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

ObservationT = TypeVar("ObservationT")


class PromptContractError(ValueError):
    """Raised when a prompt/action contract is incomplete or ambiguous.

    Example:
        >>> raise PromptContractError("message")
    """


@dataclass(frozen=True)
class ActionCatalog:
    """Stable mapping between model-visible action labels and environment action ids.

    :ivar labels: Ordered, unique action labels; index is the discrete environment action id.
    """

    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate label non-emptiness and uniqueness on construction.

        :raises PromptContractError: If labels are empty or duplicated.
        """
        if not self.labels or any(not label.strip() for label in self.labels):
            raise PromptContractError("action catalog must contain non-empty labels")
        if len(self.labels) != len(set(self.labels)):
            raise PromptContractError("action catalog labels must be unique")

    def index_of(self, label: str) -> int:
        """Return a controlled action id for one model-produced label.

        :param label: str input value
        :returns: int
        :raises PromptContractError: If the label is outside the declared action space.
        """
        try:
            return self.labels.index(label)
        except ValueError as error:
            raise PromptContractError(f"unknown action label: {label!r}") from error


@dataclass(frozen=True)
class PromptContext(Generic[ObservationT]):
    """All explicit data required to render one model decision prompt.

    :ivar goal: Episode instruction supplied by the environment/scenario.
    :ivar observation: Vendor observation or state payload consumed by the renderer.
    :ivar actions: Stable action catalogue for this environment configuration.
    :ivar dialog: Immutable previous dialogue turns, if the prompt template uses them.
    :ivar safety: Optional safety constraint text for the current episode.
    :ivar world_preset: Optional CrafText world preset; templates may use it to explain map rules.
    """

    goal: str
    observation: ObservationT
    actions: ActionCatalog
    dialog: tuple[str, ...] = ()
    safety: str = ""
    world_preset: str = ""


@dataclass(frozen=True)
class RenderedPrompt:
    """Prompt text paired with the exact action space it was rendered against.

    :ivar text: str
    :ivar actions: ActionCatalog
    :ivar template_name: str

    Example:
        >>> obj = RenderedPrompt(text=..., actions=..., template_name=...)"""

    text: str
    actions: ActionCatalog
    template_name: str


class PromptRenderer(Protocol[ObservationT]):
    """Framework-neutral renderer signature used by model/policy adapters."""

    def render(self, context: PromptContext[ObservationT]) -> RenderedPrompt:
        """Render a `PromptContext` into a `RenderedPrompt`.

        :param context: PromptContext[ObservationT] with goal, observation and actions.
        :returns: RenderedPrompt containing text and action catalogue.
        """
        ...


class MegaPromptBackend(Protocol):
    """Narrow typed view of the vendored MegaPrompts renderer."""
    def render(self, meta_info: Mapping[str, object]) -> str:
        """Render a MegaPrompts template given meta information.

        :param meta_info: Mapping of template variables required by the template.
        :returns: Rendered template text.
        """
        ...


class MegaPromptRenderer(Generic[ObservationT]):
    """Adapt a vendored MegaPrompts config without leaking its dynamic API into core.

    :param config_name: Template directory name under MegaPrompts' ``templates`` root.
    :param backend: Optional renderer injection for tests; otherwise imports ``megaprompt``.
    :raises PromptContractError: If the optional ``prompts`` extra is absent or output is blank.
    """

    def __init__(self, config_name: str, backend: MegaPromptBackend | None = None) -> None:
        """Create a `MegaPromptRenderer` for the given template name.

        :param config_name: Template directory name under MegaPrompts' templates root.
        :param backend: Optional test-injected backend; otherwise the vendored renderer is used.
        :raises PromptContractError: If config_name is empty or backend cannot be loaded.
        """
        if not config_name.strip():
            raise PromptContractError("prompt config name must be non-empty")
        self._config_name = config_name
        self._backend = backend or self._load_backend(config_name)

    @staticmethod
    def _load_backend(config_name: str) -> MegaPromptBackend:
        """Import and return the vendored MegaPrompts `Renderer`.

        :param config_name: Template directory name used to instantiate the backend.
        :returns: An object implementing `MegaPromptBackend`.
        :raises PromptContractError: If the optional `megaprompt` extra is not installed.
        """
        try:
            from megaprompt.renders import Renderer  # type: ignore[import-not-found]
        except ImportError as error:
            raise PromptContractError(
                "install `tunix-craftext[prompts]` to render vendored MegaPrompts templates"
            ) from error
        return Renderer(config_name=config_name)

    def render(self, context: PromptContext[ObservationT]) -> RenderedPrompt:
        """Render one environment-derived prompt and preserve its action catalogue.

        :param context: PromptContext[ObservationT] input value
        :returns: RenderedPrompt

        Example:
            >>> rendered = renderer.render(context)
        """
        text = self._backend.render(
            {
                "goal": context.goal,
                "obs": context.observation,
                "act": list(context.actions.labels),
                "dialog": list(context.dialog),
                "safety": context.safety,
                "world_preset": context.world_preset,
            }
        )
        if not text.strip():
            raise PromptContractError("prompt renderer returned blank text")
        return RenderedPrompt(text=text, actions=context.actions, template_name=self._config_name)
