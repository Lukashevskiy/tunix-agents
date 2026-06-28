"""Typed text-policy boundary from rendered prompts to discrete environment actions.

This module converts rendered prompt text into model completions and decodes
action tags into discrete environment actions in a stable, inspectable way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .prompts import PromptContractError, RenderedPrompt

_ACTION_TAG = re.compile(r"<action>\s*([^<\n]+?)\s*</action>", re.IGNORECASE)


class TextPolicy(Protocol):
    """A model runner that produces raw text for one fully rendered prompt.

    Implementations should return the raw model completion string for the provided
    `RenderedPrompt`.
    """

    def generate(self, prompt: RenderedPrompt) -> str:
        """Generate a raw completion string for `prompt`.

        :param prompt: RenderedPrompt containing text and action catalogue.
        :returns: Raw model completion string (may include `<action>` tags).
        """
        ...


@dataclass(frozen=True)
class DecodedAction:
    """Validated model decision ready for the environment adapter.

    :ivar action_id: Discrete id defined by the prompt's action catalogue.
    :ivar label: Model-selected action label.
    :ivar raw_text: Unmodified model completion retained for trajectory provenance.
    
    Example:
        >>> DecodedAction(action_id=0, label="move", raw_text="<action>move</action>")
    """

    action_id: int
    label: str
    raw_text: str


@dataclass(frozen=True)
class DecodeMetrics:
    """Observable decoder outcome counters for one inference attempt.

    :ivar invalid_format: int
    :ivar unknown_action: int
    :ivar masked_action: int

    Example:
        >>> obj = DecodeMetrics(invalid_format=..., unknown_action=...)"""

    invalid_format: int = 0
    unknown_action: int = 0
    masked_action: int = 0


def decode_action_outcome(
    prompt: RenderedPrompt, raw_text: str
) -> tuple[DecodedAction | None, DecodeMetrics]:
    """Decode one completion without hiding a model-format failure.

    This helper extracts the first `<action>...</action>` tag and validates it
    against the `RenderedPrompt`'s `ActionCatalog`.

    :param prompt: RenderedPrompt providing the expected action catalogue.
    :param raw_text: Raw model completion string to inspect.
    :returns: A tuple where the first element is a `DecodedAction` if decoding
        succeeded or `None` on format/validation failure, and the second element
        is a `DecodeMetrics` instance describing the failure mode.

    Example:
        >>> decoded, metrics = decode_action_outcome(prompt, raw_text)
    """
    match = _ACTION_TAG.search(raw_text)
    if match is None:
        return None, DecodeMetrics(invalid_format=1)
    label = match.group(1).strip()
    try:
        action_id = prompt.actions.index_of(label)
    except PromptContractError:
        return None, DecodeMetrics(unknown_action=1)
    return DecodedAction(action_id=action_id, label=label, raw_text=raw_text), DecodeMetrics()


def decode_action(prompt: RenderedPrompt, raw_text: str) -> tuple[DecodedAction, DecodeMetrics]:
    """Parse exactly one action tag and validate it against the rendered action catalogue.

    :param prompt: RenderedPrompt input value
    :param raw_text: str input value
    :returns: tuple[DecodedAction, DecodeMetrics]
    :raises PromptContractError: If completion lacks a valid action tag or selects an unknown label.

    Example:
        >>> decoded, metrics = decode_action(prompt, raw_text)
    """
    decoded, metrics = decode_action_outcome(prompt, raw_text)
    if decoded is not None:
        return decoded, metrics
    if metrics.invalid_format:
        raise PromptContractError("model output lacks a non-empty <action>LABEL</action> tag")
    raise PromptContractError("unknown model action in completion")


def act(policy: TextPolicy, prompt: RenderedPrompt) -> tuple[DecodedAction, DecodeMetrics]:
    """Run one text policy and decode its result through the prompt-bound action contract.

    :param policy: TextPolicy input value
    :param prompt: RenderedPrompt input value
    :returns: tuple[DecodedAction, DecodeMetrics]
    :raises PromptContractError: If the model output cannot be parsed into a valid action.

    Example:
        >>> decoded, metrics = act(policy, prompt)
    """
    return decode_action(prompt, policy.generate(prompt))
