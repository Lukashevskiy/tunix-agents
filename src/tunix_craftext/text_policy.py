"""Typed text-policy boundary from rendered prompts to discrete environment actions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .prompts import PromptContractError, RenderedPrompt


_ACTION_TAG = re.compile(r"<action>\s*([^<\n]+?)\s*</action>", re.IGNORECASE)


class TextPolicy(Protocol):
    """A model runner that produces raw text for one fully rendered prompt."""

    def generate(self, prompt: RenderedPrompt) -> str: ...


@dataclass(frozen=True)
class DecodedAction:
    """Validated model decision ready for the environment adapter.

    :ivar action_id: Discrete id defined by the prompt's action catalogue.
    :ivar label: Model-selected action label.
    :ivar raw_text: Unmodified model completion retained for trajectory provenance.
    """

    action_id: int
    label: str
    raw_text: str


@dataclass(frozen=True)
class DecodeMetrics:
    """Observable decoder outcome counters for one inference attempt."""

    invalid_format: int = 0
    unknown_action: int = 0


def decode_action(prompt: RenderedPrompt, raw_text: str) -> tuple[DecodedAction, DecodeMetrics]:
    """Parse exactly one action tag and validate it against the rendered action catalogue.

    :raises PromptContractError: If completion lacks a valid action tag or selects an unknown label.
    """
    match = _ACTION_TAG.search(raw_text)
    if match is None:
        raise PromptContractError("model output lacks a non-empty <action>LABEL</action> tag")
    label = match.group(1).strip()
    try:
        action_id = prompt.actions.index_of(label)
    except PromptContractError as error:
        raise PromptContractError(f"unknown model action {label!r}") from error
    return DecodedAction(action_id=action_id, label=label, raw_text=raw_text), DecodeMetrics()


def act(policy: TextPolicy, prompt: RenderedPrompt) -> tuple[DecodedAction, DecodeMetrics]:
    """Run one text policy and decode its result through the prompt-bound action contract."""
    return decode_action(prompt, policy.generate(prompt))
