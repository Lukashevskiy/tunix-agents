"""Backend-neutral LLM request/response contracts for prompt-driven actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .prompts import RenderedPrompt


@dataclass(frozen=True)
class LlmRequest:
    prompt: RenderedPrompt
    max_new_tokens: int = 32
    temperature: float = 0.0
    stop_sequences: tuple[str, ...] = ("</action>",)


@dataclass(frozen=True)
class LlmResponse:
    raw_text: str
    backend: str
    model: str
    latency_ms: float | None = None
    token_logprobs: tuple[float, ...] | None = None


class LlmBackend(Protocol):
    def complete(self, request: LlmRequest) -> LlmResponse: ...


@dataclass(frozen=True)
class ScriptedLlmBackend:
    raw_text: str
    model: str = "scripted"

    def complete(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(self.raw_text, "scripted", self.model, 0.0)
