"""Anti-corruption adapter from variable-length LLM outputs to Tunix tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..models.llm import LlmResponse
from .contracts import GenerationResult, InferenceBackendError


@dataclass(frozen=True)
class TunixGenerationTensors:
    """Strict padded tensors consumed by Tunix/GRPO/PPO builders.

    :param generation_tokens: Token ids padded to ``[B, T]``.
    :param actor_log_probs: Behaviour/actor log-probs padded to ``[B, T]``.
    :param generation_mask: Valid-token mask with ``True`` for generated tokens.
    :param raw_text: Generated text aligned with the leading batch axis.
    """

    generation_tokens: NDArray[np.int32]
    actor_log_probs: NDArray[np.float32]
    generation_mask: NDArray[np.bool_]
    raw_text: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate strict Tunix-compatible leading axes."""
        shape = self.generation_tokens.shape
        if len(shape) != 2:
            raise InferenceBackendError("generation_tokens must have shape [B, T]")
        if self.actor_log_probs.shape != shape:
            raise InferenceBackendError("actor_log_probs must match generation_tokens shape")
        if self.generation_mask.shape != shape:
            raise InferenceBackendError("generation_mask must match generation_tokens shape")
        if self.raw_text and len(self.raw_text) != shape[0]:
            raise InferenceBackendError("raw_text length must match batch size")


@dataclass(frozen=True)
class VllmToTunixAdapter:
    """Convert vLLM/normalized responses into padded Tunix arrays."""

    pad_token_id: int = 0
    max_length: int | None = None
    missing_logprob: float = 0.0
    allow_missing_logprobs: bool = False

    def convert_generation_result(self, result: GenerationResult) -> TunixGenerationTensors:
        """Convert normalized project generation result to padded arrays."""
        return self.convert_responses(result.responses)

    def convert_responses(self, responses: Sequence[LlmResponse]) -> TunixGenerationTensors:
        """Convert normalized ``LlmResponse`` objects to padded arrays."""
        rows: list[_GenerationRow] = []
        for index, response in enumerate(responses):
            if not response.token_ids:
                raise InferenceBackendError(f"response {index} is missing generated token ids")
            logprobs = self._logprobs_or_default(
                response.token_logprobs,
                token_count=len(response.token_ids),
                index=index,
            )
            rows.append(
                _GenerationRow(
                    tokens=tuple(int(token) for token in response.token_ids),
                    logprobs=logprobs,
                    raw_text=response.raw_text,
                )
            )
        return self._pad_rows(rows)

    def convert(self, outputs: Sequence[object]) -> TunixGenerationTensors:
        """Convert raw vLLM-like ``RequestOutput`` objects to padded arrays."""
        rows = [
            _row_from_vllm_output(output, index=index, adapter=self)
            for index, output in enumerate(outputs)
        ]
        return self._pad_rows(rows)

    def _logprobs_or_default(
        self,
        logprobs: Sequence[float] | None,
        *,
        token_count: int,
        index: int,
    ) -> tuple[float, ...]:
        if logprobs is None:
            if self.allow_missing_logprobs:
                return tuple(self.missing_logprob for _ in range(token_count))
            raise InferenceBackendError(f"response {index} is missing token logprobs")
        if len(logprobs) != token_count:
            raise InferenceBackendError(
                f"response {index} token/logprob lengths differ: {token_count} vs {len(logprobs)}"
            )
        return tuple(float(value) for value in logprobs)

    def _pad_rows(self, rows: Sequence[_GenerationRow]) -> TunixGenerationTensors:
        if not rows:
            raise InferenceBackendError("cannot convert an empty generation batch")
        width = self.max_length or max(len(row.tokens) for row in rows)
        if width <= 0:
            raise InferenceBackendError("max_length must be positive")
        batch = len(rows)
        tokens = np.full((batch, width), self.pad_token_id, dtype=np.int32)
        logprobs = np.full((batch, width), self.missing_logprob, dtype=np.float32)
        mask = np.zeros((batch, width), dtype=np.bool_)
        raw_text: list[str] = []
        for row_index, row in enumerate(rows):
            length = len(row.tokens)
            if length > width:
                raise InferenceBackendError(
                    f"response {row_index} length {length} exceeds max_length {width}"
                )
            tokens[row_index, :length] = np.asarray(row.tokens, dtype=np.int32)
            logprobs[row_index, :length] = np.asarray(row.logprobs, dtype=np.float32)
            mask[row_index, :length] = True
            raw_text.append(row.raw_text)
        return TunixGenerationTensors(tokens, logprobs, mask, tuple(raw_text))


@dataclass(frozen=True)
class _GenerationRow:
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    raw_text: str


def _row_from_vllm_output(
    output: object, *, index: int, adapter: VllmToTunixAdapter
) -> _GenerationRow:
    choices = getattr(output, "outputs", None)
    if not choices:
        raise InferenceBackendError(f"vLLM output {index} has no choices")
    choice = choices[0]
    token_ids = tuple(int(token) for token in getattr(choice, "token_ids", ()) or ())
    if not token_ids:
        raise InferenceBackendError(f"vLLM output {index} is missing generated token ids")
    logprobs = _extract_vllm_logprobs(getattr(choice, "logprobs", None))
    return _GenerationRow(
        tokens=token_ids,
        logprobs=adapter._logprobs_or_default(logprobs, token_count=len(token_ids), index=index),
        raw_text=str(getattr(choice, "text", "")),
    )


def _extract_vllm_logprobs(raw_logprobs: object) -> tuple[float, ...] | None:
    """Extract selected-token logprobs from vLLM V0/V1-like payloads."""
    if raw_logprobs is None:
        return None
    if isinstance(raw_logprobs, (str, bytes, Mapping)) or not isinstance(
        raw_logprobs, Sequence
    ):
        return None
    values: list[float] = []
    for item in raw_logprobs:
        if isinstance(item, dict):
            if not item:
                return None
            first = next(iter(item.values()))
            values.append(float(getattr(first, "logprob", first)))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            values.append(float(item))
        else:
            logprob = getattr(item, "logprob", None)
            if logprob is None:
                return None
            values.append(float(logprob))
    return tuple(values)
