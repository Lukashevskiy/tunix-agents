"""Batch Qwen/Tunix sampler boundary tests without local model weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tunix_craftext.env.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.models.llm import LlmRequest
from tunix_craftext.models.tunix_adapter import QWEN_ACTION_SYSTEM_PROMPT, QwenTunixBackend


class _Tokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        del add_generation_prompt, tokenize
        return "|".join(message["content"] for message in messages)

    def encode(self, text):
        return list(range(len(text.split())))

    def bos_id(self):
        return 0

    def pad_id(self):
        return 0


class _Sampler:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, input_strings, **kwargs):
        self.calls.append({"input_strings": input_strings, **kwargs})
        return SimpleNamespace(
            text=[f"<action>DO</action>{index}" for index, _ in enumerate(input_strings)],
            logprobs=[[-1.0], [-2.0]],
            tokens=[[10], [20]],
            padded_prompt_tokens=[[1, 0], [2, 0]],
        )


def _backend() -> tuple[QwenTunixBackend, _Sampler]:
    backend = QwenTunixBackend.__new__(QwenTunixBackend)
    sampler = _Sampler()
    backend._tokenizer = _Tokenizer()
    backend._sampler = sampler
    backend._cache_size = 128
    backend._seed = 7
    return backend, sampler


def _request(text: str, *, max_new_tokens: int = 4) -> LlmRequest:
    return LlmRequest(RenderedPrompt(text, ActionCatalog(("DO",)), "base"), max_new_tokens)


def test_qwen_backend_completes_a_prompt_batch_in_one_sampler_call() -> None:
    """Chat templates and provenance stay aligned with the input request ordering."""
    backend, sampler = _backend()

    responses = backend.complete_batch((_request("first"), _request("second")))

    assert len(sampler.calls) == 1
    assert sampler.calls[0]["input_strings"] == [
        f"{QWEN_ACTION_SYSTEM_PROMPT}|first",
        f"{QWEN_ACTION_SYSTEM_PROMPT}|second",
    ]
    assert [response.raw_text for response in responses] == [
        "<action>DO</action>0",
        "<action>DO</action>1",
    ]
    assert [response.token_ids for response in responses] == [(10,), (20,)]
    assert [response.prompt_token_ids for response in responses] == [(1,), (2,)]


def test_qwen_batch_rejects_nonuniform_generation_contract() -> None:
    """One Tunix sampler invocation has one static generation length."""
    backend, _ = _backend()

    with pytest.raises(ValueError, match="max_new_tokens"):
        backend.complete_batch(
            (_request("first", max_new_tokens=2), _request("second", max_new_tokens=4))
        )
