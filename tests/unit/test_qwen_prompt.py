"""Contract tests for model-specific Qwen action prompt preparation."""

from __future__ import annotations

import pytest

from tunix_craftext.tunix_adapter import (
    QWEN_ACTION_SYSTEM_PROMPT,
    format_qwen_action_prompt,
    qwen_required_cache_size,
)


class RecordingTokenizer:
    """Minimal public-tokenizer double for deterministic chat-template tests."""

    def __init__(self, response: str | list[int] = "formatted") -> None:
        self.response = response
        self.messages: list[dict[str, str]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str | list[int]:
        assert add_generation_prompt
        assert not tokenize
        self.messages = messages
        return self.response

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def bos_id(self) -> int:
        return 1


def test_qwen_prompt_uses_tokenizer_declared_chat_template() -> None:
    tokenizer = RecordingTokenizer("<chat>prompt</chat>")

    actual = format_qwen_action_prompt(tokenizer, "actions: NOOP, DO")

    assert actual == "<chat>prompt</chat>"
    assert tokenizer.messages == [
        {"role": "system", "content": QWEN_ACTION_SYSTEM_PROMPT},
        {"role": "user", "content": "actions: NOOP, DO"},
    ]


@pytest.mark.parametrize("prompt", ["", "  "])
def test_qwen_prompt_rejects_blank_rendered_prompt(prompt: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        format_qwen_action_prompt(RecordingTokenizer(), prompt)


def test_qwen_prompt_rejects_token_ids_from_text_sampler_boundary() -> None:
    with pytest.raises(ValueError, match="text"):
        format_qwen_action_prompt(RecordingTokenizer([1, 2]), "actions: NOOP")


def test_qwen_cache_requirement_matches_tunix_power_of_two_prompt_padding() -> None:
    tokenizer = RecordingTokenizer()

    required = qwen_required_cache_size(tokenizer, "one two three", max_new_tokens=8)

    assert required == 12
