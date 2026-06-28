"""Contract tests for Qwen prompt-token and hidden-state feature boundaries."""

from tunix_craftext.models.tunix_adapter import qwen_chat_token_ids


def test_qwen_chat_token_ids_adds_bos_only_when_tokenizer_declares_one() -> None:
    class Tokenizer:
        def encode(self, text: str) -> list[int]:
            assert text == "prompt"
            return [4, 5]

        def bos_id(self) -> int:
            return 3

    assert qwen_chat_token_ids(Tokenizer(), "prompt") == (3, 4, 5)
