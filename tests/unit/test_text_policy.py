import pytest

from tunix_craftext.prompts import ActionCatalog, PromptContractError, RenderedPrompt
from tunix_craftext.text_policy import decode_action


def test_decoder_maps_model_action_to_the_prompt_bound_id() -> None:
    prompt = RenderedPrompt("choose", ActionCatalog(("LEFT", "DO")), "base")
    decoded, metrics = decode_action(prompt, "reasoning\n<action>DO</action>")

    assert decoded.action_id == 1
    assert decoded.label == "DO"
    assert metrics.invalid_format == metrics.unknown_action == 0


@pytest.mark.parametrize("completion", ["", "<action></action>", "DO"])
def test_decoder_rejects_missing_action_tag(completion: str) -> None:
    prompt = RenderedPrompt("choose", ActionCatalog(("LEFT",)), "base")
    with pytest.raises(PromptContractError, match="action"):
        decode_action(prompt, completion)


def test_decoder_rejects_unknown_action() -> None:
    prompt = RenderedPrompt("choose", ActionCatalog(("LEFT",)), "base")
    with pytest.raises(PromptContractError, match="unknown"):
        decode_action(prompt, "<action>FLY</action>")
