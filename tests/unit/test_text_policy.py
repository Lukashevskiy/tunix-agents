import pytest

from tunix_craftext.env.prompts import ActionCatalog, PromptContractError, RenderedPrompt
from tunix_craftext.env.text_policy import decode_action, decode_action_outcome


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


@pytest.mark.parametrize(
    ("completion", "invalid_format", "unknown_action"),
    [("not an action", 1, 0), ("<action>FLY</action>", 0, 1)],
)
def test_decoder_outcome_preserves_invalid_action_reason(
    completion: str, invalid_format: int, unknown_action: int
) -> None:
    prompt = RenderedPrompt("choose", ActionCatalog(("LEFT",)), "base")

    decoded, metrics = decode_action_outcome(prompt, completion)

    assert decoded is None
    assert (metrics.invalid_format, metrics.unknown_action) == (invalid_format, unknown_action)
