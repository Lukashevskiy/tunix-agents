from tunix_craftext.env.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.models.llm import LlmRequest, ScriptedLlmBackend


def test_scripted_llm_preserves_raw_completion_for_replay() -> None:
    request = LlmRequest(RenderedPrompt("prompt", ActionCatalog(("DO",)), "base"))
    response = ScriptedLlmBackend("<action>DO</action>").complete(request)
    assert response.raw_text == "<action>DO</action>"
    assert response.backend == "scripted"


def test_scripted_llm_batch_preserves_request_order() -> None:
    """A batch-capable backend returns exactly one response per ordered request."""
    requests = (
        LlmRequest(RenderedPrompt("first", ActionCatalog(("DO",)), "base")),
        LlmRequest(RenderedPrompt("second", ActionCatalog(("DO",)), "base")),
    )

    responses = ScriptedLlmBackend("<action>DO</action>").complete_batch(requests)

    assert len(responses) == len(requests)
    assert all(response.raw_text == "<action>DO</action>" for response in responses)
