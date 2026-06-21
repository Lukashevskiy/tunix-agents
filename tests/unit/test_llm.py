from tunix_craftext.llm import LlmRequest, ScriptedLlmBackend
from tunix_craftext.prompts import ActionCatalog, RenderedPrompt

def test_scripted_llm_preserves_raw_completion_for_replay() -> None:
    request = LlmRequest(RenderedPrompt("prompt", ActionCatalog(("DO",)), "base"))
    response = ScriptedLlmBackend("<action>DO</action>").complete(request)
    assert response.raw_text == "<action>DO</action>"
    assert response.backend == "scripted"
