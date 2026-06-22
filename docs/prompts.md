# Prompts: среда → модель

`PromptContext` — явная граница между CrafText state и текстовой моделью: goal, observation,
неизменяемый `ActionCatalog`, dialog и safety constraint. `MegaPromptRenderer` адаптирует
vendored MegaPrompts и возвращает `RenderedPrompt`, который хранит тот же catalogue.

Это важно: текст модели никогда не декодируется в action id без catalogue, а незнакомая метка
даёт `PromptContractError` вместо silent fallback.

```bash
pyenv exec python -m uv sync --extra prompts
pyenv exec python -m uv run pytest -m integration tests/integration/test_megaprompts_vendor.py
```

`TextPolicy` уже получает только `RenderedPrompt`; `decode_action()` принимает строго
`<action>…</action>` и валидирует label через тот же `ActionCatalog`, возвращая `DecodedAction`
с action id и исходным model text для provenance. Сквозной integration smoke покрывает
`environment-shaped state → prompt → policy → decoder → adapter.step`.

`collect_text_episode()` поддерживает два явных режима ошибки: `invalid_action="error"`
останавливает прогон, а `invalid_action="fallback"` требует явный допустимый
`fallback_action_id`. Во втором случае replay v2 записывает `invalid_format`,
`unknown_action`, `fallback_used`, raw completion и (если backend их выдал) per-token
logprobs вместе с generated token ids. Поэтому fallback не может стать неявной подменой
действия, а replay уже содержит token-level provenance для будущего PPO/SFT bridge.

`collect_text_episode` — host-side reference pipeline: он последовательно выполняет
`PromptContext → RenderedPrompt → LlmBackend → strict decode_action → CrafTextAdapter.step`
и возвращает versioned `ReplayArtifact`. Это намеренно не JIT path: LLM I/O и текстовый
parser остаются на host, а JAX rollout используется для численного обучения после сбора.
