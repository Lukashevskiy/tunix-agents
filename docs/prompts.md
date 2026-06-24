# Prompts: среда → модель

`PromptContext` — явная граница между CrafText state и текстовой моделью: goal, renderer payload,
неизменяемый `ActionCatalog`, dialog и safety constraint. Для vendored MegaPrompts canonical
payload — structured CrafText `EnvState` (map, inventory, coordinates); pixel observation
сохраняется отдельно для численной JAX policy. `MegaPromptRenderer` возвращает `RenderedPrompt`,
который хранит тот же catalogue.

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

`collect_batched_text_decision()` и `collect_batched_text_rollout()` поддерживают два явных
режима ошибки: `invalid_action="error"` останавливает прогон, а `invalid_action="fallback"`
требует явный допустимый `fallback_action_id`. После strict decode они дополнительно enforce-ят
текущий environment `action_mask`: masked model action либо падает, либо становится observable
fallback. Replay v3 записывает `invalid_format`, `unknown_action`, `masked_action`,
`fallback_used`, raw completion и (если backend их выдал) prompt/generated token ids с per-token
logprobs. Поэтому fallback не может стать неявной подменой действия, а replay уже содержит
token-level provenance для будущего PPO/SFT/DPO bridge.

Итоговый host-side text pipeline выполняет
`CrafText EnvState batch → PromptContext → RenderedPrompt → BatchLlmBackend/Tunix → strict decode/action-mask fallback → jax.vmap(CrafTextAdapter.step)`

Для Agentic CrafText `compose_craftext_goal()` делает user-facing objective
template-visible: он объединяет `task.goal` с выбранной scenario instruction,
world preset и, для CagedCrafText, textual safety constraint. Это необходимо,
потому что MegaPrompts template может не выводить optional metadata fields сам.
Scenario никогда не заменяет пользовательскую задачу.
и возвращает per-env versioned `ReplayArtifact`. Это намеренно не JIT path для LLM I/O и
текстового parser; JAX остаётся владельцем environment stepping, token loss и learner math.
