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

Следующий слой — `TextPolicy`: он получает только `RenderedPrompt`, возвращает сырой model output,
а отдельный decoder валидирует `<action>…</action>` через `ActionCatalog` и записывает invalid-action
metric в trajectory artifact.
