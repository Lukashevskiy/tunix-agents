# Tunix CrafText

Это отдельный репозиторий для обучения агентов CrafText без VERL/Ray-оркестрации.
Он использует JAX как вычислительное ядро, Flax для состояния модели, Optax для
оптимизации, Orbax для checkpointing и оставляет Tunix единственным внешним слоем
LLM/RL-обучения.

## Принципы

- Среда — источник правды; обучение не меняет семантику CrafText/CagedCrafText.
- Горячий путь — статические массивы JAX с явными `[T, B, ...]` осями.
- Каждый новый алгоритм сначала получает контрактные тесты, затем parity/integration
  тест, и лишь затем JIT/performance-реализацию.
- Эксперимент воспроизводим по config, seed, версии кода, версии vendor и hardware.
- Документация — продукт сборки: план, ADR, диаграммы, benchmark-артефакты и Git
  provenance публикуются вместе.

> Важно: Tunix остаётся opt-in boundary. Core не импортирует его при обычном unit path;
> проверенный bridge живёт в `tunix_adapter.py`, а сбор CrafText траекторий и loss contracts
> остаются framework-neutral.

## Локальный запуск сайта

После установки `mkdocs-material` в `.venv` выполните `make serve`. Makefile предпочитает
`.venv/bin/python`, поэтому Material и MkDocs гарантированно берутся из одного interpreter.
Команда сперва обновит Dashboard текущим Git commit, прогрессом плана и benchmark artifacts,
затем запустит MkDocs. Для статичной сборки используйте `make docs`. API reference с
автоматически отображаемыми type hints собирается строго командой `make api-docs` в `site/api/`;
он также входит в `make verify`.
