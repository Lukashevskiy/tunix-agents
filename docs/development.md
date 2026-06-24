# Среда разработки: pyenv + uv

Проект использует два независимых уровня воспроизводимости:

- `pyenv` выбирает точный CPython из `.python-version`.
- `uv` разрешает зависимости в `uv.lock`, создаёт `.venv` и запускает команды в ней.

Это исключает случайный выбор Homebrew/system Python и глобальных пакетов. `.venv` —
производный артефакт: его не редактируют вручную и можно безопасно пересоздать.

## Первый запуск

```bash
pyenv install --skip-existing 3.12.13
pyenv local 3.12.13
pyenv exec python -m pip install --upgrade uv
pyenv exec python -m uv sync --extra dev --extra docs
pyenv exec python -m uv run pytest tests/unit
```

Используем `pyenv exec python -m uv`, а не bare `uv`: это остаётся корректным, даже если
`~/.pyenv/shims` ещё не добавлен в `PATH`. После один раз настроенного shell допустима короткая
форма `uv run …`, но CI и документация используют явную форму.

## Повседневный цикл

```bash
pyenv exec python -m uv run pytest tests/unit
pyenv exec python -m uv run make verify
pyenv exec python -m uv run make perf
```

`make` по-прежнему вызывает `.venv/bin/python`, так что генератор dashboard, MkDocs и Sphinx
всегда работают из lockfile-окружения. Для notebooks: `pyenv exec python -m uv sync --extra examples`.
Для полной рабочей среды (dev, docs, envs, prompts и Tunix) используйте
`pyenv exec python -m uv sync --all-extras`. Один `--extra tunix` синхронизирует
только base + Tunix и временно убирает остальные extras из текущей venv.

## Изменение зависимостей

После изменения `pyproject.toml` обновить lockfile, синхронизировать среду и проверить diff:

```bash
pyenv exec python -m uv lock
pyenv exec python -m uv sync --extra dev --extra docs
pyenv exec python -m uv run make verify
```

`uv.lock` коммитится вместе с изменением зависимостей. Не заменять lockfile вручную и не делать
`pip install` в `.venv`: это создаёт незафиксированный drift. Accelerator-specific JAX wheels
фиксируются отдельным совместимостным изменением с платформой, backend и benchmark evidence.

## Tensor contracts: jaxtyping

`jaxtyping` — прямая зависимость проекта. Общие aliases лежат в
`tunix_craftext.tensor_types`: `TimeBatchFloat`/`TimeBatchBool` означают rollout `[T, B]`,
`TokenBatch*` — padded text batch `[B, L]`, а `ActionMask` — legal actions `[B, A]`.
Новые numerical public APIs должны использовать эти aliases или явный `jaxtyping` shape/dtype,
а не голый `jax.Array`.

Это статический контракт, не замена boundary validation: `RolloutBatch.validate()` и
`TextTrajectoryBatch.validate_static()` остаются явными runtime checks. Не добавляйте
Python runtime typechecker внутрь `jit`, `vmap` или `scan`: он создаст host-side work и
исказит performance path. Shape DSL локализован в `tensor_types.py`; для него настроено
узкое Ruff исключение, потому что линтер не разбирает строки осей `jaxtyping`.
