# Среда разработки: uv-only

Проект использует два независимых уровня воспроизводимости:

- `uv python` устанавливает и выбирает точный CPython из `.python-version`.
- `uv sync` разрешает зависимости в `uv.lock`, создаёт `.venv` и запускает команды в ней.

Это исключает случайный выбор Homebrew/system Python и глобальных пакетов. `.venv` —
производный артефакт: его не редактируют вручную и можно безопасно пересоздать.

## Первый запуск

```bash
uv python install 3.12.13
uv python pin 3.12.13
uv sync --extra dev --extra docs
uv run pytest tests/unit
```

Используем bare `uv`: он сам читает `.python-version`, скачивает нужный CPython при
`uv python install`, создаёт `.venv` и запускает команды через project environment.

## Повседневный цикл

```bash
uv run pytest tests/unit
uv run make verify
uv run make perf
```

`make` вызывает `uv run python`, так что генератор dashboard, MkDocs и Sphinx всегда работают
из lockfile-окружения. Для notebooks: `uv sync --extra examples`.
Для полной рабочей среды (dev, docs, envs, prompts и Tunix) используйте
`uv sync --all-extras`. Один `--extra tunix` синхронизирует
только base + Tunix и временно убирает остальные extras из текущей venv.

Для офлайн-проверок после уже выполненного `uv sync` можно запретить implicit sync:

```bash
UV_RUN_FLAGS=--no-sync make docs
UV_RUN_FLAGS=--no-sync make test
```

Для GPU JAX сначала синхронизируйте проект, потом поставьте platform-specific wheel:

```bash
uv sync --all-extras
uv pip install -U "jax[cuda13]"  # или "jax[cuda12]" под драйвер сервера
uv run python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

Если позже снова выполнить `uv sync`, повторите `uv pip install -U "jax[cuda13]"`, чтобы lockfile
не вернул CPU-only `jaxlib` на accelerator машине.

## Изменение зависимостей

После изменения `pyproject.toml` обновить lockfile, синхронизировать среду и проверить diff:

```bash
uv lock
uv sync --extra dev --extra docs
uv run make verify
```

`uv.lock` коммитится вместе с изменением зависимостей. Не заменять lockfile вручную и не делать
`pip install` в `.venv`: это создаёт незафиксированный drift. Accelerator-specific JAX wheels
фиксируются отдельным совместимостным изменением с платформой, backend и benchmark evidence.

## Tensor contracts: jaxtyping

`jaxtyping` — прямая зависимость проекта. Общие aliases лежат в
`tunix_craftext.core.tensor_types`: `TimeBatchFloat`/`TimeBatchBool` означают rollout `[T, B]`,
`TokenBatch*` — padded text batch `[B, L]`, а `ActionMask` — legal actions `[B, A]`.
Новые numerical public APIs должны использовать эти aliases или явный `jaxtyping` shape/dtype,
а не голый `jax.Array`. Для LLM/RL boundary уже есть отдельные aliases:
`PromptTokenBatch*`, `CausalInputTokenBatchInt`, `CausalLmLogits`, `CausalLmHidden`,
`TokenBatchLogits`, `TokenBatchHidden`, `ActionLogits`, `ValueHeadKernel` и `JaxKey`.
`JaxArray` и `JaxTree` остаются escape hatch только для приватных helpers, vendor protocols
и opaque state/params, где форма действительно принадлежит внешней библиотеке.

Это статический контракт, не замена boundary validation: `RolloutBatch.validate()` и
`TextTrajectoryBatch.validate_static()` остаются явными runtime checks. Не добавляйте
Python runtime typechecker внутрь `jit`, `vmap` или `scan`: он создаст host-side work и
исказит performance path. Shape DSL локализован в `tensor_types.py`; для него настроено
узкое Ruff исключение, потому что линтер не разбирает строки осей `jaxtyping`.
