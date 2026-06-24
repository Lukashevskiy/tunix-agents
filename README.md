# Tunix CrafText

Это намеренно компактный каркас обучения CrafText и CagedCrafText на JAX.
Он заменяет не уровень окружения, а уровень оркестрации: тестируемый контракт траекторий,
чистый JAX rollout, обновления Optax, checkpointing Orbax и узкий адаптер для Tunix.

## Статус

**Sync rollout / contract-first training path.** Vendored окружения и prompt assets скопированы
без изменений в `vendor/`; лицензии и атрибуция остались там. Итоговый локальный путь уже
собирает `CrafText state batch → MegaPrompts → batched Tunix/Qwen → strict decode/action-mask
fallback → CrafText vmap(step) → replay v3 → TextTrajectoryBatch → masked token PPO smoke`.
Полноценно распределённый `RLCluster` workload, trainable actor/critic и новые objectives
вроде DPO/GRPO остаются следующими этапами и должны подключаться через typed registry/batch
contracts, а не менять transport среды.

## Быстрый старт

```bash
pyenv install --skip-existing 3.12.13
pyenv local 3.12.13
pyenv exec python -m pip install --upgrade uv
pyenv exec python -m uv sync --extra dev --extra docs
uv run pytest
```

`pyenv` управляет интерпретатором проекта; `uv.lock` хранит разрешённый граф зависимостей; `.venv`
— одноразовое окружение, созданное `uv`. Смотрите [практику среды разработки](docs/development.md)
для обновлений, opt-in extras и CI правил.

Для реальных env и Tunix установите opt-in extras после фиксации совместимой accelerator-specific
сборки JAX:

```bash
pyenv exec python -m uv sync --extra envs --extra tunix --extra replay --extra dev
```

## Внешние проекты: что мы перенимаем и где ставим границу

Мы не копируем чужие training loops в core. Каждый проект имеет закреплённую роль и отдельный
compatibility record в [`compatibility/training-stack.yaml`](compatibility/training-stack.yaml).

| Источник | Практика / роль у нас | Граница применения |
| --- | --- | --- |
| [Google Tunix](https://github.com/google/tunix) | `RLCluster`, Agentic GRPO, role meshes, model/learner API | Golden distributed training path; не заменяет typed environment/replay contracts. |
| [jax-lm](https://github.com/chuyishang/jax-lm) | discipline static shapes, mesh divisibility и preflight до загрузки весов | Вдохновляет `preflight.py`; **не** inference engine и не dependency. |
| [Qwix](https://github.com/google/qwix) | QLoRA/quantized JAX–Flax experiments | Только после output-parity и trainability fixture выбранной Tunix model. |
| [Flashbax](https://github.com/instadeepai/flashbax) | JIT-compatible bounded replay staging | Исследовательский sync path; не превращает on-policy batch в неявный off-policy replay. |
| [CommonLoopUtils](https://github.com/google/CommonLoopUtils) | structured metrics и checkpoint loop practice | Optional reporting/checkpoint layer после стабилизации JSONL evidence. |
| [mpi4jax](https://github.com/mpi4jax/mpi4jax) | explicit multi-host collectives | Только будущая async/multi-host фаза и только с target-hardware fixture. |
| [Anakin paper](https://arxiv.org/abs/2104.06272) | JAX-first actor–learner: static PyTrees, compiled numerical path, host I/O вне `jit` | Архитектурный принцип для sync-first, затем async scale-out. |

Vendored `CrafText`, `CagedCrafText` и `MegaPrompts` сохраняются неизменёнными в `vendor/`;
наша работа живёт в typed adapters и runtime boundaries, поэтому upstream snapshots можно
обновлять и проверять отдельно.

## Локальный запуск сайта

Создайте локальное окружение документации один раз, затем используйте команду репозитория вместо
глобальной установки MkDocs:

```bash
pyenv exec python -m uv sync --extra dev --extra docs
make serve
```

`make serve` сперва обновляет Dashboard текущим Git commit, roadmap, inventory возможностей и
`artifacts/benchmarks/*.json`, затем запускает локальный MkDocs server. Используйте `make docs`
для сборки того же статического `site/` без сервера. Benchmark JSON артефакты из
`artifacts/benchmarks/` появляются автоматически при следующем билде. GitHub Pages workflow делает
tоже самое при каждом пуше в `main`, еженедельно или вручную.

Без Make запустите `.venv/bin/python scripts/generate_dashboard.py && .venv/bin/python -m mkdocs
serve`. Не используйте просто `mkdocs serve`: он может выбрать глобальный интерпретатор без
Material и не обновит сгенерированные страницы.

Каждое изменение проходит через [Definition of Done](docs/delivery.md): аудит, применимые
tесты и доказательства производительности, обновление документации/статуса, intentional commit и
сборка сайта.

Прочитайте [план выполнения](docs/plan.md), [архитектуру](docs/architecture.md),
[интеграцию с Tunix](docs/tunix.md), [код/API](docs/code-reference.md) и [примеры](docs/examples.md) перед расширением тренера.
Notebook 07 показывает batched Qwen/Tunix rollout и replay export, 09/11/12 доводят тот же
pipeline до replay→token batch→masked PPO smoke cycle.
