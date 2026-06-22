# Tunix CrafText

Это намеренно компактный каркас обучения CrafText и CagedCrafText на JAX.
Он заменяет не уровень окружения, а уровень оркестрации: тестируемый контракт траекторий,
чистый JAX rollout, обновления Optax, checkpointing Orbax и узкий адаптер для Tunix.

## Статус

**Foundation / contract-first.** Vendored окружения и prompt assets скопированы без
изменений в `vendor/`; лицензии и атрибуция остались там. Локальный smoke-путь Tunix/Qwen
уже реализован, однако полноценно распределённый RLCluster workload и GRPO остаются
следующими этапами.

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
pyenv exec python -m uv sync --extra envs --extra tunix --extra dev
```

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

Прочитайте [план выполнения](docs/plan.md), [архитектуру](docs/architecture.md) и
[стратегию тестирования/benchmark](docs/quality.md) перед расширением тренера.
