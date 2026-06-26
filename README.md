# Tunix CrafText

Это намеренно компактный каркас обучения CrafText и CagedCrafText на JAX.
Он заменяет не уровень окружения, а уровень оркестрации: тестируемый контракт траекторий,
чистый JAX rollout, обновления Optax, checkpointing Orbax и узкий адаптер для Tunix.

## Статус

**Sync rollout / contract-first training path.** Vendored окружения и prompt assets скопированы
без изменений в `vendor/`; лицензии и атрибуция остались там. Итоговый локальный путь уже
собирает `CrafText state batch → MegaPrompts → batched Tunix Qwen/Gemma → strict
decode/action-mask fallback → CrafText vmap(step) → replay v3 → TextTrajectoryBatch`.
Для LLM-RL пути уже есть реальный Tunix actor boundary: Gemma/Qwen пересчитывают token
logprobs/entropy, отдельный critic role считает values, а PPO objective проверяет returns,
advantages и loss без toy learner в финальном notebook. PPO `RLCluster` constructor boundary
теперь тоже есть: actor/reference/critic/tokenizer assets создаются явно и передаются в public
Tunix `RLCluster`. Trainable LoRA/Qwix update, PPOLearner batch schema adapter и async rollout
остаются следующими этапами и должны подключаться через typed registry/batch contracts, а не
менять transport среды.

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

## CLI: единая точка управления проектом

Проект переходит от набора отдельных `scripts/*.py` к полновесному CLI `tunix-craftext`
с коротким alias `tcx`. CLI спроектирован как thin orchestration layer: он валидирует профили,
показывает статус, запускает use-case слой и пишет evidence, но не хранит внутри себя
логику CrafText, MegaPrompts, Tunix/RLCluster или benchmark runner.

Пока CLI находится в стадии реализации. Целевой первый gate:

```bash
tunix-craftext profile validate configs/grpo/qwen_agentic_local.yaml
tunix-craftext profile evidence configs/grpo/qwen_agentic_local.yaml \
  --output artifacts/runs/qwen-agentic-craftext-local-smoke/provenance.json
tunix-craftext verify golden
```

До появления console script используйте уже работающие эквиваленты:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_agentic_grpo.py \
  --profile configs/grpo/qwen_agentic_local.yaml --allow-cpu-smoke
make verify-golden
```

Для быстрой проверки GRPO без загрузки модели есть два безопасных режима:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_agentic_grpo.py \
  --profile configs/grpo/qwen_agentic_local.yaml --dry-run

PYTHONPATH=src .venv/bin/python scripts/run_agentic_grpo.py \
  --profile configs/grpo/qwen_agentic_local.yaml --scripted-smoke
```

`--dry-run` валидирует profile/topology/preflight/evidence. `--scripted-smoke` запускает реальный
CrafText agentic tool-call loop с несколькими GRPO generations и group-normalized advantages, но
без LLM/RLCluster weights. Это рабочий local gate перед реальным `GRPOLearner`; PPO/PPO-Lag/CPO
потом добавляют value critic и cost critic поверх уже проверенного rollout/tool transport.

Планируемое дерево команд:

| Команда | Для чего |
| --- | --- |
| `tunix-craftext profile validate/evidence/inspect` | проверить versioned YAML, SHA256, model/vendor provenance без загрузки весов |
| `tunix-craftext env smoke/step/inspect` | проверить CrafText/Caged adapter, instruction, world preset, legal actions |
| `tunix-craftext prompt render/decode/replay` | собрать MegaPrompts prompt и проверить strict action decoder |
| `tunix-craftext rollout random/text/agentic` | собрать replay/trajectory JSONL без update |
| `tunix-craftext train grpo` | запустить golden Tunix Agentic GRPO pipeline из profile |
| `tunix-craftext eval checkpoint/reference` | оценить checkpoint или frozen reference на fixed tasks |
| `tunix-craftext benchmark env/text/agentic` | записать performance evidence с warmup/raw samples/median/p95 |
| `tunix-craftext docs sync/build/serve` | обновить dashboard, task graph, сайт и provenance |
| `tunix-craftext verify unit/golden/full` | локальные quality gates без implicit downloads |
| `tunix-craftext audit repo/architecture/docs` | аудит dirty state, архитектурных контрактов и документации |

Подробный дизайн, TDD-план и migration strategy описаны в [CLI слое](docs/cli.md).

Пока full CLI слой мигрируется, ручное управление CrafText доступно отдельным script entrypoint.
Он печатает instruction/constraint, vitals/inventory, tactical ASCII map вокруг игрока и legal
actions перед каждым ручным действием:

```bash
PYTHONPATH=src .venv/bin/python scripts/manual_craftext_agent.py \
  --config configs/manual/caged_wood_achievements_energy.yaml \
  --horizon 16

PYTHONPATH=src .venv/bin/python scripts/visualize_trajectory.py \
  --trajectory artifacts/trajectories/manual-craftext-latest.json

PYTHONPATH=src .venv/bin/python scripts/export_trajectory_gif.py \
  --trajectory artifacts/trajectories/manual-craftext-latest.json \
  --output artifacts/visualizations/manual-craftext-latest.gif \
  --fps 4 \
  --scale 4
```

## Внешние проекты: что мы перенимаем и где ставим границу

Мы не копируем чужие training loops в core. Каждый проект имеет закреплённую роль и отдельный
compatibility record в [`compatibility/training-stack.yaml`](compatibility/training-stack.yaml).

| Источник | Практика / роль у нас | Граница применения |
| --- | --- | --- |
| [Google Tunix](https://github.com/google/tunix) | `RLCluster`, Agentic GRPO, role meshes, model/learner API | Golden distributed training path; не заменяет typed environment/replay contracts. |
| [jax-lm](https://github.com/chuyishang/jax-lm) | discipline static shapes, mesh divisibility и preflight до загрузки весов | Вдохновляет `preflight.py`; **не** inference engine и не dependency. |
| [NVIDIA JAX-Toolbox](https://github.com/NVIDIA/JAX-Toolbox) | NVIDIA JAX containers, GPU CI matrix, Nsight/`nsys-jax` profiling, JAX↔vLLM rollout offloading pattern | Practice source для accelerator images/profiling/async rollout design; не runtime dependency CPU MVP и не замена Tunix/RLCluster. |
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

## Observability и Comet ML

Первичный слой логирования — локальный, versioned и воспроизводимый:
`metrics.jsonl`, `validation_trajectories.jsonl`, `artifacts.jsonl`, replay/trajectory files,
profiles и checkpoints в `artifacts/runs/<run-id>/`. Внешние трекеры подключаются как mirror.

Для Comet ML используйте optional adapter:

```python
from tunix_craftext.comet_adapter import CometMlSink
from tunix_craftext.observability import JsonlRunLogger

local = JsonlRunLogger(run_dir)
comet = CometMlSink.create_experiment(project_name="tunix-craftext")
```

Правило проекта: сначала пишем локальный evidence, затем отправляем те же records/artifacts
в Comet. Так training run остаётся проверяемым даже без сети или API key.

Прочитайте [план выполнения](docs/plan.md), [архитектуру](docs/architecture.md),
[интеграцию с Tunix](docs/tunix.md), [код/API](docs/code-reference.md) и [примеры](docs/examples.md) перед расширением тренера.
Notebook 07 показывает batched Qwen/Tunix rollout и replay export, 09/11/12 доводят тот же
pipeline до replay→token batch→real actor/critic PPO evaluation. Notebook 12 теперь
Gemma-first: `GemmaTunixBackend` генерирует rollout, actor пересчитывает token logprobs/entropy,
`TunixValueCritic` отдельно считает values, и `evaluate_separate_llm_actor_critic_ppo()` проверяет
returns/advantages/loss. Notebook 14 отдельно меряет generation pipeline по batch/horizon/repeats.
