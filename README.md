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
uv python install 3.12.13
uv python pin 3.12.13
uv sync --extra dev --extra docs
uv run pytest
```

`uv` управляет интерпретатором, `.venv` и lockfile. `.python-version` остаётся project-local
pin для `uv python pin`, а `.venv` — одноразовое окружение, созданное `uv`. Смотрите
[практику среды разработки](docs/development.md) для обновлений, opt-in extras и CI правил.

Для реальных env и Tunix установите opt-in extras после фиксации совместимой accelerator-specific
сборки JAX:

```bash
uv sync --extra envs --extra tunix --extra replay --extra dev
```

## CLI: единая точка управления проектом

Проект переходит от набора отдельных `scripts/*.py` к полновесному CLI `tunix-craftext`
с коротким alias `tcx`. CLI спроектирован как thin orchestration layer: он валидирует профили,
показывает статус, запускает use-case слой и пишет evidence, но не хранит внутри себя
логику CrafText, MegaPrompts, Tunix/RLCluster или benchmark runner.

Пока CLI находится в стадии реализации. Целевой первый gate:

```bash
tunix-craftext profile validate configs/training/grpo/qwen_agentic_local.yaml
tunix-craftext profile evidence configs/training/grpo/qwen_agentic_local.yaml \
  --output artifacts/runs/qwen-agentic-craftext-local-smoke/provenance.json
tunix-craftext verify golden
```

До появления console script используйте уже работающие эквиваленты:

```bash
uv run python scripts/run_agentic_grpo.py \
  --profile configs/training/grpo/qwen_agentic_local.yaml --allow-cpu-smoke
make verify-golden
```

Для быстрой проверки GRPO без загрузки модели есть два безопасных режима:

```bash
uv run python scripts/run_agentic_grpo.py \
  --profile configs/training/grpo/qwen_agentic_local.yaml --dry-run

uv run python scripts/run_agentic_grpo.py \
  --profile configs/training/grpo/qwen_agentic_local.yaml --scripted-smoke
```

`--dry-run` валидирует profile/topology/preflight/evidence. `--scripted-smoke` запускает реальный
CrafText agentic tool-call loop с несколькими GRPO generations и group-normalized advantages, но
без LLM/RLCluster weights. Это рабочий local gate перед реальным `GRPOLearner`; PPO/PPO-Lag/CPO
потом добавляют value critic и cost critic поверх уже проверенного rollout/tool transport.
Golden profile ссылается на `configs/inference/vllm/qwen25_05b_grpo_sync.yaml`: это
GRPO/RLCluster-safe vLLM profile с меньшим KV/HBM budget, потому что на той же GPU
уже живут actor/reference JAX weights. Direct rollout notebooks (`17`, `18`) используют
обычные `qwen25_05b_sync.yaml` / `qwen25_05b_async.yaml` profiles.
Если direct vLLM path уже работает, а Tunix `RLCluster` падает на vLLM worker RPC/weight-sync,
используйте `examples/notebooks/22_external_vllm_sync_grpo_rollout.ipynb`: он собирает replay
траектории через внешний `VllmInferenceEngine`, группирует их в `ExternalGrpoBatch`, считает
GRPO advantages, строит token batch и считает critic-free GRPO loss/metrics без создания
`RLCluster`. Это стартовая база для
GRPO до actor update и следующий вход для LoRA/Qwix/PPO critic слоя.
По умолчанию GRPO task batches берут `goal` из CrafText scenario instructions: batch содержит
и текст задачи, и `instruction_index`, поэтому reset среды выбирает тот же scenario row, который
видит модель. Если нужно старое поведение с одной ручной целью из profile, используйте
`--task-source profile-goal`; для CrafText задач доступны `--task-sampling cycle|fixed|random`.

Перед запуском на большом GPU-сервере прогоните отдельный readiness gate. Он не обучает модель:
проверяет profile/topology/preflight, видимость JAX devices, наличие snapshot, запись provenance,
`metrics.jsonl`, `validation_trajectories.jsonl`, `artifacts.jsonl`, validation artifact и
checkpoint directory probe.

Сначала полезно снять “рентген” installed stack: какие версии ожидает текущий `pyproject.toml`,
что реально установлено на runner, какие imports уже падают и какую CUDA/JAX/Torch платформу видит
процесс.

```bash
uv run python scripts/inspect_accelerator_stack.py \
  --extra tunix --extra envs --extra prompts --extra vllm --extra vllm-gpu-kernels \
  --output artifacts/runs/accelerator-stack.json

make accelerator-stack
```

Если там уже видно broken import вроде `torchvision::nms`, чините binary stack до запуска
ноутбуков/GRPO. Если Torch видит GPU, а JAX пишет
`Unable to initialize backend 'cuda'`, это отдельный JAX CUDA plugin/PJRT mismatch: для unit
tests можно временно запускать `JAX_PLATFORMS=cpu make test`, а для реального rollout/training
нужно переустановить JAX CUDA wheel/plugin под driver/CUDA stack целевого runner.

```bash
uv run python scripts/check_server_readiness.py \
  --profile configs/training/grpo/qwen_agentic_local.yaml \
  --mode evidence \
  --require-accelerator \
  --require-snapshot \
  --output artifacts/runs/qwen-agentic-craftext-local-smoke/server-readiness.json

uv run python scripts/check_server_readiness.py \
  --profile configs/training/grpo/qwen_agentic_local.yaml \
  --mode scripted \
  --scripted-horizon 2 \
  --require-accelerator \
  --require-snapshot
```

`evidence` проверяет файловый и observability-контур без среды. `scripted` дополнительно
запускает короткий CrafText tool-call validation loop без LLM weights, чтобы убедиться, что
validation trajectories реально создаются до дорогого модельного запуска.

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
uv run python scripts/manual_craftext_agent.py \
  --config configs/env/manual/caged_wood_achievements_energy.yaml \
  --horizon 16

uv run python scripts/visualize_trajectory.py \
  --trajectory artifacts/trajectories/manual-craftext-latest.json

uv run python scripts/export_trajectory_gif.py \
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
| [NVIDIA JAX-Toolbox](https://github.com/NVIDIA/JAX-Toolbox) | NVIDIA JAX containers, GPU CI matrix, Nsight/`nsys-jax` profiling, JAX↔vLLM rollout offloading pattern | Practice source для accelerator images/profiling/async rollout design; переносим архитектурный pattern в `tunix_craftext.inference`, но не копируем runtime code. |
| [Qwix](https://github.com/google/qwix) | QLoRA/quantized JAX–Flax experiments | Только после output-parity и trainability fixture выбранной Tunix model. |
| [Flashbax](https://github.com/instadeepai/flashbax) | JIT-compatible bounded replay staging | Исследовательский sync path; не превращает on-policy batch в неявный off-policy replay. |
| [CommonLoopUtils](https://github.com/google/CommonLoopUtils) | structured metrics и checkpoint loop practice | Optional reporting/checkpoint layer после стабилизации JSONL evidence. |
| [mpi4jax](https://github.com/mpi4jax/mpi4jax) | explicit multi-host collectives | Только будущая async/multi-host фаза и только с target-hardware fixture. |
| [Anakin paper](https://arxiv.org/abs/2104.06272) | JAX-first actor–learner: static PyTrees, compiled numerical path, host I/O вне `jit` | Архитектурный принцип для sync-first, затем async scale-out. |

Vendored `CrafText`, `CagedCrafText` и `MegaPrompts` сохраняются неизменёнными в `vendor/`;
наша работа живёт в typed adapters и runtime boundaries, поэтому upstream snapshots можно
обновлять и проверять отдельно.

Для production rollout generation вводится отдельная inference boundary. Базовый проект и
unit tests не требуют vLLM, но целевой Linux/GPU runner может поставить optional extra:

```bash
uv sync --extra tunix --extra envs --extra prompts --extra vllm
```

`VllmInferenceEngine` принимает `EngineProfile` из strict generation YAML, сохраняет ordered
batch/cardinality contract и возвращает нормализованные `LlmResponse`. Sync и async варианты
живут в `configs/inference/vllm/qwen25_05b_sync.yaml` и
`configs/inference/vllm/qwen25_05b_async.yaml`; профиль GRPO подключает выбранный файл через
`generation_config`.
Sync и async collectors унифицированы через общий `GenerationRecord`: downstream код видит
`index`, исходный `GenerationBatch` и `GenerationResult` независимо от режима исполнения.
Для RL стандартом остаётся sync mega-batch: один environment step собирает все prompts в один
batch и делает один вызов generation backend. Native async vLLM вынесен в отдельный
`AsyncVllmInferenceEngine` на `AsyncLLMEngine`; sync `VllmInferenceEngine` больше не притворяется
async через worker thread.

GPU-kernel extras отделены от базового vLLM extra. `flashinfer-python` можно поставить через
extra:

```bash
uv sync --extra envs --extra prompts --extra vllm --extra vllm-gpu-kernels
```

`flash-attn` намеренно не входит в `pyproject.toml`: для него лучше скачать prebuilt wheel под
конкретные Python/CUDA/PyTorch на целевой машине и поставить вручную:

```bash
uv pip install /path/to/flash_attn-*.whl
uv run python -c "import flash_attn; print(flash_attn.__version__)"
```

Если `VllmInferenceEngine.from_profile(...)` падает на
`RuntimeError: operator torchvision::nms does not exist`, это несовместимая binary-пара
`torch`/`torchvision`, а не ошибка CrafText pipeline. Для text-only Qwen сначала проверьте:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
uv run python -c "import torchvision; print(torchvision.__version__)"
```

Если второй импорт падает, либо удалите `torchvision` из text-only vLLM окружения, либо
переустановите совместимые CUDA wheels `torch`/`torchvision` для конкретного runner. После
этого повторите notebook cell с `VllmInferenceEngine.from_profile(...)`.

Если notebook падает на
`Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}`,
это уже не ошибка prompt/env слоя: vLLM subprocess не смог поднять engine. Перед повторным запуском
сначала сохраните accelerator report:

```bash
make accelerator-stack
```

и проверьте в нём `jax`, `torch`, `torchvision`, `vllm`, `flashinfer`, путь к локальному model
snapshot и memory knobs из `configs/inference/vllm/*.yaml` (`dtype`, `max_model_len`,
`tensor_parallel_size`). Настоящая причина обычно напечатана vLLM строками выше в stderr.
Если root cause говорит `Free memory ... is less than desired GPU memory utilization`, значит
vLLM пытается зарезервировать слишком большую долю VRAM. Для прямого `VllmInferenceEngine`
это регулируется в `engine.metadata.gpu_memory_utilization`, а для Tunix rollout server path —
в `tunix.vllm_hbm_utilization`. В стандартных Qwen configs оба значения держим на `0.35`.
Перед запуском notebook можно быстро оценить текущий budget:

```bash
make vllm-memory
# или
uv run python scripts/estimate_vllm_memory.py --config configs/inference/vllm/qwen25_05b_sync.yaml --strict
```

Утилита не импортирует vLLM: она читает generation YAML, текущую `torch.cuda.mem_get_info`,
суммирует локальные weight files и оценивает KV-cache по `config.json`, если snapshot уже скачан.

Для Tunix `RLCluster` + vLLM GRPO отдельно проверьте compatibility layer weight-sync:

```bash
PYTHONPATH=src uv run python scripts/inspect_vllm_tunix_compat.py --strict
```

Если direct vLLM notebooks (`17`, `18`) работают, а GRPO падает на
`collective_rpc("delete_kv_cache")`, это не проблема обычного inference path. Это означает,
что установленный vLLM worker API не предоставляет cache RPC hooks, которые текущий Tunix
`VllmSampler.update_params()` вызывает при JAX→vLLM weight sync. Диагностический отчёт покажет,
есть ли нативные hooks или нужен vLLM worker extension.

Для one-GPU режима, где JAX/Tunix и vLLM живут на одной карте, выставляйте JAX memory knobs
до старта Python/Jupyter:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# или более жёсткий cap для JAX allocator:
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
```

После изменения этих переменных нужен restart kernel/process: JAX читает их при первом импорте.
`make accelerator-stack` показывает текущие значения в поле `environment` и предупреждает, если
GPU JAX backend активен, а preallocation/fraction не настроены.

Если перед этим появляется
`os.fork() is incompatible with multithreaded code, and JAX is multithreaded`, значит vLLM
стартует subprocess через fork уже после инициализации JAX. Для notebook path перезапустите
kernel и создавайте `VllmInferenceEngine` до тяжёлых JAX вычислений. Наши generation YAML
задают `engine.metadata.multiprocessing_method: spawn`, а адаптер выставляет
`VLLM_WORKER_MULTIPROC_METHOD=spawn` до импорта vLLM, если пользователь не задал env вручную.
Для production/offload предпочтительнее держать vLLM server отдельным процессом.

`VanillaInferenceEngine`, `VllmInferenceEngine` и reserved `SglangInferenceEngine`
подключаются через единый `build_inference_engine(...)` registry.

Для ручной проверки этого слоя есть три notebooks:

- `examples/notebooks/17_sync_vllm_craftext_rollout.ipynb`
- `examples/notebooks/18_async_vllm_craftext_rollout.ipynb`
- `examples/notebooks/22_external_vllm_sync_grpo_rollout.ipynb`

Эти notebooks читают `configs/inference/vllm/*.yaml`, а не создают backend profile руками в ячейках.

## Локальный запуск сайта

Создайте локальное окружение документации один раз, затем используйте команду репозитория вместо
глобальной установки MkDocs:

```bash
uv sync --extra dev --extra docs
make serve
```

`make serve` сперва обновляет Dashboard текущим Git commit, roadmap, inventory возможностей и
`artifacts/benchmarks/*.json`, затем запускает локальный MkDocs server. Используйте `make docs`
для сборки того же статического `site/` без сервера. Benchmark JSON артефакты из
`artifacts/benchmarks/` появляются автоматически при следующем билде. GitHub Pages workflow делает
tоже самое при каждом пуше в `main`, еженедельно или вручную.

Без Make запустите `uv run python scripts/generate_dashboard.py && uv run mkdocs serve`.
Не используйте просто глобальный `mkdocs serve`: он может выбрать системный интерпретатор без
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
from tunix_craftext.artifacts.comet_adapter import CometMlSink
from tunix_craftext.artifacts.observability import JsonlRunLogger

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
