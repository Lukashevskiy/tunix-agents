# Inference boundary и strict generation contract

## Зачем нужен строгий generation contract

Ошибка Qwen `embedding gather` показала, что нельзя считать trainer mesh и rollout generation
одним и тем же интерфейсом. Даже если `RLCluster` правильно создал роли, generation backend
имеет свои ограничения: layout embeddings, KV-cache, tensor parallel degree, batching,
latency/provenance и stop/logprob contract.

Поэтому проект вводит не “ещё один backend”, а строгий контракт способа генерации:

```text
tunix_craftext.inference
├── EngineProfile           # backend/model/tensor_parallel/max_len provenance
├── GenerationBatch         # ordered static batch LlmRequest + group/policy metadata
├── GenerationResult        # ordered LlmResponse + profile + policy version
├── GenerationRecord        # index + batch + result, same for sync and async
├── InferenceEngine         # sync generate(batch) protocol
├── AsyncInferenceEngine    # async generate_async(batch) over the same payload
├── TunixGenerationContract # compiler to Tunix rollout_engine/RolloutConfig
├── VanillaInferenceEngine  # wraps existing BatchLlmBackend instances
├── VllmInferenceEngine     # optional vLLM adapter
├── SglangInferenceEngine   # reserved SGLang adapter boundary
├── build_inference_engine  # profile-driven backend registry
├── RequestsLlmBackend      # adapter back to existing BatchLlmBackend
└── collectors              # sync/async collection returning GenerationRecord
```

Это повторяет архитектурную идею NVIDIA JAX-Toolbox/JAX↔vLLM offloading: trainer и rollout
engine разделены, а между ними есть явная data/control boundary. Мы не копируем runtime code
из внешнего репозитория; мы переносим форму разделения ответственности.

## Sync и async — один payload

Синхронный и асинхронный rollout отличаются только способом исполнения:

```python
from tunix_craftext.inference import GenerationBatch, as_async_engine

batch = GenerationBatch(
    requests=tuple(requests),
    group_id="task-0007",
    policy_version=12,
)

result = sync_engine.generate(batch)
async_result = await as_async_engine(sync_engine).generate_async(batch)
```

Для collectors используется тот же record type:

```python
from tunix_craftext.inference import collect_generation_results_sync

records = collect_generation_results_sync(sync_engine, (batch,))
async_records = await collect_generation_results(async_engine, (batch,), max_in_flight=4)
```

В обоих случаях сохраняются:

- порядок и cardinality запросов;
- единые decoding knobs внутри batch;
- `group_id` для GRPO/trajectory grouping;
- `policy_version` для actor/rollout weight sync и stale-policy аудита;
- normalized `LlmResponse` с raw text, token ids, prompt ids, logprobs и latency.

## Компиляция в Tunix

Tunix уже имеет адаптированные rollout configs: `ClusterConfig.rollout_engine` принимает
`"vanilla"`, `"vllm"` и `"sglang_jax"`, а `RolloutConfig` содержит `rollout_vllm_*`,
`tensor_parallel_size`, `data_parallel_size`, server-mode и async scheduling knobs.

Наш `TunixGenerationContract` не заменяет эти поля; он валидирует наш строгий смысловой
контракт и компилирует его в Tunix:

```python
from tunix_craftext.inference import TunixGenerationContract

generation = TunixGenerationContract(
    engine="vllm",
    max_prompt_length=1024,
    max_tokens_to_generate=32,
    kv_cache_size=2048,
    tensor_parallel_size=1,
    vllm_server_mode=True,
    vllm_async_scheduling=True,
    vllm_hbm_utilization=0.35,
    vllm_model_version="qwen2.5-0.5b",
)

cluster_config = build_agentic_grpo_cluster_config(topology, spec, generation)
```

## Текущие backend lanes

| Backend | Статус | Назначение |
| --- | --- | --- |
| `scripted` / existing `BatchLlmBackend` | готово | CPU tests, notebooks, deterministic evidence |
| `single-device-jax` | planned | one-GPU bring-up без symbolic `fsdp,tp` generation |
| `vllm-offload` | strict contract + adapter boundary готов | production rollout generation на Linux/GPU runner |
| `sglang` / `sglang-jax` | явная граница, runtime ещё не подключён | будущая альтернатива vLLM/Tunix rollout |
| `vanilla-jax-sharded` | blocked для Qwen `fsdp,tp` | ждёт upstream Tunix sharded embedding-gather fix |

`build_inference_engine(profile, vanilla_backends=...)` является единой factory. Vanilla-like
backends требуют заранее созданный `BatchLlmBackend`, потому что загрузка их весов
проектно-специфична; vLLM и будущие server/offload backends создаются из `EngineProfile`.

## Минимальный vLLM profile

```python
from tunix_craftext.inference import EngineProfile, VllmInferenceEngine

profile = EngineProfile(
    name="qwen25-05b-vllm-rollout",
    backend="vllm-offload",
    model="artifacts/models/qwen25-05b-instruct",
    tensor_parallel_size=1,
    max_model_len=2048,
    dtype="bfloat16",
)
engine = VllmInferenceEngine.from_profile(profile)
```

## Declarative generation config

Для реальных запусков `EngineProfile` и Tunix rollout knobs не создаются руками в ноутбуках.
Они живут в strict YAML:

```python
from pathlib import Path

from tunix_craftext.inference import load_generation_pipeline_config

generation = load_generation_pipeline_config(
    Path("configs/generation/qwen_vllm_sync.yaml")
)

engine_profile = generation.profile
tunix_kwargs = generation.tunix.to_tunix_rollout_kwargs()
max_in_flight = generation.async_collection.max_in_flight
```

Sync и async варианты имеют одинаковый payload contract, но разные execution knobs:

- `configs/generation/qwen_vllm_sync.yaml` — deterministic ordered collector, one batch at a time.
- `configs/generation/qwen_vllm_async.yaml` — bounded async collector and Tunix vLLM server mode.

GRPO profile ссылается на этот YAML через `generation_config`, а evidence manifest записывает
его вместе с model/topology/workload provenance. Это делает rollout generation воспроизводимой
частью эксперимента, а не скрытой настройкой в ноутбуке.

Для установки на целевом Linux/GPU runner:

```bash
uv sync --extra tunix --extra envs --extra prompts --extra vllm
```

После установки снимите platform report до открытия notebooks:

```bash
uv run python scripts/inspect_accelerator_stack.py \
  --extra tunix --extra envs --extra prompts --extra vllm --extra vllm-gpu-kernels \
  --output artifacts/runs/accelerator-stack.json
```

Отчёт показывает platform tags, JAX devices, Torch CUDA state, версии requirements из
`pyproject.toml`, а также import errors для `vllm`, `torchvision`, `flashinfer`, `flash_attn`
и других runtime modules. Поле `recommendations` сразу подсказывает типовые действия:
`JAX_PLATFORMS=cpu` для CPU unit lane при broken JAX CUDA plugin, переустановка JAX CUDA wheel
для real GPU lane, удаление/переустановка broken `torchvision` для text-only vLLM.

GPU-kernel extras намеренно отделены от базового vLLM extra, чтобы macOS/CPU dev path не
пытался собирать CUDA-specific wheels. Сейчас extra ставит только wheel-friendly
`flashinfer-python`:

```bash
uv sync --extra envs --extra prompts --extra vllm --extra vllm-gpu-kernels
```

`flash-attn` ставим вручную prebuilt wheel-ом под конкретный runner:

```bash
uv pip install /path/to/flash_attn-*.whl
uv run python -c "import flash_attn; print(flash_attn.__version__)"
```

Если импорт vLLM падает внутри `transformers → torchvision` с
`operator torchvision::nms does not exist`, значит installed `torchvision` не совпадает с
installed `torch`/CUDA wheel stack. Это нужно чинить до запуска rollout:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
uv run python -c "import torchvision; print(torchvision.__version__)"
```

Для text-only Qwen допустимый быстрый workaround — убрать broken `torchvision` из vLLM env,
чтобы Transformers не пытался импортировать image utilities. Для multimodal или если vLLM wheel
явно требует torchvision, переустановите matching CUDA wheels `torch`/`torchvision`.

Если `VllmInferenceEngine.from_profile(...)` доходит до
`Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}`,
значит import уже прошёл, но vLLM worker subprocess умер на старте. В notebook это часто выглядит
как “чёрный ящик”, потому что настоящая причина напечатана строками выше в stderr дочернего
процесса. Наша обёртка переводит это в `InferenceBackendError` с profile summary; дальше порядок
такой:

```bash
make accelerator-stack
```

Проверяем report по пунктам: `torch.cuda_available`, JAX CUDA или `JAX_PLATFORMS=cpu` для тестов,
broken `torchvision`, наличие локального model snapshot, `dtype`, `max_model_len`,
`tensor_parallel_size` и доступную VRAM.

Если root cause выглядит так:

```text
Free memory on device cuda:0 (...) on startup is less than desired GPU memory utilization
```

то vLLM стартует с большим reservation budget. У прямого `VllmInferenceEngine` этот budget
задаётся через `engine.metadata.gpu_memory_utilization`; у Tunix rollout server path — через
`tunix.vllm_hbm_utilization`. Эти поля намеренно дублируются в generation YAML, потому что sync
notebook path создаёт `vllm.LLM(...)` напрямую, а Tunix path компилирует свои rollout kwargs.
Для Qwen 0.5B smoke configs выставлено `0.35`, чтобы не конкурировать с JAX/Tunix в том же GPU
процессе.

Перед запуском можно сделать lightweight preflight без импорта vLLM:

```bash
make vllm-memory
# или strict-вариант для CI / server readiness
uv run python scripts/estimate_vllm_memory.py \
  --config configs/generation/qwen_vllm_sync.yaml \
  --safety-margin-gib 1.0 \
  --strict
```

JSON содержит:

- `requested_gib` — сколько vLLM попросит у GPU по `gpu_memory_utilization`;
- `free_gib` / `total_gib` из `torch.cuda.mem_get_info`;
- `fits_current_free_memory` — помещается ли reservation с safety margin;
- `snapshot_weights_gib` — сумма локальных `.safetensors`/`.bin`, если snapshot скачан;
- `estimated_kv_cache_gib` — грубая оценка KV-cache по `config.json`, `max_model_len`,
  `max_num_seqs` и dtype.

Это preflight estimate, не замена точному vLLM allocator/block planner, но он ловит главный
класс ошибок до запуска EngineCore: слишком высокий reservation budget при уже занятой VRAM.

### JAX memory preallocation

Да, для shared one-GPU режима обычно нужно отключить JAX preallocation или поставить явный cap.
Иначе JAX/Tunix может заранее занять VRAM, а vLLM увидит слишком мало `free_gib` и упадёт ещё до
загрузки модели.

Перед запуском Python/Jupyter:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

или, если хочется оставить JAX фиксированный кусок памяти:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
```

Важно: эти переменные должны быть выставлены **до первого `import jax`**. Если notebook уже
импортировал JAX, нужен restart kernel. `make accelerator-stack` теперь включает поле
`environment` с `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_MEM_FRACTION`,
`XLA_PYTHON_CLIENT_ALLOCATOR`, `JAX_PLATFORMS` и `VLLM_WORKER_MULTIPROC_METHOD`; если JAX GPU и
Torch CUDA активны, но JAX memory knobs не заданы, report добавит рекомендацию
`jax-memory-preallocation-enabled`.

### JAX + vLLM multiprocessing

Warning вида
`os.fork() is incompatible with multithreaded code, and JAX is multithreaded`
означает, что vLLM пытается стартовать worker через `fork()` после того, как JAX уже поднял
многопоточный runtime. Это не harmless warning: возможен deadlock.

Правила для нашего pipeline:

1. В notebook сначала загрузить generation config и создать `VllmInferenceEngine`, затем делать
   тяжёлые JAX операции.
2. Если warning уже появился — restart kernel; смена env после `fork()` не исправляет процесс.
3. В `configs/generation/qwen_vllm_*.yaml` фиксируем
   `engine.metadata.multiprocessing_method: spawn`.
4. `VllmInferenceEngine` выставляет `VLLM_WORKER_MULTIPROC_METHOD=spawn` до импорта vLLM, если
   пользователь не задал `VLLM_WORKER_MULTIPROC_METHOD` явно.
5. Для production/offload path лучше запускать vLLM server отдельным процессом и обращаться к
   нему через backend contract: тогда JAX trainer process и vLLM worker lifecycle разделены.

Доступные extras: `vllm-flashinfer`, `vllm-gpu-kernels`, `sglang`.

## Следующий gate

1. Подключить `InferenceEngine` к batched CrafText rollout collector через
   `RequestsLlmBackend`.
2. Добавить actor weight export/sync metadata в `EngineProfile.metadata`.
3. Сделать hardware-gated test: один ordered batch prompt → vLLM/Tunix rollout engine →
   action decode → CrafText step → replay artifact.
4. После этого вводить RLCluster actor weight sync в rollout engine.

## Notebooks

- `17_sync_vllm_craftext_rollout.ipynb` — sync vLLM path через
  `collect_generation_results_sync()` для contract smoke, затем `RequestsLlmBackend` и
  существующий `collect_batched_text_rollout()`.
- `18_async_vllm_craftext_rollout.ipynb` — async vLLM path через
  `collect_generation_results()` с bounded `max_in_flight` и тем же
  `GenerationBatch/GenerationResult` payload.
