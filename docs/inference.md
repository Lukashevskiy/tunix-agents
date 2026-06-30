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
├── RequestsLlmBackend      # adapter back to existing BatchLlmBackend
└── VllmInferenceEngine     # optional vLLM adapter
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
| `vanilla-jax-sharded` | blocked для Qwen `fsdp,tp` | ждёт upstream Tunix sharded embedding-gather fix |

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

Для установки на целевом Linux/GPU runner:

```bash
uv sync --extra tunix --extra envs --extra prompts --extra vllm
```

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
