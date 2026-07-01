# Профилирование модулей

Профилирование делится на два уровня:

1. **Локальный evidence layer** — всегда доступный `PhaseProfiler`, который пишет wall-time
   по фазам и JSON-артефакты.
2. **Accelerator lane** — NVIDIA Nsight Systems / `nsys-jax`, NVTX ranges и контейнеры,
   вдохновлённые [`NVIDIA/JAX-Toolbox`](https://github.com/NVIDIA/JAX-Toolbox).

## Локальный PhaseProfiler

`tunix_craftext.artifacts.profiling.PhaseProfiler` нужен для измерения отдельных модулей pipeline:

- `prompt.render`
- `llm.generate`
- `actor.score`
- `env.step`
- `replay.export`
- `ppo.update`

```python
from pathlib import Path

from tunix_craftext.artifacts.profiling import PhaseProfiler, block_until_ready, save_profile

profiler = PhaseProfiler(enable_nvtx=False)

with profiler.section("prompt"):
    rendered = renderer.render(context)

with profiler.section("llm"):
    responses = backend.complete_batch(requests)

with profiler.section("ppo"):
    state, metrics = full_token_ppo_update(state, batch, gamma=0.99)
    block_until_ready(metrics)

save_profile(
    Path("artifacts/profiles/full-cycle.json"),
    profiler.events(),
    metadata={"pipeline": "qwen-craftext-full-cycle"},
)
```

`block_until_ready()` важен для JAX: без синхронизации Python timer может измерить только
dispatch, а не реальное выполнение на устройстве.

## Практики из JAX-Toolbox

Из JAX-Toolbox мы переносим не код runtime, а инженерные практики:

- контейнеры публикуются как `ghcr.io/nvidia/jax:*` и фиксируют CUDA/cuDNN/NCCL/JAX stack;
- GPU CI разделяет backend-independent, single-GPU и multi-GPU checks;
- Nsight Systems — первый инструмент для timeline/launch latency;
- `nsys-jax` добавляет JAX/XLA metadata, выгружает `.parquet`/`.csv.xz` и source references;
- targeted profiling лучше full-run profiling: пропустить compile/warmup и снимать только
  устойчивые итерации;
- `nvtx.annotate()` полезен на Python boundary, а внутри JIT — `jax.named_scope`/`jax.named_call`;
- JAX↔vLLM offloading pattern пригодится для async rollout: control plane handshake отдельно,
  data plane отдельно, trainer и inference engine имеют разные meshes.

## Будущий accelerator command

До Docker/accelerator runner локальные команды остаются простыми:

```bash
python scripts/benchmark_text_pipeline.py --config configs/env/text/qwen_craftext.yaml
```

На NVIDIA runner целевой wrapper будет выглядеть так:

```bash
nsys-jax -o artifacts/profiles/qwen-craftext/%q{SLURM_PROCID} -- \
  python scripts/benchmark_text_pipeline.py --config configs/env/text/qwen_craftext.yaml
```

Важное правило: `XLA_FLAGS` не должен перетираться внутри запускаемой команды; иначе
`nsys-jax` не сможет собрать XLA metadata.
