# Observability и логирование обучения

Обучение нельзя контролировать только по stdout и последнему checkpoint. Для GRPO/PPO
первичным источником правды становится append-only evidence layer:

- `metrics.jsonl` — train/val/eval/benchmark scalar events;
- `validation_trajectories.jsonl` — ссылки на полные validation trajectories;
- `trajectory/` — полный replay/trajectory artifact с prompt, raw completion, action,
  reward, masks, token ids/logprobs и fallback provenance;
- `checkpoints/` — actor/critic/reference trainer state, optimizer state и policy version;
- `weights/` — экспортированные actor/critic/reference weights или LoRA/Qwix adapters;
- `profiles/` — phase timings и будущие Nsight/`nsys-jax` traces.

## Python API

```python
from pathlib import Path

from tunix_craftext.artifacts.observability import (
    JsonlRunLogger,
    MetricRecord,
    RunArtifact,
    ValidationTrajectoryRecord,
    checkpoint_artifact,
    validation_trajectory_artifact,
    validation_visualization_artifact,
    weights_artifact,
)

logger = JsonlRunLogger(Path("artifacts/runs/qwen-grpo-smoke"))

logger.write_metric(
    MetricRecord(
        run_id="qwen-grpo-smoke",
        step=1,
        split="train",
        phase="update",
        policy_version=1,
        checkpoint_path="checkpoints/actor/1",
        metrics={
            "loss": 0.42,
            "kl": 0.01,
            "return_mean": 1.0,
            "invalid_action_rate": 0.0,
        },
    )
)

logger.write_validation_trajectory(
    ValidationTrajectoryRecord(
        run_id="qwen-grpo-smoke",
        step=1,
        task_id="safe-avoid-enemy",
        trajectory_path="trajectory/val/safe-avoid-enemy-step-1.json",
        return_sum=0.8,
        episode_length=8,
        success=True,
        policy_version=1,
        metrics={"fallback_count": 0},
    )
)

logger.write_artifact(
    validation_visualization_artifact(
        run_id="qwen-grpo-smoke",
        path="trajectory/val/safe-avoid-enemy-step-1.png",
        step=1,
        task_id="safe-avoid-enemy",
        policy_version=1,
    )
)

logger.write_artifact(
    checkpoint_artifact(
        run_id="qwen-grpo-smoke",
        path="checkpoints/actor/1",
        step=1,
        role="actor",
        policy_version=1,
    )
)

logger.write_artifact(
    weights_artifact(
        run_id="qwen-grpo-smoke",
        path="weights/actor-lora-step-1.safetensors",
        step=1,
        role="actor",
        policy_version=1,
        quantization="bf16",
    )
)
```

## Что логировать во время train

Минимальный набор для каждого update:

- `loss`, `kl`, `entropy`;
- `return_mean`, `return_p50`, `return_p95`;
- `success_rate`;
- `invalid_action_rate`, `fallback_rate`, `masked_action_rate`;
- `episode_length_mean`;
- `tool_call_count`;
- `tokens_per_second`, `env_steps_per_second`, `update_seconds`;
- `policy_version`, `checkpoint_path`.
- artifact manifest для checkpoint/weights/optimizer state.

Для PPO дополнительно:

- `value_loss`, `value_mean`, `explained_variance`;
- `advantage_mean`, `advantage_std`;
- `actor_weight_delta`, `critic_weight_delta` на one-update gates.

Для PPO-Lag/CPO позже:

- `cost_return_mean`;
- `cost_value_loss`;
- `lagrange_multiplier`;
- `constraint_violation_rate`.

## Validation trajectories

Validation не должен сохранять только агрегаты. Для каждого fixed task нужно сохранять полный
trajectory artifact, а в `validation_trajectories.jsonl` писать компактную ссылку и summary:

- `task_id`;
- `step`;
- `policy_version`;
- `trajectory_path`;
- `return_sum`;
- `episode_length`;
- `success`;
- дополнительные scalar metrics.

Это позволяет открыть конкретный провал, увидеть prompt, completion, decoded action,
fallback и состояние среды, а не гадать по усреднённой кривой.

## Server readiness перед большим запуском

Перед запуском на целевой машине с GPU сначала проверьте не обучение, а evidence-контур:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_server_readiness.py \
  --profile configs/grpo/qwen_agentic_local.yaml \
  --mode evidence \
  --require-accelerator \
  --require-snapshot \
  --output artifacts/runs/qwen-agentic-craftext-local-smoke/server-readiness.json
```

Команда должна записать:

- `provenance.json` — profile hash, git revision, model/vendor/dependency provenance;
- `metrics.jsonl` — scalar event `eval/server_readiness`;
- `validation_trajectories.jsonl` — ссылку на validation artifact;
- `artifacts.jsonl` — config/profile/checkpoint/validation artifact manifest;
- `validation/server-readiness-*.json` — полный validation replay/smoke artifact;
- `checkpoints/readiness-probe/` — проверку доступности checkpoint directory.

Если нужен реальный CrafText/CagedCrafText tool-loop без LLM allocation:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_server_readiness.py \
  --profile configs/grpo/qwen_agentic_local.yaml \
  --mode scripted \
  --scripted-horizon 2 \
  --require-accelerator \
  --require-snapshot
```

`--require-accelerator` превращает CPU backend в failure, а не warning. `--require-snapshot`
делает отсутствие локальных весов blocking ошибкой. Локально эти флаги можно не ставить, чтобы
проверять только формат и запись evidence.

## Будущие sinks

TensorBoard, W&B, Prometheus и Comet ML являются вторичными sinks. Они не должны
становиться единственным местом хранения: JSONL остаётся воспроизводимым локальным
контрактом, который можно читать в тестах, CLI, сайте и release card.

## Comet ML adapter

Comet подключается отдельным optional adapter, а не импортируется core-модулем:

```python
from tunix_craftext.artifacts.comet_adapter import CometMlSink

comet = CometMlSink.create_experiment(
    project_name="tunix-craftext",
    workspace="my-workspace",
)

record = MetricRecord(
    run_id="qwen-grpo-smoke",
    step=1,
    split="train",
    phase="update",
    metrics={"loss": 0.42, "kl": 0.01},
)

logger.log_metric(record)  # local JSONL first
comet.log_metric(record)   # mirror to Comet second
```

Правило: сначала сохранить локальный artifact (`metrics.jsonl`, replay, validation
visualization, checkpoint, weights), затем отправить ссылку/файл в Comet через
`RunArtifact`. Если Comet недоступен, локальные evidence остаются полными.

## Адаптер под любой командный logger

Если в команде уже есть локальный logger, его не нужно встраивать в core. Достаточно
сопоставить методы logger-а с базовым `ArtifactSink` контрактом:

```python
from tunix_craftext.artifacts.observability import LoggerMethodMapping, MappedLoggerSink

team_sink = MappedLoggerSink(
    team_logger,
    mapping=LoggerMethodMapping(
        log_metrics="scalars_write",
        log_artifact="file_write",
        log_text="text_write",
        log_image="image_write",
    ),
)

team_sink.log_metric(record)
team_sink.log_artifact(artifact)
```

Если logger странный и методами его не описать, можно передать прямые callables:

```python
team_sink = MappedLoggerSink(
    object(),
    log_metrics=lambda metrics, **kw: my_metrics_client.send(metrics, **kw),
    log_artifact=lambda path, **kw: my_artifact_store.upload(path, **kw),
    log_text=lambda text, **kw: my_event_log.write(text),
)
```

Так новый adapter под локальную инфраструктуру команды остаётся тонким glue-слоем поверх
`MetricRecord`, `ValidationTrajectoryRecord` и `RunArtifact`.
