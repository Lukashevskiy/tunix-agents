# Конфигурация MVP

Все запуски начинаются с versioned YAML в `configs/`, а не с аргументов командной строки.
`load_mvp_config()` строго валидирует canonical schema: ключи каждого раздела должны совпасть
точно, а неизвестные/missing keys и неподходящие значения завершаются `ConfigError`.

Первый контракт — `configs/mvp/tiny_craftext.yaml`:

| Раздел | Смысл |
| --- | --- |
| `run` | имя запуска и seed |
| `environment` | vendored CrafText, Craftax preset, scenario, instruction, batch/horizon |
| `prompt` | renderer и MegaPrompts template |
| `policy` | implementation и реакция на invalid action |
| `artifacts` | trajectory, rendered prompt, metrics для replay/provenance |

`configs/mvp/qwen_craftext.yaml` — соответствующий профиль запуска локального Qwen Host-side.
Он использует реальный шаблон `MegaPrompts base`, явную политику `tunix` и observable `NOOP`
fallback. Он запускается через `scripts/run_text_episode.py`; веса передаются как явный локальный input.

`configs/manual/caged_wood_achievements_energy.yaml` — ручной operator профиль для полного
Caged CrafText сценария. Он соединяет scenario
`budget/achievements/easy/wood_achievements` с world preset `caged_craftext_play`: сам scenario
задаёт wood-achievement instruction, а preset добавляет boxed world, `player_energy` и
`action_energy_drain` с ценой `1` energy за действие. Этот профиль является дефолтом для
`scripts/manual_craftext_agent.py`, чтобы руками собранные replay trajectories сразу проверяли
wood task под energy constraint.

Поддерживаемая версия схемы сейчас только `1`; изменение полей — отдельная migration/ADR.
`build_craftext_runtime()` связывает validated `MvpRunConfig` с vendored world preset и
instruction wrapper. Для `implementation: craftext` это `RawInstructionWrapper → CrafTextAdapter`
и `TextEnvState`; для `caged-craftext` — `CMDPInstructionWrapper → CagedCrafTextAdapter` с
text constraint, синхронным с выбранной instruction. `scenario_config` обязан существовать в
dataset соответствующего implementation: Caged scenario нельзя указать для plain CrafText.
`world_preset`, instruction и constraint затем становятся явным context MegaPrompts. Реальный
reset, fixed-key mini-trajectory, batched rollout/replay export и JAX scan parity покрываются
unit/integration lanes.

```bash
uv sync --extra envs --extra prompts
```

## Agentic GRPO profile

Production-facing training больше не должен стартовать из россыпи флагов. Первый canonical
profile — `configs/grpo/qwen_agentic_local.yaml`. Он связывает:

| Раздел | Смысл |
| --- | --- |
| `run` | имя, seed и user goal, передаваемый в Agentic task stream |
| `environment_config` | validated CrafText/Caged MVP config |
| `topology_config` | Tunix role mesh для `actor`, `rollout`, `reference` |
| `generation_config` | strict sync/async inference backend config для rollout generation |
| `model` | Qwen model id, explicit local snapshot, revision и licence record |
| `workload` | GRPO steps, micro-batches, generation count, sequence limits, KV cache и learning rate |
| `evidence` | root, trajectories JSONL, metrics JSONL, checkpoints и provenance JSON |
| `vendor_manifest` | vendored CrafText/Caged/MegaPrompts provenance input |

`load_agentic_grpo_profile()` строго валидирует schema до загрузки весов. При запуске

```bash
uv run python scripts/run_agentic_grpo.py \
  --profile configs/grpo/qwen_agentic_local.yaml
```

runner сначала пишет `evidence.provenance`: git revision, SHA256 profile, SHA256 vendor
manifest, generation backend contract, model snapshot/revision/licence, workload knobs и
package versions. Только после этого начинается accelerator/model allocation. Старые CLI-флаги
остаются только для debug/smoke, но golden path должен использовать profile.

Пути внутри profile остаются в исходном виде для provenance (`configs/...`,
`artifacts/...`), но runtime обязан резолвить их через `resolve_profile_path()` от корня
репозитория/profile. Это защищает notebooks, `server_readiness` и `run_agentic_grpo.py`
от ошибок текущей директории: ячейка может запускаться из `/workspace`, а
`generation_config: configs/generation/qwen_vllm_sync.yaml` всё равно должен читаться из
репозитория.

## Generation pipeline config

Inference/generation вынесен в отдельный versioned YAML, чтобы один и тот же training profile
можно было запускать через sync collector, async collector, vLLM, vanilla backend или будущий
SGLang lane без переписывания ноутбуков.

Canonical configs:

| Config | Назначение |
| --- | --- |
| `configs/generation/qwen_vllm_sync.yaml` | один ordered batch за раз; базовый воспроизводимый vLLM rollout smoke |
| `configs/generation/qwen_vllm_async.yaml` | bounded async collection с `max_in_flight` и Tunix vLLM server/async scheduling knobs |

Схема состоит из трёх блоков:

| Раздел | Смысл |
| --- | --- |
| `engine` | normalized `EngineProfile`: backend, model snapshot, TP, max len, dtype, sync/async mode |
| `tunix` | project-owned `TunixGenerationContract`, компилируемый в Tunix `RolloutConfig` |
| `async` | queue/concurrency knobs для async collector; для sync lane остаётся `max_in_flight: 1` |

Проверить выбранный backend без загрузки модели:

```bash
uv run python scripts/run_agentic_grpo.py \
  --profile configs/grpo/qwen_agentic_local.yaml \
  --dry-run
```

В dry-run payload появится `generation`: имя engine, backend family, sync/async mode и vLLM
server/scheduling flags.
