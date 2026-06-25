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
pyenv exec python -m uv sync --extra envs --extra prompts
```

## Agentic GRPO profile

Production-facing training больше не должен стартовать из россыпи флагов. Первый canonical
profile — `configs/grpo/qwen_agentic_local.yaml`. Он связывает:

| Раздел | Смысл |
| --- | --- |
| `run` | имя, seed и user goal, передаваемый в Agentic task stream |
| `environment_config` | validated CrafText/Caged MVP config |
| `topology_config` | Tunix role mesh для `actor`, `rollout`, `reference` |
| `model` | Qwen model id, explicit local snapshot, revision и licence record |
| `workload` | GRPO steps, micro-batches, generation count, sequence limits, KV cache и learning rate |
| `evidence` | root, trajectories JSONL, metrics JSONL, checkpoints и provenance JSON |
| `vendor_manifest` | vendored CrafText/Caged/MegaPrompts provenance input |

`load_agentic_grpo_profile()` строго валидирует schema до загрузки весов. При запуске

```bash
PYTHONPATH=src .venv/bin/python scripts/run_agentic_grpo.py \
  --profile configs/grpo/qwen_agentic_local.yaml
```

runner сначала пишет `evidence.provenance`: git revision, SHA256 profile, SHA256 vendor
manifest, model snapshot/revision/licence, workload knobs и package versions. Только после
этого начинается accelerator/model allocation. Старые CLI-флаги остаются только для
debug/smoke, но golden path должен использовать profile.
