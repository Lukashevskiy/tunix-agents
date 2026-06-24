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
`CrafTextAdapter`; реальный reset, fixed-key mini-trajectory, batched rollout/replay export и
JAX scan parity покрываются unit/integration lanes. Следующий шаг — trainable actor/critic
config section для RLCluster workload.

```bash
pyenv exec python -m uv sync --extra envs --extra prompts
```
