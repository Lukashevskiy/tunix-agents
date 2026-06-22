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

`configs/mvp/qwen_craftext.yaml` is the corresponding local-Qwen host-side episode profile.
It uses the real `MegaPrompts base` template, an explicit `tunix` policy and observable `NOOP`
fallback. It is run with `scripts/run_text_episode.py`; weights remain an explicit local input.

Поддерживаемая schema version сейчас только `1`; изменение полей — отдельная migration/ADR.
`build_craftext_runtime()` уже связывает validated `MvpRunConfig` с vendored world preset и
`CrafTextAdapter`; real reset и fixed-key mini-trajectory проверены integration test. Следующий
шаг — batched 2×8 golden trajectory и JAX scan parity.

```bash
pyenv exec python -m uv sync --extra envs --extra prompts
```
