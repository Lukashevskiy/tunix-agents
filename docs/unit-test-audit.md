# Unit Test Audit

Дата аудита: 2026-06-26.

## Текущий gate

Unit pipeline теперь состоит из четырёх CPU-friendly слоёв:

1. `ruff check src tests scripts`
2. `mypy src/tunix_craftext`
3. `pytest tests/unit`
4. `pytest tests/unit --cov=src/tunix_craftext --cov-fail-under=80`

Coverage считается branch-aware по `src/tunix_craftext`; CI пишет `coverage.xml` и
`term-missing` отчёт. Локальный эквивалент:

```bash
make coverage
```

## Последний baseline

Команда:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit \
  --cov=src/tunix_craftext \
  --cov-report=term-missing:skip-covered \
  --cov-report=xml \
  --cov-fail-under=80
```

Результат:

- `227 passed`
- total branch-aware coverage: `80.53%`
- hard gate: `80%`
- `agentic_ppo.py`: `97%`

## Что усилено в этом аудите

- `AgenticPPOConfig` теперь проверяет negative cases для PPO optimizer contract:
  `num_iterations`, `epsilon`, `clip_range_value`, `epsilon_c`, `kl_method`.
- `AgenticPPOLearner._process_results()` покрыт для:
  - legacy fallback path;
  - dict-style Tunix rollout config by mode;
  - rollout actor logprob recomputation;
  - reference KL branch при `beta != 0`;
  - rich `traj["mdp_steps"]` path через `PpoExperienceBuilder`;
  - MDP path с reference logprobs.
- `universal_mdp_steps_from_trajectory()` покрыт для:
  - variable-length 1D rows;
  - batched rows;
  - action mask preservation;
  - policy-token mask semantics;
  - missing fields, invalid token ranks, batch mismatch, logprob mismatch и scalar-vector mismatch.
- Старые research/smoke фикстуры синхронизированы с обязательным
  `TextTrajectoryBatch.invalid_action`.

## Осознанные зоны ниже production coverage

Эти зоны не блокируют текущий gate, но должны подниматься перед real accelerator lane:

| Модуль | Причина | Следующий тестовый шаг |
| --- | --- | --- |
| `tunix_adapter.py` | много hardware/model boundary и optional Tunix paths | fake tokenizer/model fixtures для generation/scoring branches без загрузки весов |
| `runtime.py`, `episode.py` | real env integration частично живёт в integration tests | lightweight fake env adapter для unit-level reset/step/episode policy |
| `agentic_grpo_smoke.py` | smoke runner и CLI-ish orchestration | перенести business logic в use-case слой и покрыть runner без subprocess |
| `interop/safetensors.py` | file-format bridge | tmp safetensors round-trip fixture |

Правило: новый hot-path модуль должен добавлять unit coverage вместе с контрактом, а не
компенсироваться случайным ростом общего процента.
