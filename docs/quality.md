# TDD, интеграция и производительность

## Пирамида проверок

| Линия | Что доказывает | Запуск |
| --- | --- | --- |
| Unit | формы, masks, GAE/loss/checkpoint schema | `pytest tests/unit` |
| Integration | реальная среда, prompts, Tunix bridge, resume | `pytest -m integration` |
| Performance | median/p95 throughput, compile time, HBM | `pytest -m performance --benchmark-only` |
| E2E | config → train → checkpoint → eval → report | nightly / release |

## TDD порядок для каждого модуля

1. Написать пример с малыми числами и ожидаемым результатом.
2. Добавить property test: shapes, seed determinism, mask invariants.
3. Реализовать reference pure function.
4. Добавить JIT/vectorized path и parity test с reference.
5. Добавить benchmark только после корректности.

## Performance protocol

Каждая запись benchmark обязана содержать commit, dirty flag, config hash, seed, device,
JAX version, mesh, batch/horizon, warmup count и median/p95. Метрики:

- `env_steps_per_second`, `tokens_per_second`, `updates_per_second`;
- wall-clock compile time и steady-state step time;
- device memory peak, host-device bytes, recompilation count;
- sampler latency и доля invalid actions.

Порог регрессии задаётся на сценарий, пока ориентир — предупреждение при >5% median
регрессии на идентичном hardware. Performance не проходит на shared/unknown machine.

`make perf` сохраняет pytest-benchmark artifact в `artifacts/benchmarks/rollout-latest.json`;
dashboard автоматически извлекает из него mean, median и OPS. Перед сравнением переименуйте
artifact по сценарию и commit либо перенесите его в CI artifact storage.

После real CrafText scan parity benchmark matrix обязана покрывать как минимум `batch_size` 1, 2,
8 и `horizon` 8, 32, 128. Для каждой точки записывать отдельно compile latency и steady-state
env-steps/s; сравнивать только одинаковые preset, seed, device и JAX version.

`scripts/benchmark_environments.py` записывает schema
`tunix-craftext.environment-benchmark/v2`: raw blocking samples, mean/median/p95/min/max,
config SHA-256, full Git revision/dirty state, JAX version и hardware. Throughput считается по
median; варианты сравниваются с `craftext-full` только при одинаковых batch/horizon. Для
материальной записи используйте минимум 10 (предпочтительно 20) repeats:

```bash
make perf-env  # defaults: 20 repeats; full × tiny × Caged, B=1/2/8/32, T=8/32/128/512
```

Smoke with three repeats полезен только для проверки pipeline и **не** считается baseline.
На CPU runner честно записывает `memory_peak_bytes` и `host_device_bytes` как `null`: эти
метрики потребуют backend-specific profiler на целевом accelerator.

## MVP Python compatibility gate

После MVP GitHub Actions workflow `MVP Python compatibility` запускается на тегах `mvp-v*` (или
вручную) в матрице Python 3.11, 3.12 и 3.13. Он устанавливает `dev,docs` extras и запускает unit
и integration lanes. Matrix намеренно не запускает perf: shared GitHub runners непригодны для
сравнения производительности.

Локальная разработка и CI устанавливают зависимости через `uv sync --locked`; таким образом
проверяется именно закоммиченный `uv.lock`, а не новое разрешение версий во время workflow.
