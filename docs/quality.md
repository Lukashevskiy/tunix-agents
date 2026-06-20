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
