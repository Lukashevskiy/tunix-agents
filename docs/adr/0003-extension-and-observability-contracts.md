# ADR 0003: extension и observability contracts

**Status:** accepted

## Decision

Алгоритм — typed набор pure JAX функций (`advantages`, `loss`, immutable metrics),
зарегистрированный по имени. Модель — adapter/profile с tokenizer, mesh axes, loader и
weight mapping. Core transport contracts и orchestration loop не знают конкретные модельные
семейства или objective.

Новый algorithm может расширить trajectory schema только через явную versioned migration;
запрет на изменение loop не означает запрет на необходимые данные вроде token masks или
reference log-probs.

Metrics остаются JAX PyTree внутри compiled work и извлекаются только на epoch boundary.
Profiling — явная CLI операция с warmup и ограниченным Perfetto trace; библиотека не запускает
постоянный profiler server. Shape assertions живут в unit/host boundary, а не в hot JIT path.

## Consequences

PPO становится первой registry implementation. GRPO/GSPO не добавляются по эвристической
формуле: требуется первичный источник, shape contract `[B, G, T]` и numerical fixtures.
