# ADR 0005: синхронный JAX-first pipeline перед async distributed execution

## Статус

Принято.

## Контекст

Нам нужен воспроизводимый путь `CrafText → prompt → Qwen → action → trajectory → PPO`,
который сначала доказывает корректность data contracts, а затем масштабируется. Статья
[Anakin](https://arxiv.org/abs/2104.06272) задаёт полезный паттерн: state, transport и update
должны быть JAX-совместимыми, иметь статические формы и жить в compiled execution, а не в
Python-оркестраторе.

Пользователь выбрал Qwix для LoRA/QLoRA, Flashbax для replay staging, CommonLoopUtils для
loop observability и mpi4jax для будущего multi-host transport. Эти компоненты отличаются
зрелостью и operational risk: Flashbax — pure JAX library, Qwix совместим с текущим
Flax/JAX, а mpi4jax требует установленный MPI toolchain и реального multi-host runner.

## Решение

1. **Минимум — synchronous vanilla pipeline.** Один update-window собирает fixed-shape
   trajectories, превращает их в `TextTrajectoryBatch`, добавляет в bounded Flashbax
   item-buffer и выполняет PPO update. `add/sample` должны быть JIT-safe. Буфер не делает
   PPO off-policy: data не переживают policy update без отдельной correction strategy.
2. **Qwix — единственная новая LoRA/QLoRA реализация.** Она подключается через extra
   `lora`; перед train use обязательны architecture-specific output-parity, gradient и
   checkpoint metadata fixtures. Старый безопасный LoRA merge остаётся interoperability
   utility, но не объявляется QLoRA training path.
3. **CLU — optional loop boundary.** Он будет добавлен при введении structured per-update
   metrics/checkpoint orchestration, не в JIT inner loop и не как скрытый runtime singleton.
4. **Максимум — async distributed phase.** Rollout workers, bounded queue и learner должны
   иметь versioned message schema, policy-version/staleness metric, back-pressure и
   preemption/resume test. mpi4jax допускается только после отдельной MPI/JAX fixture;
   Tunix остаётся владельцем model sharding и placement.

Полная проверяемая матрица источников, ревизий, лицензий и статусов хранится в
`compatibility/training-stack.yaml`.

## Последствия

- Синхронный путь даёт baseline correctness и performance до добавления сложностей сети,
  stale policy и fault tolerance.
- Flashbax replay имеет фиксированные token/prompt widths; variable-length texts должны
  bucket-иться до insertion, иначе появятся нежелательные recompilation.
- Нельзя заявлять async scaling, пока нет multi-host benchmark с topology, commit, mesh,
  policy lag и throughput/latency breakdown.
- Локальный macOS/one-device smoke остаётся корректной функциональной проверкой, но не
  свидетельством распределённой производительности.
