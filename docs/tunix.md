# Интеграция с Tunix

Tunix намеренно является необязательной зависимостью. Проект фиксирует официальный
[`google/tunix`](https://github.com/google/tunix) revision в
`compatibility/tunix.yaml`; он не должен использовать
непохожий PyPI-дистрибутив `tunix==0.0.0`.

Установите bridge-окружение явно:

```bash
pyenv exec python -m uv sync --all-extras
```

Публичная граница намеренно узкая: `PPOConfig`/`PPOLearner` вместе с
явным Qwen loader и sampler boundary в `tunix_adapter.py`. Локальный снимок Qwen 2.5
0.5B может sample-иться через `QwenTunixBackend`; это smoke-профиль интеграции, а не
замена архитектуры обучения. Наши контракты environment/prompt/replay остаются
framework-neutral.

Перед каждым вызовом Qwen backend применяет заявленный tokenizer chat template
и вычисляет статическую ёмкость кеша, требуемую для power-of-two padding prompt в Tunix.
Он возвращает raw completion, latency, сгенерированные token ids и logprobs по токенам.
Слишком маленький кеш приводит к проектной `ValueError` до входа в Tunix.
Опциональный интеграционный smoke также подтверждает полный путь
`Qwen → decode/fallback → CrafText → replay v3` с локальными весами.

`text_trajectory_from_replay()` преобразует этот replay в упакованный JAX
`TextTrajectoryBatch`: необработанные prompt/generated token ids, behaviour logprobs,
маски token/policy и reward среды на каждом итоговом токене completion.
Шаги с fallback остаются аудируемыми, но исключаются из `policy_mask`, чтобы
будущее PPO/SFT обновление не могло silent learn из не-модельного действия.

`masked_token_returns()` и `masked_token_ppo_loss()` соответствуют чистому JAX loss
boundary: они работают только с корректными `[B, T]` токенами и игнорируют padding/
fallback mask позиции. Они требуют будущей actor recomputation и critic values;
сами по себе они не претендуют на полный Qwen value bridge.

Для будущего критика `QwenTunixBackend.hidden_states()` открывает финальные
`[B, T, D]` признаки закреплённой модели через публичный Qwen `skip_lm_head=True`.
Реальный интеграционный fixture подтверждает, что у профиля 0.5B `D=896`; прикрепление,
обучение и checkpointing value head остаются отдельным явным workload шагом.

Базовая среда не импортирует Tunix. Это сохраняет `make test`, сбор CrafText и smoke
обучение Flax/Optax независимыми от тяжёлого стека модели и tokenizer, сохраняя при этом
точный и воспроизводимый bridge source.

Ни одни веса модели не загружаются как побочный эффект установки или обычного теста.
Опциональный реальный Qwen smoke запускается только тогда, когда явный локальный снимок
существует в `artifacts/models/qwen25-05b-instruct`.

Tunix владеет распределённым исполнением, но проект должен объявлять топологию: будущий
workload path использует `RLCluster` с versioned `role_to_mesh` mapping для actor,
rollout, critic и reference. Он может применять sharding/offload Tunix; он не должен добавлять
в этом репозитории второй GPU scheduler. Решение архитектуры и отличие от локального sampler
задокументированы в [ADR 0004](adr/0004-tunix-cluster-topology.md).

Топология живёт в `configs/topology/`: `qwen_local_smoke.yaml` размещает все роли на
устройстве 0, а `qwen_four_device_colocated.yaml` документирует четырехустройственный mesh.
Оба профиля строго валидируются до построения accelerator workload; профиль, запрашивающий
недоступные устройства, завершается раньше. `tunix_role_to_meshes()` — единственный адаптер,
сопоставляющий эти именованные роли с официальным Tunix `Role` enum.

`tests/integration/test_tunix_topology_hardware.py` намеренно hardware-gated: он пропускает
тест на системах с менее чем четырьмя видимыми устройствами и проверяет четырехустройственный
профиль на accelerator runner. Это проверяет declaration placement, а не масштабную производительность;
последняя относится к performance lane.

Соответствующие versioned profiles — `configs/models/gemma3_270m_instruction.yaml`
и `configs/models/qwen25_05b_instruction.yaml`. Их зафиксированные флаги download/license
намеренно остаются `false`: они описывают портируемый репозиторий, а не приватное состояние
одной машины.
