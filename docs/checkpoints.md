# Checkpoints и возобновление PPO

`tunix_craftext.checkpoints` сохраняет обучаемую часть `flax.training.TrainState`
через Orbax: `step`, `params` и `opt_state`. Функции модели и Optax transform не
сериализуются — при restore они намеренно берутся из локального `state_template`.
Это сохраняет checkpoint переносимым между процессами и не превращает Python-callable
в неявный бинарный контракт.

Рядом лежит `tunix_craftext_metadata.json` со строго версионируемой схемой
`tunix-craftext.checkpoint/v1`, идентификатором run, digest конфигурации и типом policy.
Неизвестная схема приводит к явной ошибке до восстановления состояния.

```python
from pathlib import Path

import jax

from tunix_craftext.checkpoints import CheckpointMetadata, restore_checkpoint, save_checkpoint
from tunix_craftext.learner import create_state

state = create_state(jax.random.PRNGKey(0), observation_dim=3, actions=2)
metadata = CheckpointMetadata(
    run_id="smoke-001",
    config_digest="sha256:validated-config",
    policy_kind="flax-actor-critic",
)
save_checkpoint(Path("artifacts/checkpoints/smoke-001"), state, metadata)

# Шаблон обеспечивает текущие apply_fn и optimizer transform.
resumed_state, resumed_metadata = restore_checkpoint(
    Path("artifacts/checkpoints/smoke-001"), state
)
```

Unit test проверяет важное свойство: после обновления save и restore дают те же параметры
и метрики на **следующем** PPO update. Это доказывает, что восстановлены не только веса
модели, но и состояние оптимизатора.

Известная текущая граница: этот API охватывает локальный Flax actor-critic learner.
Будущий Tunix adapter предоставит собственный policy state template при повторном
использовании той же schema/provenance логики.
