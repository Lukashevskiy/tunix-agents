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

# The template supplies current apply_fn and optimizer transform.
resumed_state, resumed_metadata = restore_checkpoint(
    Path("artifacts/checkpoints/smoke-001"), state
)
```

Unit test verifies the important property: after an update, save and restore produce
the same parameters and metrics on the **next** PPO update. This proves optimizer
moments as well as model weights were restored.

Known current boundary: this API covers the local Flax actor-critic learner. A future
Tunix adapter will provide its own policy state template while reusing the same
schema/provenance rule.
