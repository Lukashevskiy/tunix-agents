# Документация по коду и публичным абстракциям

Эта страница связывает site-документацию с основными Python-модулями. Автоматически
сгенерированная MkDocs-страница по docstrings находится в разделе
[`Автодока API`](_generated/api-reference.md), Sphinx API reference по-прежнему собирается
командой `make api-docs`, а здесь описаны роли абстракций, границы слоёв и минимальные примеры
использования в текущем итоговом pipeline.

## Карта слоёв

| Слой | Основные модули | Что гарантирует |
| --- | --- | --- |
| Environment boundary | `tunix_craftext.adapters`, `tunix_craftext.runtime`, `tunix_craftext.config` | Vendor CrafText/CagedCrafText превращаются в typed `reset/step`, `action_mask`, `terminated/truncated` и reproducible config runtime. |
| Prompt/model boundary | `tunix_craftext.prompts`, `tunix_craftext.text_policy`, `tunix_craftext.llm`, `tunix_craftext.tunix_adapter` | `EnvState` становится `RenderedPrompt`, Tunix/Qwen возвращает ordered completions/token provenance, strict decoder не принимает неизвестные labels. |
| Rollout transport | `tunix_craftext.rollout`, `tunix_craftext.batched_rollout`, `tunix_craftext.contracts` | Численные trajectories остаются `[T, B, ...]`; text rollout enforce-ит `action_mask` перед `CrafTextAdapter.step` и экспортирует per-env replay. |
| Replay/training batch | `tunix_craftext.replay`, `tunix_craftext.text_trajectory`, `tunix_craftext.flashbax_replay` | Replay v3 хранит prompt/completion/action/reward/token evidence, `masked_action`, fallback и преобразуется в fixed-shape token batches. |
| Objectives/learner | `tunix_craftext.algorithms`, `tunix_craftext.algorithm_registry`, `tunix_craftext.learner`, `tunix_craftext.checkpoints` | PPO/returns/loss functions чистые и JAX-friendly; будущие DPO/GRPO должны добавляться через typed registry/batch contracts. |
| Model/cluster interoperability | `tunix_craftext.interop`, `tunix_craftext.tunix_topology`, `tunix_craftext.rlcluster_workload` | LoRA/state-dict conversion, declared Tunix role→mesh mapping и hardware-gated RLCluster config отделены от core rollout. |

## Минимальный CrafText runtime

```python
from pathlib import Path
import jax

from tunix_craftext.config import load_mvp_config
from tunix_craftext.runtime import build_craftext_runtime

root = Path.cwd()
config = load_mvp_config(root / "configs/mvp/tiny_craftext.yaml")
runtime = build_craftext_runtime(config)
reset = runtime.adapter.reset(jax.random.PRNGKey(config.run.seed))

print(reset.action_mask.shape)
print(runtime.actions.labels)
```

`runtime.adapter` — единственная точка, где training code видит vendor environment. Дальше код
работает с normalized `EnvironmentReset`/`EnvironmentStep`, а не с динамическим vendor API.

## Prompt → Tunix/Qwen → batched CrafText rollout

```python
from pathlib import Path

from tunix_craftext.batched_rollout import (
    collect_batched_text_rollout,
    replays_from_batched_rollout,
)
from tunix_craftext.prompts import MegaPromptRenderer
from tunix_craftext.tunix_adapter import QwenTunixBackend

snapshot = Path("artifacts/models/qwen25-05b-instruct")
renderer = MegaPromptRenderer(config.prompt.template)
backend = QwenTunixBackend(snapshot, cache_size=2048, seed=config.run.seed)
fallback_action_id = runtime.actions.index_of("NOOP")

rollout = collect_batched_text_rollout(
    runtime.adapter,
    renderer,
    backend,
    actions=runtime.actions,
    batch_size=2,
    horizon=2,
    seed=config.run.seed,
    goal="Stay alive and choose one currently valid action.",
    max_new_tokens=8,
    invalid_action="fallback",
    fallback_action_id=fallback_action_id,
)
replays = replays_from_batched_rollout(
    rollout,
    config_path="configs/mvp/qwen_craftext.yaml",
    commit="notebook-or-script",
    backend="tunix-single-device:Qwen",
)
```

`collect_batched_text_rollout()` вызывает MegaPrompts на host, делает ordered `complete_batch()`
через backend, декодирует action labels, сверяет decoded action с текущим `action_mask` и только
после этого вызывает `jax.vmap(CrafTextAdapter.step)`. Если model action masked, replay сохранит
`masked_action=1` и `fallback_used=True` при включённом fallback.

## Replay → token batch → trainable token PPO update

```python
import jax

from tunix_craftext.algorithms import masked_token_returns
from tunix_craftext.learner import create_token_state, full_token_ppo_update, token_ppo_update
from tunix_craftext.text_trajectory import text_trajectory_from_replay

batch = text_trajectory_from_replay(replays[0])
returns = masked_token_returns(batch.rewards, batch.token_mask, gamma=0.99)
state = create_token_state(jax.random.PRNGKey(0), token_bucket_count=512)
state, metrics = full_token_ppo_update(state, batch, gamma=0.99)
```

`full_token_ppo_update()` использует `batch.token_mask`: все generated tokens входят в actor,
critic и entropy terms, padding исключён, fallback-marked rows не выкидываются. Для safety-first
запусков остаётся `token_ppo_update()`, который использует `batch.policy_mask` и исключает
fallback decisions. Обе функции внутри вызывают `masked_token_ppo_loss`, но передают туда
пересчитанные actor `new_logprobs`, critic values и entropy. `PromptConditionedTokenActorCritic`
отвечает за локальный smoke; production learner заменит этот компактный bridge на Qwen/RLCluster
actor/value path, не меняя replay или rollout contracts.

## Algorithm registry для PPO/DPO/GRPO

```python
from tunix_craftext.algorithm_registry import get_algorithm

ppo = get_algorithm("ppo")
print(ppo.name)
```

Новые objectives добавляются как отдельные registry entries с собственным typed batch contract и
hand-computed tests. Не fork-айте `collect_batched_text_rollout()` ради DPO/GRPO: transport
собирает evidence, а objective решает, как интерпретировать batch.

## Где смотреть примеры

- `examples/notebooks/07_qwen_craftext_full_trajectory.ipynb` — batched Tunix/MegaPrompts/CrafText replay export.
- `examples/notebooks/09_batched_qwen_craftext_rollout.ipynb` — B×T rollout и terminal reset semantics.
- `examples/notebooks/10_replay_to_token_ppo.ipynb` — replay evidence в token batch/loss tensors.
- `examples/notebooks/12_full_cycle_craftext_training.ipynb` — compact scripted full-cycle smoke без приватных весов.
