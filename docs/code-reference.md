# Документация по коду и публичным абстракциям

Эта страница связывает site-документацию с основными Python-модулями. Автоматически
сгенерированная MkDocs-страница по docstrings находится в разделе
[`Автодока API`](_generated/api-reference.md), Sphinx API reference по-прежнему собирается
командой `make api-docs`, а здесь описаны роли абстракций, границы слоёв и минимальные примеры
использования в текущем итоговом pipeline.

## Карта слоёв

| Слой | Основные модули | Что гарантирует |
| --- | --- | --- |
| Core contracts | `tunix_craftext.core` | JAX/jaxtyping aliases, rollout contracts and resource mesh helpers shared by every layer. |
| Environment boundary | `tunix_craftext.adapters`, `tunix_craftext.env` | Vendor CrafText/CagedCrafText превращаются в typed `reset/step`, `action_mask`, `terminated/truncated`, reproducible config runtime, prompt context и strict text policy. |
| Model boundary | `tunix_craftext.models` | `RenderedPrompt` идёт в LLM backend, actor генерирует ordered completions и пересчитывает token scores/values; Tunix/Qwen/Gemma loaders живут отдельно от env/rollout. |
| Rollout transport | `tunix_craftext.rollouts` | `reference` остаётся fixed-shape JAX baseline; `batched` делает host prompt/LLM/decode + `jax.vmap(step)`; `hybrid` фиксирует PPO-ready actor logprobs, critic values, token masks и step masks. |
| Training builders | `tunix_craftext.training` | Universal MDP step contract, MDP-time GAE over `[T, B]`, advantage/return broadcasting на generated tokens, Agentic PPO bridge, GRPO profile/evidence and Flashbax staging. |
| Replay/training artifacts | `tunix_craftext.artifacts` | Replay v3 хранит prompt/completion/action/reward/token evidence, `masked_action`, fallback, text trajectory batches, checkpoints, profiling, provenance and run observability. |
| Research objectives/learner | `tunix_craftext.research.algorithms`, `tunix_craftext.research.algorithm_registry`, `tunix_craftext.research.learner`, `tunix_craftext.artifacts.checkpoints` | PPO/returns/loss functions чистые и JAX-friendly, но это research/smoke слой; production GRPO/PPO идёт через Tunix Agentic `RLCluster`. |
| Model/cluster interoperability | `tunix_craftext.interop`, `tunix_craftext.tunix`, `tunix_craftext.models.tunix_adapter` | LoRA/state-dict conversion, declared Tunix role→mesh mapping, preflight и hardware-gated RLCluster config отделены от core rollout. |

Top-level runtime modules were removed after the semantic package split. New code should import
from `core`, `env`, `rollouts`, `models`, `training`, `artifacts`, `tunix`, and `research`
directly. The root `tunix_craftext` package remains a compact public facade for common symbols,
but the old root-level rollout/config module files no longer exist.

## Минимальный CrafText runtime

```python
from pathlib import Path
import jax

from tunix_craftext.env.config import load_mvp_config
from tunix_craftext.env.runtime import build_craftext_runtime

root = Path.cwd()
config = load_mvp_config(root / "configs/env/smoke/tiny_craftext.yaml")
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

from tunix_craftext.rollouts.batched import (
    collect_batched_text_rollout,
    replays_from_batched_rollout,
)
from tunix_craftext.env.prompts import MegaPromptRenderer
from tunix_craftext.models.tunix_adapter import QwenTunixBackend

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
    config_path="configs/env/text/qwen_craftext.yaml",
    commit="notebook-or-script",
    backend="tunix-single-device:Qwen",
)
```

`collect_batched_text_rollout()` вызывает MegaPrompts на host, делает ordered `complete_batch()`
через backend, декодирует action labels, сверяет decoded action с текущим `action_mask` и только
после этого вызывает `jax.vmap(CrafTextAdapter.step)`. Если model action masked, replay сохранит
`masked_action=1` и `fallback_used=True` при включённом fallback.

Для colocated single-GPU smoke/benchmark можно явно закрепить JAX-среду на том же accelerator,
что и локальную Tunix-модель, но для vLLM sidecar чаще полезнее оставить GPU под inference и
держать env lane на CPU:

```python
from tunix_craftext.rollouts.batched import EnvironmentDevicePolicy, cpu_environment_device_policy

rollout = collect_batched_text_rollout(
    runtime.adapter,
    renderer,
    backend,
    actions=runtime.actions,
    batch_size=8,
    horizon=32,
    seed=config.run.seed,
    goal="Stay alive and choose one currently valid action.",
    max_new_tokens=8,
    invalid_action="fallback",
    fallback_action_id=fallback_action_id,
    device_policy=cpu_environment_device_policy(),
)
```

`EnvironmentDevicePolicy` делает `device_put` для keys/state/action masks/action ids и использует
`jit(vmap(reset/step))`. Внутри `collect_batched_text_rollout()` state/action-mask carry уже
считается размещённым и не перекладывается на device повторно на каждом decision. Для prompt
render и strict decode делается один явный host snapshot state/action-mask на decision; это
сознательный компромисс, потому что текстовая часть всё равно Python/host-side. Если CUDA backend
недоступен, pipeline падает до rollout, а не молча переключается на CPU.

## Hybrid rollout → PPO-ready actor/critic evidence

```python
from tunix_craftext.rollouts.hybrid import (
    HybridPpoStep,
    compute_masked_step_token_ppo_loss,
    hybrid_trajectory_from_steps,
)

step = HybridPpoStep(
    action_ids=action_ids,                      # [B]
    prompt_tokens=prompt_ids,                   # [B, P]
    prompt_token_mask=prompt_mask,              # [B, P]
    generation_tokens=completion_ids,           # [B, L]
    generation_token_mask=completion_mask,      # [B, L]
    actor_log_probs=rollout_token_logprobs,     # [B, L]
    values=critic_values,                       # [B]
    step_mask=alive_before_step,                # [B]
    action_mask=legal_action_mask,              # [B, A]
)
trajectory = hybrid_trajectory_from_steps([step])
```

`HybridPpoStep` — новый explicit boundary для рекомендаций внешнего PPO-аудита: динамическая
история промптов и generation остаются на host/Tunix стороне, а CrafText transition выполняется
через batched JAX step. `compute_masked_step_token_ppo_loss()` доказывает базовую семантику:
padding generated tokens и post-terminal rows не вносят вклад в actor PPO loss. Старые
`collect_rollout_scan*` не используются как production LLM-RL collector.
`shaped_step_rewards_from_text_trajectory()` добавляет явный не-положительный штраф за invalid
format / unknown action / masked action / fallback evidence: поведение получает отрицательный
reward signal, но fallback-token rows всё равно не попадают в actor loss.

## Model profile → LLM actor backbone

```python
from pathlib import Path

from tunix_craftext.models.profile import load_model_profile

profile = load_model_profile(Path("configs/models/gemma3_270m_instruction.yaml"))
print(profile.model_id)
```

`ModelProfile` фиксирует architecture/model id/source/licence/resource intent без загрузки
весов. `TunixCausalLmActor` — production-shaped boundary: `generate_batch()` нужен для rollout,
`score_actor_tokens()` возвращает actor token logprobs/entropy, а `critic().score_values()`
отдельно возвращает critic values для PPO/GRPO update.

```python
import jax

from tunix_craftext.models.tunix_actor import (
    build_gemma_tunix_actor,
    init_linear_value_head,
)

value_head = init_linear_value_head(jax.random.PRNGKey(0), hidden_dim=640)
actor = build_gemma_tunix_actor(
    profile_path=Path("configs/models/gemma3_270m_instruction.yaml"),
    snapshot=Path("artifacts/models/gemma3-270m-it"),
    cache_size=1024,
    value_head=value_head,
)
critic = actor.critic()
```

Gemma builder намеренно требует уже существующий локальный snapshot: проект не скачивает веса как
побочный эффект импорта, теста или построения документации.

## Replay → token batch → real actor/critic PPO evaluation

```python
from tunix_craftext.artifacts.text_trajectory import text_trajectory_from_replay
from tunix_craftext.research.llm_ppo import evaluate_separate_llm_actor_critic_ppo

batch = text_trajectory_from_replay(replays[0])
actor_scores = actor.score_actor_tokens(
    prompt_token_ids=batch.prompt_token_ids,
    prompt_token_mask=batch.prompt_token_mask,
    token_ids=batch.token_ids,
    token_mask=batch.token_mask,
)
critic_values = critic.score_values(
    prompt_token_ids=batch.prompt_token_ids,
    prompt_token_mask=batch.prompt_token_mask,
    token_ids=batch.token_ids,
    token_mask=batch.token_mask,
)
evaluation = evaluate_separate_llm_actor_critic_ppo(
    batch, actor_scores, critic_values, learning_mask=batch.policy_mask
)
```

`evaluate_separate_llm_actor_critic_ppo()` использует `batch.policy_mask`: padding исключён,
fallback decisions не попадают в policy learning, а actor/critic роли объединяются только на
objective boundary. Внутри остаётся тот же чистый JAX primitive `masked_token_ppo_loss`, поэтому
новые PPO/DPO/GRPO objectives можно тестировать на hand-computed tensors до подключения
тяжёлого model runtime. Компактные функции `token_ppo_update()`/`full_token_ppo_update()` остаются
механическим smoke для Flax/Optax update; production learner должен заменить их на
Qwen/Gemma/RLCluster actor/value path, не меняя replay или rollout contracts.

## Algorithm registry для PPO/DPO/GRPO

```python
from tunix_craftext.research.algorithm_registry import get_algorithm

ppo = get_algorithm("ppo")
print(ppo.name)
```

Новые objectives добавляются как отдельные registry entries с собственным typed batch contract и
hand-computed tests. Не fork-айте `collect_batched_text_rollout()` ради DPO/GRPO: transport
собирает evidence, а objective решает, как интерпретировать batch.

## PPO assets → Tunix RLCluster

```python
from pathlib import Path

from tunix_craftext.tunix import (
    RLClusterWorkloadSpec,
    build_ppo_cluster,
    load_tunix_topology,
    load_ppo_gemma_assets,
)

topology = load_tunix_topology(Path("configs/tunix/topology/qwen_local_smoke.yaml"))
spec = RLClusterWorkloadSpec(
    max_steps=10,
    eval_every_n_steps=5,
    mini_batch_size=4,
    train_micro_batch_size=2,
    rollout_micro_batch_size=2,
    max_prompt_length=128,
    max_new_tokens=8,
    kv_cache_size=256,
)
assets = load_ppo_gemma_assets(Path("artifacts/models/gemma3-270m-it"), topology)
cluster = build_ppo_cluster(topology, spec, assets)
```

Этот слой не загружает веса при импорте. Asset-функции являются hardware-gated boundary:
actor/reference загружаются на declared role meshes, critic создаётся как Tunix-compatible value
model, а `build_ppo_cluster()` вызывает публичный `RLCluster(actor, critic, reference, tokenizer)`.
`AgenticPPOLearner` принимает rich `traj["mdp_steps"]`, строит `UniversalMDPStep`, считает
`PpoExperienceBuilder` и отдаёт Tunix actor/critic update-ready batch. Следующий production
шаг — collector-side запись того же `mdp_steps` contract из `TrajectoryCollectEngine`, чтобы
old logprobs, critic values, rewards, masks и policy version приходили без replay guessing.

## Где смотреть примеры

- `examples/notebooks/07_qwen_craftext_full_trajectory.ipynb` — batched Tunix/MegaPrompts/CrafText replay export.
- `examples/notebooks/09_batched_qwen_craftext_rollout.ipynb` — B×T rollout и terminal reset semantics.
- `examples/notebooks/10_replay_to_token_ppo.ipynb` — replay evidence → `HybridPpoStep` →
  masked token/step PPO boundary.
- `examples/notebooks/11_end_to_end_batched_qwen_ppo.ipynb` — batched Qwen/CrafText rollout
  в per-env hybrid PPO evidence trajectories.
- `examples/notebooks/12_full_cycle_craftext_training.ipynb` — real Gemma/Tunix rollout,
  separate actor/critic scoring, `HybridPpoStep`, PPO evaluation, replay evidence и profiling.
- `examples/notebooks/14_generation_benchmark.ipynb` — generation pipeline benchmark по
  batch/horizon/repeats.
- `examples/notebooks/15_agentic_grpo_full_trainer.ipynb` — ручная сборка Agentic GRPO:
  profile/runtime/preflight, evidence/checkpoint paths, Qwen assets, RLCluster, ToolAgent и
  `GRPOLearner`.
