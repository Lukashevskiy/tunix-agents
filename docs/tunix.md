# Tunix integration boundary

Tunix is deliberately an optional dependency. The project pins the official
[`google/tunix`](https://github.com/google/tunix) revision in
`compatibility/tunix.yaml`; it must not use the
unrelated `tunix==0.0.0` PyPI distribution.

Install the bridge environment explicitly:

```bash
pyenv exec python -m uv sync --all-extras
```

The public boundary is intentionally narrow: `PPOConfig`/`PPOLearner` plus the
explicit Qwen loader and sampler boundary in `tunix_adapter.py`. A local Qwen 2.5
0.5B snapshot can be sampled through `QwenTunixBackend`; this is an integration
smoke profile, not a replacement for the training architecture. Our
environment/prompt/replay contracts remain framework-neutral.

Before each Qwen call, the backend applies the tokenizer's declared chat template
and computes the static cache capacity required by Tunix's power-of-two prompt
padding. It returns raw completion, latency, generated token ids and per-token logprobs. A too-small
cache fails with a project-level `ValueError` before entering Tunix. The optional
integration smoke additionally proves the complete `Qwen → decode/fallback →
CrafText → replay v2` path with local weights.

`text_trajectory_from_replay()` turns that replay into a padded JAX
`TextTrajectoryBatch`: generated token ids, behaviour logprobs, token/policy masks and
the environment reward on each completion's final token. Steps that used fallback remain
auditable but are excluded from `policy_mask`, so an eventual PPO/SFT update cannot silently
learn from a non-model action.

`masked_token_returns()` and `masked_token_ppo_loss()` are the corresponding pure JAX loss
boundary: they operate only on valid `[B, T]` tokens and ignore padding/fallback mask positions.
They need future actor recomputation and critic values; they do not claim to be a complete
Qwen value bridge by themselves.

The base environment does not import Tunix. This keeps `make test`, CrafText
collection and Flax/Optax smoke learning independent of the heavyweight model
and tokenizer stack while preserving an exact, reproducible bridge source.

No model weights are downloaded as a side effect of installation or ordinary test
execution. The optional real-Qwen smoke runs only when the explicit local snapshot
exists under `artifacts/models/qwen25-05b-instruct`.

Tunix owns distributed execution, but the project must declare topology: the future
workload path uses `RLCluster` with a versioned `role_to_mesh` mapping for actor,
rollout, critic and reference. It may use Tunix resharding/offload; it must not add a
second GPU scheduler in this repository. The architecture decision and the distinction
from the local sampler are recorded in [ADR 0004](adr/0004-tunix-cluster-topology.md).

Topology lives in `configs/topology/`: `qwen_local_smoke.yaml` colocates every role on
device 0, while `qwen_four_device_colocated.yaml` documents a four-device mesh. Both are
strictly validated before an accelerator workload is constructed; a profile requesting
unavailable devices fails early. `tunix_role_to_meshes()` is the sole adapter that maps
these named roles to Tunix's official `Role` enum.

`tests/integration/test_tunix_topology_hardware.py` is intentionally hardware-gated: it skips
below four visible devices and verifies the four-device profile on an accelerator runner. This
tests placement declaration, not scale-up performance; the latter belongs to the performance lane.

The corresponding versioned profiles are `configs/models/gemma3_270m_instruction.yaml`
and `configs/models/qwen25_05b_instruction.yaml`. Their committed download/license
flags deliberately remain `false`: they describe a portable repository, not the
private local state of one machine.
