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

The corresponding versioned profiles are `configs/models/gemma3_270m_instruction.yaml`
and `configs/models/qwen25_05b_instruction.yaml`. Their committed download/license
flags deliberately remain `false`: they describe a portable repository, not the
private local state of one machine.
