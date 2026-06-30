#!/usr/bin/env python3
"""Run the explicit Tunix Agentic GRPO CrafText golden pipeline."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np


def task_batches(
    *, goal: str, seed: int, batch_size: int, count: int, horizon: int
) -> Iterator[dict[str, object]]:
    """Yield deterministic serializable task batches consumed by ``GRPOLearner``."""
    if not goal.strip() or batch_size <= 0 or count <= 0 or horizon <= 0:
        raise ValueError("goal, batch_size, count and horizon must be positive")
    for batch_index in range(count):
        start = seed + batch_index * batch_size
        yield {
            "goal": [goal] * batch_size,
            "seed": np.arange(start, start + batch_size, dtype=np.int32),
            "horizon": np.full(batch_size, horizon, dtype=np.int32),
        }


def craftext_task_batches(
    *,
    config_path: Path,
    seed: int,
    batch_size: int,
    count: int,
    horizon: int,
    goal_prefix: str,
    mode: str = "cycle",
) -> Iterator[dict[str, object]]:
    """Yield task batches whose goals and instruction indices come from CrafText."""
    from tunix_craftext.env.config import load_mvp_config
    from tunix_craftext.env.runtime import build_craftext_runtime
    from tunix_craftext.env.tasks import CrafTextTaskSampler

    config = load_mvp_config(config_path)
    runtime = build_craftext_runtime(config)
    sampler = CrafTextTaskSampler.from_runtime(
        runtime,
        horizon=horizon,
        mode=mode,  # type: ignore[arg-type]
        fixed_instruction_index=config.environment.instruction_index,
        goal_prefix=goal_prefix,
    )
    yield from sampler.batches(seed=seed, batch_size=batch_size, count=count)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one reproducible, local-weight Agentic GRPO run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        help="Canonical Agentic GRPO profile; preferred over individual workload flags.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mvp/qwen_craftext.yaml"))
    parser.add_argument(
        "--topology", type=Path, default=Path("configs/topology/qwen_agentic_grpo_local.yaml")
    )
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=Path("configs/generation/qwen_vllm_sync.yaml"),
        help="Declarative rollout generation config for sync/async backend selection.",
    )
    parser.add_argument(
        "--snapshot", type=Path, default=Path("artifacts/models/qwen25-05b-instruct")
    )
    parser.add_argument("--goal", default="Stay alive and inspect the world.")
    parser.add_argument(
        "--task-source",
        choices=("profile-goal", "craftext-instructions"),
        default="craftext-instructions",
        help="Use profile goal as every task or sample goals from CrafText scenario instructions.",
    )
    parser.add_argument(
        "--task-sampling",
        choices=("cycle", "fixed", "random"),
        default="cycle",
        help="Sampling mode when --task-source craftext-instructions is active.",
    )
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--kv-cache-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Optional Tunix checkpoint root; profile runs default to evidence.checkpoints.",
    )
    parser.add_argument("--skip-jit", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate profile/topology/preflight/evidence without loading model assets.",
    )
    parser.add_argument(
        "--scripted-smoke",
        action="store_true",
        help="Run local CrafText tool-call GRPO grouping with scripted actions and no model.",
    )
    parser.add_argument(
        "--scripted-output",
        type=Path,
        default=Path("artifacts/runs/agentic-grpo-scripted-smoke.json"),
    )
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="Attempt the unsupported local CPU Qwen rollout for upstream debugging only.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Build assets, cluster and learner, then perform the requested GRPO updates."""
    args = parse_args(arguments)

    from tunix_craftext.training.grpo_profile import (
        build_grpo_evidence_manifest,
        load_agentic_grpo_profile,
    )

    if args.profile is not None:
        profile = load_agentic_grpo_profile(args.profile)
        args.config = profile.environment_config
        args.topology = profile.topology_config
        args.generation_config = profile.generation_config
        args.snapshot = profile.model.snapshot
        args.goal = profile.run.goal
        args.max_steps = profile.workload.max_steps
        args.batch_size = profile.workload.mini_batch_size
        args.num_generations = profile.workload.num_generations
        args.max_new_tokens = profile.workload.max_new_tokens
        args.max_prompt_length = profile.workload.max_prompt_length
        args.kv_cache_size = profile.workload.kv_cache_size
        args.learning_rate = profile.workload.learning_rate
        args.checkpoint_root = profile.evidence.checkpoints
        profile.evidence.provenance.parent.mkdir(parents=True, exist_ok=True)
        profile.evidence.provenance.write_text(
            json.dumps(
                build_grpo_evidence_manifest(profile, profile_path=args.profile),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.max_steps <= 0 or args.batch_size <= 0 or args.num_generations < 2:
        raise ValueError("max_steps/batch_size must be positive and num_generations at least two")

    import jax

    from tunix_craftext.env.config import load_mvp_config
    from tunix_craftext.inference import load_generation_pipeline_config
    from tunix_craftext.tunix import (
        AgenticGrpoWorkloadSpec,
        RLClusterWorkloadError,
        load_tunix_topology,
        pinned_qwen_tensor_shape,
        validate_agentic_grpo_preflight,
    )

    config = load_mvp_config(args.config)
    generation = load_generation_pipeline_config(args.generation_config)
    topology = load_tunix_topology(args.topology)
    spec = AgenticGrpoWorkloadSpec(
        args.max_steps,
        args.max_steps,
        args.batch_size,
        args.batch_size,
        1,
        args.max_prompt_length,
        args.max_new_tokens,
        args.kv_cache_size,
        args.learning_rate,
        checkpoint_root_directory=args.checkpoint_root,
        num_generations=args.num_generations,
        max_concurrency=args.num_generations,
    )
    real_rollout_preflight_error: str | None = None
    try:
        validate_agentic_grpo_preflight(
            topology,
            spec,
            pinned_qwen_tensor_shape(),
            rollout_backend=_rollout_backend_for_generation(generation),
        )
    except RLClusterWorkloadError as error:
        real_rollout_preflight_error = str(error)
        if not args.dry_run and not args.scripted_smoke:
            raise

    with contextlib.redirect_stdout(io.StringIO()):
        preview_batch = next(
            craftext_task_batches(
                config_path=args.config,
                seed=config.run.seed,
                batch_size=args.batch_size,
                count=1,
                horizon=config.environment.horizon,
                goal_prefix=args.goal,
                mode=args.task_sampling,
            )
            if args.task_source == "craftext-instructions"
            else task_batches(
                goal=args.goal,
                seed=config.run.seed,
                batch_size=args.batch_size,
                count=1,
                horizon=config.environment.horizon,
            )
        )
    if args.dry_run:
        instruction_indices = preview_batch.get("instruction_index")
        print(
            json.dumps(
                {
                    "schema": "tunix-craftext.agentic-grpo-dry-run/v1",
                    "config": str(args.config),
                    "topology": str(args.topology),
                    "generation_config": str(args.generation_config),
                    "snapshot": str(args.snapshot),
                    "snapshot_exists": args.snapshot.is_dir(),
                    "backend": jax.default_backend(),
                    "real_rollout_preflight": {
                        "ok": real_rollout_preflight_error is None,
                        "error": real_rollout_preflight_error,
                    },
                    "task_source": args.task_source,
                    "task_sampling": args.task_sampling,
                    "generation": {
                        "engine_name": generation.profile.name,
                        "backend": generation.profile.backend,
                        "mode": generation.profile.mode,
                        "max_in_flight": generation.async_collection.max_in_flight,
                        "tunix_engine": generation.tunix.engine,
                        "vllm_server_mode": generation.tunix.vllm_server_mode,
                        "vllm_async_scheduling": generation.tunix.vllm_async_scheduling,
                    },
                    "batch_preview": {
                        "goal": preview_batch["goal"],
                        "seed": np.asarray(preview_batch["seed"]).tolist(),
                        "horizon": np.asarray(preview_batch["horizon"]).tolist(),
                        "instruction_index": np.asarray(instruction_indices).tolist()
                        if instruction_indices is not None
                        else None,
                    },
                    "workload": {
                        "max_steps": spec.max_steps,
                        "mini_batch_size": spec.mini_batch_size,
                        "num_generations": spec.num_generations,
                        "max_new_tokens": spec.max_new_tokens,
                        "checkpoint_root_directory": str(spec.checkpoint_root_directory)
                        if spec.checkpoint_root_directory
                        else None,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.scripted_smoke:
        from tunix_craftext.training.agentic_grpo_smoke import (
            collect_scripted_grpo_group_sync,
            save_scripted_grpo_smoke,
        )

        action_sequences = tuple(
            tuple(
                "NOOP" if generation % 2 == 0 else "LEFT"
                for _ in range(config.environment.horizon)
            )
            for generation in range(args.num_generations)
        )
        results = collect_scripted_grpo_group_sync(
            config_path=args.config,
            goal=args.goal,
            seed=config.run.seed,
            group_id=0,
            action_sequences=action_sequences,
            horizon=config.environment.horizon,
        )
        save_scripted_grpo_smoke(args.scripted_output, results)
        print(f"wrote scripted GRPO smoke evidence: {args.scripted_output}")
        return

    if not args.snapshot.is_dir():
        raise FileNotFoundError(f"Expected explicit local Qwen snapshot: {args.snapshot}")
    if jax.default_backend() == "cpu" and not args.allow_cpu_smoke:
        raise RuntimeError(
            "Agentic Qwen GRPO requires an accelerator mesh. The pinned Tunix Qwen vanilla "
            "sampler cannot resolve its sharded gather on a local CPU singleton mesh; use an "
            "accelerator runner or pass --allow-cpu-smoke only to reproduce that upstream failure."
        )

    from tunix.rl.agentic.agentic_grpo_learner import (  # type: ignore[import-untyped]
        GRPOConfig,
        GRPOLearner,
    )
    from tunix.rl.agentic.agents.tool_agent import ToolAgent  # type: ignore[import-untyped]
    from tunix.rl.agentic.parser.chat_template_parser.parser import (  # type: ignore[import-untyped]
        QwenChatTemplateParser,
    )

    from tunix_craftext.env.agentic_craftext import (
        CrafTextAgenticEnvironment,
        CrafTextStepTool,
        agentic_environment_kwargs,
    )
    from tunix_craftext.models.tunix_adapter import load_qwen_hf_tokenizer
    from tunix_craftext.tunix import (
        build_agentic_grpo_cluster,
        load_agentic_grpo_qwen_assets,
    )
    logging.info("Loading Agentic GRPO Qwen assets from %s", args.snapshot)
    assets = load_agentic_grpo_qwen_assets(args.snapshot, topology)
    cluster = build_agentic_grpo_cluster(topology, spec, assets, generation.tunix)
    try:
        tokenizer = load_qwen_hf_tokenizer(args.snapshot)
        learner = GRPOLearner(
            rl_cluster=cluster,
            algo_config=GRPOConfig(
                num_generations=args.num_generations,
                max_response_length=args.max_new_tokens,
                max_concurrency=args.num_generations,
                system_prompt="Use craftext_step for every environment action.",
            ),
            chat_parser=QwenChatTemplateParser(tokenizer, enable_thinking=False),
            agent_class=ToolAgent,
            agent_kwargs={
                "system_prompt": "Use craftext_step for every environment action.",
                "tool_parser_name": "qwen",
                "tool_map": {"craftext_step": CrafTextStepTool},
            },
            env_class=CrafTextAgenticEnvironment,
            env_kwargs=agentic_environment_kwargs(args.config),
        )
        train_batches = (
            craftext_task_batches(
                config_path=args.config,
                seed=config.run.seed,
                batch_size=args.batch_size,
                count=args.max_steps,
                horizon=config.environment.horizon,
                goal_prefix=args.goal,
                mode=args.task_sampling,
            )
            if args.task_source == "craftext-instructions"
            else task_batches(
                goal=args.goal,
                seed=config.run.seed,
                batch_size=args.batch_size,
                count=args.max_steps,
                horizon=config.environment.horizon,
            )
        )
        learner.train(
            train_batches,
            skip_jit=args.skip_jit,
        )
    finally:
        cast(Any, cluster).close()


def _rollout_backend_for_generation(generation: Any) -> str:
    """Map strict generation config to the static preflight backend lane."""
    if generation.profile.backend == "vllm-offload" or generation.tunix.engine == "vllm":
        return "vllm-offload"
    if generation.tunix.engine == "sglang_jax":
        return "single-device-jax"
    if generation.profile.tensor_parallel_size == 1:
        return "single-device-jax"
    return "vanilla-jax-sharded"


if __name__ == "__main__":
    main()
