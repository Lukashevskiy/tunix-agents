#!/usr/bin/env python3
"""Run the explicit Tunix Agentic GRPO CrafText golden pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

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
        "--snapshot", type=Path, default=Path("artifacts/models/qwen25-05b-instruct")
    )
    parser.add_argument("--goal", default="Stay alive and inspect the world.")
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

    from tunix_craftext.grpo_profile import (
        build_grpo_evidence_manifest,
        load_agentic_grpo_profile,
    )

    if args.profile is not None:
        profile = load_agentic_grpo_profile(args.profile)
        args.config = profile.environment_config
        args.topology = profile.topology_config
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
    from tunix_craftext.tunix import (
        AgenticGrpoWorkloadSpec,
        load_tunix_topology,
        pinned_qwen_tensor_shape,
        validate_agentic_grpo_preflight,
    )

    config = load_mvp_config(args.config)
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
    validate_agentic_grpo_preflight(topology, spec, pinned_qwen_tensor_shape())

    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": "tunix-craftext.agentic-grpo-dry-run/v1",
                    "config": str(args.config),
                    "topology": str(args.topology),
                    "snapshot": str(args.snapshot),
                    "snapshot_exists": args.snapshot.is_dir(),
                    "backend": jax.default_backend(),
                    "batch_preview": next(
                        task_batches(
                            goal=args.goal,
                            seed=config.run.seed,
                            batch_size=args.batch_size,
                            count=1,
                            horizon=config.environment.horizon,
                        )
                    )["seed"].tolist(),
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
        from tunix_craftext.agentic_grpo_smoke import (
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

    from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
    from tunix.rl.agentic.agents.tool_agent import ToolAgent
    from tunix.rl.agentic.parser.chat_template_parser.parser import QwenChatTemplateParser

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
    cluster = build_agentic_grpo_cluster(topology, spec, assets)
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
        learner.train(
            task_batches(
                goal=args.goal,
                seed=config.run.seed,
                batch_size=args.batch_size,
                count=args.max_steps,
                horizon=config.environment.horizon,
            ),
            skip_jit=args.skip_jit,
        )
    finally:
        cluster.close()


if __name__ == "__main__":
    main()
