#!/usr/bin/env python3
"""Run the explicit Tunix Agentic GRPO CrafText golden pipeline."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--skip-jit", action="store_true")
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="Attempt the unsupported local CPU Qwen rollout for upstream debugging only.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Build assets, cluster and learner, then perform the requested GRPO updates."""
    args = parse_args(arguments)
    if not args.snapshot.is_dir():
        raise FileNotFoundError(f"Expected explicit local Qwen snapshot: {args.snapshot}")
    if args.max_steps <= 0 or args.batch_size <= 0 or args.num_generations < 2:
        raise ValueError("max_steps/batch_size must be positive and num_generations at least two")

    import jax

    if jax.default_backend() == "cpu" and not args.allow_cpu_smoke:
        raise RuntimeError(
            "Agentic Qwen GRPO requires an accelerator mesh. The pinned Tunix Qwen vanilla "
            "sampler cannot resolve its sharded gather on a local CPU singleton mesh; use an "
            "accelerator runner or pass --allow-cpu-smoke only to reproduce that upstream failure."
        )

    from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
    from tunix.rl.agentic.agents.tool_agent import ToolAgent
    from tunix.rl.agentic.parser.chat_template_parser.parser import QwenChatTemplateParser

    from tunix_craftext.agentic_craftext import (
        CrafTextAgenticEnvironment,
        CrafTextStepTool,
        agentic_environment_kwargs,
    )
    from tunix_craftext.config import load_mvp_config
    from tunix_craftext.rlcluster_workload import (
        AgenticGrpoWorkloadSpec,
        build_agentic_grpo_cluster,
        load_agentic_grpo_qwen_assets,
    )
    from tunix_craftext.tunix_adapter import load_qwen_hf_tokenizer
    from tunix_craftext.tunix_topology import load_tunix_topology

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
        num_generations=args.num_generations,
        max_concurrency=args.num_generations,
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
