"""Keep public tutorial notebooks valid and discoverable without requiring Jupyter at test time."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "examples" / "notebooks"


def test_example_notebooks_are_valid_nbformat_with_runnable_imports() -> None:
    """Validate the notebook JSON and its primary public API import.

    :raises AssertionError: If a tutorial notebook disappears, has invalid format or no longer
        teaches the corresponding public symbol.
    """
    expected = {
        "01_rollout_contract.ipynb": "collect_rollout",
        "02_craftext_adapter.ipynb": "CrafTextAdapter",
        "03_model_interop_lora.ipynb": "merge_lora_adapters",
        "04_megaprompts_environment_to_prompt.ipynb": "MegaPromptRenderer",
        "05_caged_random_policy_trajectory.ipynb": "sample_masked_actions",
        "06_qwen_craftext_manual_episode.ipynb": "QwenTunixBackend",
        "07_qwen_craftext_full_trajectory.ipynb": "collect_batched_text_rollout",
        "08_parallel_craftext_pipeline.ipynb": "jax.vmap",
        "09_batched_qwen_craftext_rollout.ipynb": "collect_batched_text_rollout",
        "10_replay_to_token_ppo.ipynb": "HybridPpoStep",
        "11_end_to_end_batched_qwen_ppo.ipynb": "hybrid_step_from_text_trajectory",
        "12_full_cycle_craftext_training.ipynb": "last_valid_token_values",
        "13_replay_visualization.ipynb": "load_trajectory",
        "14_generation_benchmark.ipynb": "build_gemma_tunix_actor",
        "15_agentic_grpo_full_trainer.ipynb": "GRPOLearner",
        "16_server_grpo_object_training.ipynb": "JsonlRunLogger",
    }
    for filename, required_symbol in expected.items():
        notebook = json.loads((NOTEBOOKS / filename).read_text(encoding="utf-8"))
        source = "".join("".join(cell["source"]) for cell in notebook["cells"])
        assert notebook["nbformat"] == 4
        assert required_symbol in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(
                    "".join(cell["source"]),
                    f"{filename}:cell-{index}",
                    "exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                )


def test_batched_qwen_ppo_notebook_handles_fallback_only_evidence() -> None:
    """Notebook 11 must not crash when Qwen rollout produces fallback-only rows."""
    notebook = json.loads(
        (NOTEBOOKS / "11_end_to_end_batched_qwen_ppo.ipynb").read_text(encoding="utf-8")
    )
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "actor_loss_tokens = int(jnp.sum(hybrid_step.actor_loss_token_mask))" in source
    assert "if actor_loss_tokens:" in source
    assert "fallback-only evidence; actor loss skipped" in source
