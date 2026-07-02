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
        "19_host_prompt_threading_profile.ipynb": "HostBatchPolicy",
        "20_env_device_policy_benchmark.ipynb": "EnvironmentDevicePolicy",
        "21_sync_vllm_grpo_learning.ipynb": "GRPOLearner",
        "22_external_vllm_sync_grpo_rollout.ipynb": "external_grpo_batch_from_replays",
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


def test_sync_vllm_grpo_notebook_uses_profile_path_and_placement_contracts() -> None:
    """Notebook 21 should stay aligned with profile path and rollout placement policy."""
    notebook = json.loads(
        (NOTEBOOKS / "21_sync_vllm_grpo_learning.ipynb").read_text(encoding="utf-8")
    )
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "resolve_profile_path" in source
    assert "ROOT / profile." not in source
    assert "EnvironmentDevicePolicy" in source
    assert "HostBatchPolicy" in source
    assert "JAX_PLATFORMS=cpu" in source
    assert "local_vllm_rollout_contract" in source
    assert "sync_vllm_generation.vllm_model_version == str(snapshot_path)" in source
    assert "MetricLoggerFactory" in source
    assert "CompositeArtifactSink" in source
    assert "MetricRecord" not in source
    assert "logger.log_metric" not in source


def test_external_vllm_grpo_notebook_uses_direct_engine_and_evidence_contract() -> None:
    """Notebook 22 should keep GRPO startup on the known-working external vLLM path."""
    notebook = json.loads(
        (NOTEBOOKS / "22_external_vllm_sync_grpo_rollout.ipynb").read_text(encoding="utf-8")
    )
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "VllmInferenceEngine.from_profile" in source
    assert "RequestsLlmBackend(engine)" in source
    assert "external_grpo_batch_from_replays" in source
    assert "evaluate_external_llm_actor_grpo" in source
    assert "token_batch_from_external_grpo" in source
    assert "external_grpo_update" in source
    assert "mean logprob delta after one update" in source
    assert "MetricLoggerFactory" in source
    assert "CompositeArtifactSink" in source
    assert "live_metric_pipeline.log" in source
    assert "metric snapshots jsonl" in source
    assert "save_external_grpo_batch" in source
    assert "cpu_environment_device_policy()" in source
    assert "GRPOLearner" not in source


def test_server_grpo_notebook_uses_metric_pipeline_not_manual_records() -> None:
    """Notebook 16 should log training/readiness metrics through the metric factory."""
    notebook = json.loads(
        (NOTEBOOKS / "16_server_grpo_object_training.ipynb").read_text(encoding="utf-8")
    )
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "MetricLoggerFactory" in source
    assert "CompositeArtifactSink" in source
    assert "MetricRecord" not in source
    assert "logger.log_metric" not in source


def test_rollout_notebooks_pin_cpu_env_sidecar_policy() -> None:
    """Rollout profiling examples should remember the explicit CPU env policy."""
    sync_notebook = json.loads(
        (NOTEBOOKS / "17_sync_vllm_craftext_rollout.ipynb").read_text(encoding="utf-8")
    )
    device_notebook = json.loads(
        (NOTEBOOKS / "20_env_device_policy_benchmark.ipynb").read_text(encoding="utf-8")
    )
    sync_source = "".join("".join(cell["source"]) for cell in sync_notebook["cells"])
    device_source = "".join("".join(cell["source"]) for cell in device_notebook["cells"])

    assert "cpu_environment_device_policy()" in sync_source
    assert "explicit_cpu_sidecar" in device_source
    assert "cpu_environment_device_policy()" in device_source
