"""vLLM memory preflight diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from tunix_craftext.diagnostics.vllm_memory import (
    CudaMemorySnapshot,
    estimate_vllm_memory_from_config,
)


def test_vllm_memory_estimate_flags_reservation_that_exceeds_free_memory(
    tmp_path: Path,
) -> None:
    config = _write_generation_config(
        tmp_path,
        gpu_memory_utilization=0.35,
        model="Qwen/Qwen2.5-0.5B-Instruct",
    )

    estimate = estimate_vllm_memory_from_config(
        config,
        memory_provider=lambda _: CudaMemorySnapshot(
            free_gib=6.82,
            total_gib=31.36,
            device="NVIDIA RTX",
        ),
    )

    assert round(estimate.requested_gib, 2) == 10.98
    assert estimate.fits_current_free_memory is False
    assert "vLLM reservation exceeds current free memory plus safety margin" in estimate.notes


def test_vllm_memory_estimate_reads_local_weights_and_qwen_kv_cache(
    tmp_path: Path,
) -> None:
    model = tmp_path / "qwen"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"0" * 1024)
    (model / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 896,
                "num_hidden_layers": 24,
                "num_attention_heads": 14,
                "num_key_value_heads": 2,
            }
        ),
        encoding="utf-8",
    )
    config = _write_generation_config(
        tmp_path,
        gpu_memory_utilization=0.20,
        model=str(model),
    )

    estimate = estimate_vllm_memory_from_config(
        config,
        memory_provider=lambda _: CudaMemorySnapshot(
            free_gib=20.0,
            total_gib=32.0,
            device="NVIDIA RTX",
        ),
    )

    assert estimate.fits_current_free_memory is True
    assert estimate.snapshot_weights_gib is not None
    assert estimate.estimated_kv_cache_gib is not None
    assert estimate.estimated_model_plus_kv_gib is not None


def _write_generation_config(
    tmp_path: Path,
    *,
    gpu_memory_utilization: float,
    model: str,
) -> Path:
    path = tmp_path / "generation.yaml"
    path.write_text(
        f"""
schema_version: 1
engine:
  name: qwen-vllm
  backend: vllm-offload
  model: {model}
  tensor_parallel_size: 1
  max_model_len: 2048
  dtype: bfloat16
  mode: sync
  policy_version: 0
  metadata:
    gpu_memory_utilization: {gpu_memory_utilization}
    max_num_batched_tokens: 4096
    max_num_seqs: 32
tunix:
  engine: vllm
  max_prompt_length: 1024
  max_tokens_to_generate: 32
  temperature: 1.0
  kv_cache_size: 2048
  return_logprobs: true
  tensor_parallel_size: 1
  data_parallel_size: -1
  expert_parallel_size: 1
  vllm_server_mode: false
  vllm_async_scheduling: false
  vllm_hbm_utilization: {gpu_memory_utilization}
  vllm_model_version: qwen2.5-0.5b
  vllm_init_with_random_weights: false
  vllm_max_num_batched_tokens: 4096
  vllm_max_num_seqs: 32
  vllm_kwargs: {{}}
  vllm_sampling_kwargs: {{}}
async:
  max_in_flight: 1
  queue_maxsize: 8
  submission_timeout_s: 0.0
""",
        encoding="utf-8",
    )
    return path
