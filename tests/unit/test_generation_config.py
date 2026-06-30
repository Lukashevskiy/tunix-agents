"""Strict YAML contracts for rollout generation pipelines."""

from __future__ import annotations

from pathlib import Path

import pytest

from tunix_craftext.inference import (
    InferenceBackendError,
    generation_config_to_manifest,
    load_generation_pipeline_config,
)

ROOT = Path(__file__).resolve().parents[2]
SYNC_CONFIG = ROOT / "configs/generation/qwen_vllm_sync.yaml"
ASYNC_CONFIG = ROOT / "configs/generation/qwen_vllm_async.yaml"


def test_sync_generation_config_loads_engine_and_tunix_contract() -> None:
    config = load_generation_pipeline_config(SYNC_CONFIG)

    assert config.profile.name == "qwen25-05b-vllm-sync"
    assert config.profile.backend == "vllm-offload"
    assert config.profile.mode == "sync"
    assert config.tunix.engine == "vllm"
    assert config.tunix.vllm_server_mode is False
    assert config.async_collection.max_in_flight == 1

    manifest = generation_config_to_manifest(config, path=SYNC_CONFIG)
    assert manifest["path"].endswith("configs/generation/qwen_vllm_sync.yaml")
    assert manifest["engine"]["tensor_parallel_size"] == 1
    assert manifest["tunix"]["rollout_vllm_model_version"] == "qwen2.5-0.5b"


def test_async_generation_config_loads_bounded_collector_settings() -> None:
    config = load_generation_pipeline_config(ASYNC_CONFIG)

    assert config.profile.name == "qwen25-05b-vllm-async"
    assert config.profile.mode == "async"
    assert config.tunix.vllm_server_mode is True
    assert config.tunix.vllm_async_scheduling is True
    assert config.async_collection.max_in_flight == 2
    assert config.async_collection.queue_maxsize == 16


def test_generation_config_rejects_unknown_root_keys(tmp_path: Path) -> None:
    invalid = tmp_path / "generation.yaml"
    invalid.write_text(
        SYNC_CONFIG.read_text(encoding="utf-8") + "\nextra: nope\n",
        encoding="utf-8",
    )

    with pytest.raises(InferenceBackendError, match="root keys"):
        load_generation_pipeline_config(invalid)


def test_generation_config_rejects_engine_model_len_smaller_than_tunix_contract(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "generation.yaml"
    invalid.write_text(
        SYNC_CONFIG.read_text(encoding="utf-8").replace("max_model_len: 2048", "max_model_len: 64"),
        encoding="utf-8",
    )

    with pytest.raises(InferenceBackendError, match="engine.max_model_len"):
        load_generation_pipeline_config(invalid)
