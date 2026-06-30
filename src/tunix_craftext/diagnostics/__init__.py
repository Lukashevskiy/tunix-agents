"""Diagnostics helpers for target-platform bring-up."""

from .accelerator_stack import build_accelerator_stack_report
from .vllm_memory import estimate_vllm_memory_from_config

__all__ = ["build_accelerator_stack_report", "estimate_vllm_memory_from_config"]
