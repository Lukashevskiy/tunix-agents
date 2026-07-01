UV_RUN_FLAGS ?=
PYTHON ?= uv run $(UV_RUN_FLAGS) python
PERF_ARTIFACT ?= artifacts/benchmarks/rollout-latest.json

.PHONY: audit test coverage integration perf perf-env perf-text docs api-docs serve provenance sync-tasks accelerator-stack vllm-memory verify verify-golden

audit:
	$(PYTHON) .codex/skills/repository-audit/scripts/audit_repo.py

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/unit

coverage:
	PYTHONPATH=src $(PYTHON) -m pytest tests/unit --cov=src/tunix_craftext --cov-report=term-missing:skip-covered --cov-report=xml --cov-fail-under=83

integration:
	PYTHONPATH=src $(PYTHON) -m pytest -m integration

perf:
	mkdir -p artifacts/benchmarks
	PYTHONPATH=src $(PYTHON) -m pytest -m performance --benchmark-only --benchmark-json=$(PERF_ARTIFACT)

perf-env:
	PYTHONPATH=src $(PYTHON) scripts/benchmark_environments.py --configs configs/env/benchmarks/craftext_full.yaml configs/env/benchmarks/craftext_tiny.yaml configs/env/benchmarks/caged_craftext_full.yaml --output artifacts/benchmarks/environment-matrix.json

perf-text:
	PYTHONPATH=src $(PYTHON) scripts/benchmark_text_pipeline.py --isolate-runs --output artifacts/benchmarks/text-pipeline-latest.json

docs:
	$(PYTHON) scripts/generate_dashboard.py
	$(PYTHON) -m mkdocs build --strict

api-docs:
	$(PYTHON) -m sphinx -W --keep-going sphinx site/api

serve:
	$(PYTHON) scripts/generate_dashboard.py
	$(PYTHON) -m mkdocs serve

provenance:
	$(PYTHON) scripts/export_provenance.py

sync-tasks:
	$(PYTHON) .codex/skills/task-board-sync/scripts/sync_task_views.py

accelerator-stack:
	PYTHONPATH=src $(PYTHON) scripts/inspect_accelerator_stack.py --strict

vllm-memory:
	PYTHONPATH=src $(PYTHON) scripts/estimate_vllm_memory.py --config configs/inference/vllm/qwen25_05b_sync.yaml

verify: audit test integration sync-tasks api-docs
	@echo "Verified audit, unit/integration tests, task-board sync, MkDocs and Sphinx API documentation."
	@echo "For any hot-path change, also run 'make perf' and save or explicitly waive benchmark evidence."

verify-golden:
	PYTHONPATH=src $(PYTHON) -m ruff check src tests scripts
	PYTHONPATH=src $(PYTHON) -m mypy src/tunix_craftext
	PYTHONPATH=src $(PYTHON) -m pytest tests/unit/test_agentic_craftext.py tests/unit/test_grpo_profile.py tests/unit/test_rlcluster_workload.py tests/unit/test_run_agentic_grpo.py tests/unit/test_preflight.py
	$(PYTHON) .codex/skills/task-board-sync/scripts/sync_task_views.py
	$(PYTHON) scripts/generate_dashboard.py
	$(PYTHON) -m mkdocs build --strict
	$(PYTHON) .codex/skills/repository-audit/scripts/audit_repo.py
	@echo "Verified golden Agentic GRPO contracts without model downloads or accelerator allocation."
