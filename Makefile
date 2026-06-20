PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PERF_ARTIFACT ?= artifacts/benchmarks/rollout-latest.json

.PHONY: audit test integration perf docs api-docs serve provenance sync-tasks verify

audit:
	$(PYTHON) .codex/skills/repository-audit/scripts/audit_repo.py

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/unit

integration:
	PYTHONPATH=src $(PYTHON) -m pytest -m integration

perf:
	mkdir -p artifacts/benchmarks
	PYTHONPATH=src $(PYTHON) -m pytest -m performance --benchmark-only --benchmark-json=$(PERF_ARTIFACT)

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

verify: audit test integration sync-tasks api-docs
	@echo "Verified audit, unit/integration tests, task-board sync, MkDocs and Sphinx API documentation."
	@echo "For any hot-path change, also run 'make perf' and save or explicitly waive benchmark evidence."
