PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: audit test integration perf docs api-docs serve provenance verify

audit:
	$(PYTHON) .codex/skills/repository-audit/scripts/audit_repo.py

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/unit

integration:
	PYTHONPATH=src $(PYTHON) -m pytest -m integration

perf:
	$(PYTHON) -m pytest -m performance --benchmark-only

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

verify: audit test integration docs api-docs
	@echo "Verified audit, unit/integration tests, MkDocs and Sphinx API documentation."
	@echo "For any hot-path change, also run 'make perf' and save or explicitly waive benchmark evidence."
