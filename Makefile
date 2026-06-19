.PHONY: audit test integration perf docs serve provenance verify

audit:
	python .codex/skills/repository-audit/scripts/audit_repo.py

test:
	PYTHONPATH=src pytest tests/unit

integration:
	PYTHONPATH=src pytest -m integration

perf:
	pytest -m performance --benchmark-only

docs:
	python scripts/generate_dashboard.py
	mkdocs build --strict

serve:
	python scripts/generate_dashboard.py
	mkdocs serve

provenance:
	python scripts/export_provenance.py

verify: audit test integration docs
	@echo "Verified audit, unit/integration tests and generated documentation."
	@echo "For any hot-path change, also run 'make perf' and save or explicitly waive benchmark evidence."
