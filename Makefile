.PHONY: test integration perf docs serve provenance

test:
	pytest tests/unit

integration:
	pytest -m integration

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
