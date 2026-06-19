.PHONY: test integration perf docs provenance

test:
	pytest tests/unit

integration:
	pytest -m integration

perf:
	pytest -m performance --benchmark-only

docs:
	mkdocs build --strict

provenance:
	python scripts/export_provenance.py
