# EventFlow — development targets (phase 0)
#
# Run inside the dev venv:  pip install -e ".[dev]"
# Then:  make audit   (single command, writes audit/phase-1/)
#
# PYTHON can be overridden, e.g.:  make audit PYTHON=/path/to/venv/bin/python

PYTHON ?= python

.PHONY: audit test lint type

audit:
	$(PYTHON) scripts/audit.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m black --check src tests

type:
	$(PYTHON) -m mypy src
