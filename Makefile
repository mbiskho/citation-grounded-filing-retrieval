.PHONY: install test cov lint fmt type ingest index ask app goldset calibrate evaluate eval clean all

PY ?= python3

install:
	$(PY) -m pip install -e ".[dev,ingest,app]"

test:
	pytest -q

cov:
	pytest --cov=filingqa --cov-report=term-missing

lint:
	ruff check src tests app

fmt:
	ruff format src tests
	ruff check --fix src tests

type:
	mypy

ingest:
	filingqa ingest --user-agent "filingqa-research $(USER)@example.com"

index:
	filingqa index

ask:
	filingqa ask "$(Q)"

app:
	filingqa app

goldset:
	filingqa goldset

calibrate:
	filingqa calibrate

evaluate:
	filingqa evaluate --split test

# Full reproduction: regenerate the gold set, tune on dev, score on test.
eval: goldset calibrate evaluate

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

all: lint type test evaluate
