.PHONY: install data features train eval compare explain dashboard api api-prod drift test lint format clean help

PYTHON ?= python3

help:
	@echo "industrial-anomaly-detection — make targets"
	@echo ""
	@echo "  install     install package + dev deps in editable mode"
	@echo "  data        download raw CWRU bearing dataset to data/raw/"
	@echo "  features    extract time + frequency features → data/features/"
	@echo "  train       fit unsupervised models on the feature matrix"
	@echo "  eval        compute metrics with bootstrap CI on a held-out set"
	@echo "  explain     generate SHAP plots for predicted anomalies"
	@echo "  dashboard   launch the Streamlit dashboard locally"
	@echo "  api         launch FastAPI dev server at http://localhost:8000"
	@echo "  api-prod    launch FastAPI production server (no --reload)"
	@echo "  drift       compute PSI drift report → results/drift_report.json"
	@echo "  test        run pytest with coverage"
	@echo "  lint        run ruff check + format check"
	@echo "  format      apply ruff format"
	@echo "  clean       remove caches, build artefacts and coverage"

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m src.cli download

features:
	$(PYTHON) -m src.cli features

train:
	$(PYTHON) -m src.cli train

eval:
	$(PYTHON) -m src.cli eval

compare:
	$(PYTHON) -m src.cli compare

explain:
	$(PYTHON) -m src.cli explain

dashboard:
	streamlit run src/dashboard.py

api:
	$(PYTHON) -m src.cli api

api-prod:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1

drift:
	$(PYTHON) -m src.cli drift

test:
	$(PYTHON) -m pytest -v --cov=src --cov-report=term-missing tests/

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
