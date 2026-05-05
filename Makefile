.PHONY: install data features train eval explain dashboard test lint format clean help

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
	@echo "  test        run pytest with coverage"
	@echo "  lint        run ruff check + format check"
	@echo "  format      apply ruff format"
	@echo "  clean       remove caches, build artefacts and coverage"

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	@echo "[data] not implemented yet — see PLANO.md sprint 1"

features:
	@echo "[features] not implemented yet — see PLANO.md sprint 1"

train:
	@echo "[train] not implemented yet — see PLANO.md sprint 1"

eval:
	@echo "[eval] not implemented yet — see PLANO.md sprint 1"

explain:
	@echo "[explain] not implemented yet — see PLANO.md sprint 3"

dashboard:
	@echo "[dashboard] not implemented yet — see PLANO.md sprint 4"
	@echo "(once available: streamlit run src/dashboard.py)"

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
