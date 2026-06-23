PACKAGE=sktime_cython

# Use uv if it's on PATH (its venv hides bare `pip`/`python`), else fall back.
ifeq ($(shell command -v uv 2>/dev/null),)
PIP := pip
PYTHON := python
else
PIP := uv pip
PYTHON := uv run python
endif

.PHONY: help install build lint test clean

.DEFAULT_GOAL := help

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
		 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode with dev dependencies
	$(PIP) install -e ".[dev]"

build: ## Recompile Cython extensions (editable reinstall, uses build isolation)
	$(PIP) install -e . --force-reinstall --no-deps

lint: ## Run the pre-commit hooks (ruff lint + format) on all files, as CI does
	$(PYTHON) -m pre_commit run --all-files

test: ## Run unit tests
	$(PYTHON) -m pytest $(PACKAGE)/ -v

clean: ## Clean build artifacts
	rm -rf ./dist
	rm -rf ./build
	rm -rf ./wheelhouse
	rm -rf ./*.egg-info
	find $(PACKAGE) -type f -name "*.so" -delete
	find $(PACKAGE) -type f -name "*.pyd" -delete
	find $(PACKAGE) -type f -name "*.c" -delete
	find $(PACKAGE) -type f -name "*.pyc" -delete
	find $(PACKAGE) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
