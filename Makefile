PACKAGE=sktime_cython

.PHONY: help install build test clean

.DEFAULT_GOAL := help

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
		 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode with dev dependencies
	pip install cython numpy setuptools
	pip install -e ".[dev]" --no-build-isolation

build: ## Build Cython extensions in-place
	python setup.py build_ext --inplace

test: ## Run unit tests
	python -m pytest $(PACKAGE)/ -v

clean: ## Clean build artifacts
	rm -rf ./dist
	rm -rf ./build
	rm -rf ./wheelhouse
	rm -rf ./*.egg-info
	find . -type f -name "*.so" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name "*.c" -path "*/$(PACKAGE)/*" -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
