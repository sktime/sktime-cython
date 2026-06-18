# sktime-cython

Cython extensions for [sktime](https://github.com/sktime/sktime).

This repository is an empty Python package template with GitHub Actions CI/CD configured for Cython-based packages.

## Overview

- **Python package**: `sktime_cython`
- **Build system**: setuptools + Cython
- **CI**: GitHub Actions (test + release)

## Development Setup

```bash
# Install build dependencies and package in editable mode
pip install "cython>=3.0" "numpy>=1.21" setuptools
pip install -e ".[dev]" --no-build-isolation

# Run tests
python -m pytest sktime_cython/ -v
```

Or using `make`:

```bash
make install  # install with build dependencies
make test     # run tests
make clean    # clean build artifacts
```

## CI Workflows

### `test.yml`

Runs on every push and pull request to `main`. Jobs:

- **`code-quality`**: runs pre-commit hooks (ruff formatting and linting)
- **`test-nosoftdeps`**: builds Cython extensions and runs tests on Python 3.11 / Ubuntu
- **`test-full`**: builds Cython extensions and runs tests across Python 3.10–3.13 on Ubuntu, macOS, and Windows

The `test.yml` workflow installs Cython and NumPy before building the package, so `.pyx` files are compiled and available for testing.

### `release.yml`

Triggered on GitHub release. Jobs:

- **`check_tag`**: verifies the release tag matches the version in `pyproject.toml`
- **`build_wheels`**: uses [`cibuildwheel`](https://cibuildwheel.pypa.io/) to build binary wheels for Linux, macOS (Intel + Apple Silicon), and Windows
- **`build_sdist`**: builds a source distribution (`.tar.gz`) that includes the `.pyx` source files
- **`upload_wheels`**: publishes all wheels and the sdist to PyPI using trusted publishing

## Package Structure

```
sktime_cython/
  __init__.py                         # package root
  _cython/
    __init__.py
    example.pyx                       # example Cython extension
    tests/
      __init__.py
      test_example.py                 # tests for the example extension
```

## License

BSD 3-Clause License - see [LICENSE](LICENSE).