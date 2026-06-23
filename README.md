# sktime-cython

Cython-compiled estimators for [sktime](https://github.com/sktime/sktime).

This package hosts ahead-of-time compiled implementations of sktime algorithms,
isolating the C-compilation and binary-wheel complexity from the main `sktime`
package. Estimators here expose a plain numpy-in/numpy-out compute layer with
**no sktime runtime dependency**; `sktime` keeps thin `BaseTransformer` /
`BaseEstimator` wrappers that delegate to this package.

Why a separate package: compiled extensions need a C toolchain, per-platform
wheels, and `cibuildwheel` release machinery. Keeping that here lets `sktime`
stay pure-Python while still offering compiled, numba-free fast paths.

## Estimators

| Estimator | Compute API | Notes |
|-----------|-------------|-------|
| Multivariate MiniRocket | `sktime_cython.fit` / `transform` | numba-free; equivalent to `MiniRocketMultivariate`, no JIT warmup |

More estimators are added as `.pyx` kernels under `sktime_cython/_cython/` with
a numpy compute layer alongside.

Each estimator lives in its own submodule (nothing is re-exported at the top
level, so estimators never collide on common names like `fit`/`transform`):

```python
import numpy as np
from sktime_cython.minirocket import fit, transform

X = np.random.rand(100, 6, 500).astype("float32")
params = fit(X, num_kernels=10_000, random_state=42)
features = transform(X, params, n_jobs=-1)   # GIL-released kernel, real threads
```

Runtime dependency: numpy only.

## Development

Requires a C compiler and Python 3.10+. Run from the project root.

```bash
# editable install with the dev extra (pulls sktime + numba for equivalence
# tests, plus pre-commit); compiles the Cython extensions via build isolation
uv pip install -e ".[dev]"        # or: pip install -e ".[dev]"

# install the git pre-commit hooks ONCE — this is what stops CI lint failures
pre-commit install

# run the tests
python -m pytest sktime_cython -v
```

After editing a `.pyx`, recompile with `pip install -e .` again (or `make build`).

`make` shortcuts (auto-detect `uv`, falling back to `pip`/`python`):
`make install`, `make build`, `make test`, `make clean`.

## Before you push (avoid CI failures)

CI runs the pre-commit hooks and **fails if they reformat anything**. Running
`pre-commit install` (above) auto-runs them on every `git commit`. To check the
whole tree on demand:

```bash
pre-commit run --all-files
```

This runs `ruff check` (lint) and `ruff format`. If `ruff format` reports "files
were modified", it already fixed them — `git add` the changes and commit again.

## CI workflows

- **`test.yml`** (push / PR to `main`): a `code-quality` job running the
  pre-commit hooks, plus a matrix that builds the Cython extensions and runs
  pytest across Python 3.10–3.14 on Linux, macOS, and Windows.
- **`release.yml`** (on GitHub release): builds binary wheels with
  [`cibuildwheel`](https://cibuildwheel.pypa.io/) for all platforms, builds an
  sdist (with `.pyx` sources), and publishes to PyPI via trusted publishing.

## Adding an estimator

1. Drop the kernel `.pyx` (and optional `.pyi` stub) under `sktime_cython/_cython/`.
2. Register it as an `Extension` in `setup.py`.
3. Add a numpy compute-layer module `sktime_cython/<name>.py` exposing the
   estimator's public functions. Keep them in the submodule — do not re-export
   at the top level (avoids name collisions across estimators).
4. Add an equivalence test under `sktime_cython/_cython/tests/`.

## Package structure

```
sktime_cython/
  __init__.py                                    # public compute-layer exports
  minirocket.py                                  # numpy fit/transform layer
  _cython/
    __init__.py
    _minirocket_multivariate_cython.pyx          # compiled kernels
    _minirocket_multivariate_cython.pyi          # type stubs
    tests/
      __init__.py
      test_minirocket.py
```

## License

BSD 3-Clause License — see [LICENSE](LICENSE).
