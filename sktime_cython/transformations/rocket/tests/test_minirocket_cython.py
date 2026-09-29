"""Tests for the Cython MiniRocket transform.

The shape / threading / guard tests are self-contained (no sktime) so they run
in cibuildwheel's isolated wheel-test env, which installs only pytest. The
equivalence-vs-numba test imports sktime lazily and is skipped where sktime is
absent (install the ``dev`` extra to run it).
"""

import importlib.util

import numpy as np
import pytest

from sktime_cython.transformations.rocket._minirocket import (
    rocket_fit,
    rocket_transform,
)

# sktime is only present with the `dev` extra; the cibuildwheel wheel-test env
# installs pytest only. find_spec detects absence without importing.
_HAS_SKTIME = importlib.util.find_spec("sktime") is not None


def _panel(seed, n_columns=3, n_timepoints=60):
    rng = np.random.RandomState(seed)
    return rng.normal(size=(6, n_columns, n_timepoints)).astype(np.float32)


@pytest.mark.skipif(not _HAS_SKTIME, reason="sktime not installed (dev extra)")
@pytest.mark.parametrize("n_columns", [1, 4])
@pytest.mark.parametrize(
    "num_kernels,max_dilations_per_kernel,random_state",
    [(84, 32, 42), (168, 16, 7), (84, 32, 0)],
)
def test_cython_matches_numba(
    n_columns, num_kernels, max_dilations_per_kernel, random_state
):
    """Cython transform must match the numba implementation (groundtruth)."""
    from sktime.transformations.rocket import MiniRocketMultivariate

    X = _panel(random_state, n_columns=n_columns)

    numba_out = MiniRocketMultivariate(
        num_kernels=num_kernels,
        max_dilations_per_kernel=max_dilations_per_kernel,
        random_state=random_state,
    ).fit_transform(X)

    params = rocket_fit(
        X,
        num_kernels=num_kernels,
        max_dilations_per_kernel=max_dilations_per_kernel,
        random_state=random_state,
    )
    cython_out = rocket_transform(X, params)

    assert cython_out.shape == numba_out.shape
    np.testing.assert_allclose(cython_out, numba_out.to_numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("num_kernels", [84, 168])
def test_output_shape(num_kernels):
    """transform yields (n_instances, num_kernels) float32 features."""
    X = _panel(0)
    out = rocket_transform(X, rocket_fit(X, num_kernels=num_kernels, random_state=0))
    assert out.shape == (X.shape[0], num_kernels)
    assert out.dtype == np.float32


def test_threaded_matches_serial():
    """n_jobs>1 must match the single-threaded result."""
    X = _panel(3)
    params = rocket_fit(X, num_kernels=168, random_state=1)
    np.testing.assert_array_equal(
        rocket_transform(X, params, n_jobs=1), rocket_transform(X, params, n_jobs=4)
    )


def test_too_short_series_raises():
    """n_timepoints < 9 is rejected."""
    X = _panel(0, n_timepoints=8)
    with pytest.raises(ValueError, match="n_timepoints must be >= 9"):
        rocket_fit(X)


@pytest.mark.skipif(not _HAS_SKTIME, reason="sktime not installed (dev extra)")
@pytest.mark.parametrize(
    "num_kernels,max_dilations_per_kernel,random_state",
    [(84, 32, 42), (168, 16, 7), (84, 32, 0)],
)
def test_univariate_matches_numba(num_kernels, max_dilations_per_kernel, random_state):
    """2D input must match sktime's univariate numba ``MiniRocket``."""
    from sktime.transformations.rocket import MiniRocket

    X = _panel(random_state, n_columns=1)
    expected = MiniRocket(
        num_kernels=num_kernels,
        max_dilations_per_kernel=max_dilations_per_kernel,
        random_state=random_state,
    ).fit_transform(X)

    X2d = X[:, 0, :]
    params = rocket_fit(
        X2d,
        num_kernels=num_kernels,
        max_dilations_per_kernel=max_dilations_per_kernel,
        random_state=random_state,
    )
    np.testing.assert_allclose(
        rocket_transform(X2d, params), expected.to_numpy(), rtol=1e-4, atol=1e-5
    )


def test_2d_equals_single_channel_3d():
    """(n, d) input is exactly the (n, 1, d) single-channel case."""
    X = _panel(5, n_columns=1)
    params = rocket_fit(X[:, 0, :], num_kernels=168, random_state=2)
    for a, b in zip(params, rocket_fit(X, num_kernels=168, random_state=2)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        rocket_transform(X[:, 0, :], params, n_jobs=3), rocket_transform(X, params)
    )


@pytest.mark.parametrize("shape", [(0, 20), (0, 1, 20), (20,), (1, 1, 1, 20)])
def test_invalid_shape_raises(shape):
    """Input must be a non-empty 2D or 3D panel."""
    with pytest.raises(ValueError):
        rocket_fit(np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("n_columns", [1, 3])
def test_float64_matches_float32_cast(n_columns):
    """float64 input is read without a copy but computed exactly as float32."""
    X = _panel(6, n_columns=n_columns).astype(np.float64) * np.pi
    X32 = X.astype(np.float32)
    params = rocket_fit(X, num_kernels=168, random_state=4)
    for a, b in zip(params, rocket_fit(X32, num_kernels=168, random_state=4)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        rocket_transform(X, params, n_jobs=2), rocket_transform(X32, params)
    )
    np.testing.assert_array_equal(
        rocket_transform(X.astype(np.int64), params),
        rocket_transform(X.astype(np.int64).astype(np.float32), params),
    )
