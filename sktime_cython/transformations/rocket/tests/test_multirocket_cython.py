"""Tests for the Cython MultiRocket transform.

Mirrors ``test_minirocket_cython``: the shape / threading / guard tests are
self-contained (no sktime) so they run in cibuildwheel's isolated wheel-test
env, and the equivalence-vs-numba test is skipped where sktime is absent.
"""

import importlib.util

import numpy as np
import pytest

from sktime_cython.transformations.rocket._multirocket import (
    multirocket_fit,
    multirocket_transform,
)

_HAS_SKTIME = importlib.util.find_spec("sktime") is not None


def _sktime_has_flag():
    """Whether the installed sktime exposes ``original_implementation``."""
    if not _HAS_SKTIME:
        return False
    from sktime.transformations.rocket import MultiRocketMultivariate

    return "original_implementation" in MultiRocketMultivariate.get_param_names()


# At large num_kernels a value can sit on the bias threshold and be classified
# differently by the two float32 summation orders, flipping the PPV/MPV/MIPV
# triplet of one feature. Measured rate is ~2e-5 of elements; allow 1e-4.
_MISMATCH_BUDGET = 1e-4


def _panel(seed, n_columns=3, n_timepoints=60):
    rng = np.random.RandomState(seed)
    return rng.normal(size=(6, n_columns, n_timepoints)).astype(np.float64)


@pytest.mark.skipif(not _HAS_SKTIME, reason="sktime not installed (dev extra)")
@pytest.mark.parametrize("original_implementation", [False, True])
@pytest.mark.parametrize("n_columns", [1, 4])
@pytest.mark.parametrize(
    "num_kernels,max_dilations_per_kernel,random_state,n_timepoints",
    [(84, 32, 42, 60), (168, 16, 7, 60), (6250, 32, 0, 137)],
)
def test_cython_matches_numba(
    n_columns,
    num_kernels,
    max_dilations_per_kernel,
    random_state,
    n_timepoints,
    original_implementation,
):
    """Cython transform must match the numba implementation (groundtruth)."""
    from sktime.transformations.rocket import MultiRocketMultivariate

    kwargs = {}
    if _sktime_has_flag():
        kwargs["original_implementation"] = original_implementation
    elif not original_implementation:
        # sktime before the fix only implements the original behaviour
        pytest.skip("installed sktime predates original_implementation")

    X = _panel(random_state, n_columns=n_columns, n_timepoints=n_timepoints)

    numba_out = MultiRocketMultivariate(
        num_kernels=num_kernels,
        max_dilations_per_kernel=max_dilations_per_kernel,
        random_state=random_state,
        **kwargs,
    ).fit_transform(X)

    params = multirocket_fit(
        X,
        num_kernels=num_kernels,
        max_dilations_per_kernel=max_dilations_per_kernel,
        random_state=random_state,
    )
    cython_out = multirocket_transform(
        X, params, original_implementation=original_implementation
    )

    assert cython_out.shape == numba_out.shape
    close = np.isclose(cython_out, numba_out.to_numpy(), rtol=1e-4, atol=1e-5)
    assert 1 - close.mean() <= _MISMATCH_BUDGET


@pytest.mark.parametrize("num_kernels", [84, 168])
def test_output_shape(num_kernels):
    """transform yields 4 features per kernel, on X and its difference."""
    X = _panel(0)
    out = multirocket_transform(
        X, multirocket_fit(X, num_kernels=num_kernels, random_state=0)
    )
    assert out.shape == (X.shape[0], 2 * 4 * (num_kernels // 84) * 84)
    assert out.dtype == np.float32


def test_threaded_matches_serial():
    """n_jobs>1 must match the single-threaded result."""
    X = _panel(3)
    params = multirocket_fit(X, num_kernels=168, random_state=1)
    np.testing.assert_array_equal(
        multirocket_transform(X, params, n_jobs=1),
        multirocket_transform(X, params, n_jobs=4),
    )


def test_float32_and_float64_both_accepted():
    """The kernels are fused over float32/float64; both dispatch."""
    X = _panel(5)
    params = multirocket_fit(X, num_kernels=84, random_state=0)
    out64 = multirocket_transform(X, params)
    out32 = multirocket_transform(X.astype(np.float32), params)
    assert out64.shape == out32.shape
    # the two dtypes round differently, so a value sitting on a bias threshold
    # can be classified either way; budget for that rather than requiring
    # elementwise closeness
    close = np.isclose(out64, out32, rtol=1e-3, atol=1e-3)
    assert 1 - close.mean() <= 1e-2


@pytest.mark.parametrize("bad", [0, 40])
def test_invalid_dilations_raise(bad):
    """Dilations outside the kernel's in-bounds invariants are rejected."""
    X = _panel(0)
    p, p1 = multirocket_fit(X, num_kernels=84, random_state=0)
    p = (*p[:2], np.array([bad], dtype=np.int32), *p[3:])
    with pytest.raises(ValueError, match="dilations"):
        multirocket_transform(X, (p, p1))
