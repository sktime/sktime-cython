"""Tests for the Cython DTW distance.

The self-contained tests (shapes, validation, inf sentinels) run in
cibuildwheel's isolated wheel-test env, which installs only pytest. The
equivalence-vs-numba tests import sktime lazily and are skipped where sktime is
absent (install the ``dev`` extra to run them). Set
``SKTIME_CYTHON_REQUIRE_SKTIME=1`` to turn those skips into failures, so a CI
job that installs the dev extra cannot silently stop comparing.
"""

import importlib.util
import os

import numpy as np
import pytest

from sktime_cython.dists_kernels import _dtw_cython as _cy
from sktime_cython.dists_kernels._dtw import (
    _itakura_parallelogram,
    _sakoe_chiba,
    _to_timeseries,
    dtw_cost_matrix,
    dtw_distance,
)

# sktime is only present with the `dev` extra; the cibuildwheel wheel-test env
# installs pytest only. find_spec detects absence without importing.
_HAS_SKTIME = importlib.util.find_spec("sktime") is not None
_REQUIRE_SKTIME = os.environ.get("SKTIME_CYTHON_REQUIRE_SKTIME") == "1"

# When sktime is required, do not skip: the lazy import inside the test body
# then fails loudly instead of the comparison quietly disappearing.
needs_sktime = pytest.mark.skipif(
    not _HAS_SKTIME and not _REQUIRE_SKTIME,
    reason="sktime not installed (dev extra)",
)

# (m1, m2) pairs: equal, x shorter, x longer - the orientations that used to
# produce a transposed or out-of-bounds bounding matrix.
UNEQUAL_LENGTHS = [(7, 19), (19, 7), (30, 5), (5, 30), (24, 25)]


def _series(seed, d=1, m=40):
    rng = np.random.RandomState(seed)
    return rng.normal(size=(d, m))


def _numba_dtw_distance(x, y, **kwargs):
    from sktime.dists_kernels._numba_distances import dtw_distance as numba_dtw

    return numba_dtw(x, y, **kwargs)


def _numba_cost_matrix(x, y, bounding_matrix):
    """Groundtruth cost matrix: the numba kernel this extension ports.

    Called directly rather than through ``dtw_alignment_path``, which crashes
    on unequal-length series while backtracking the path.
    """
    from sktime.dists_kernels._numba_distances._dtw_numba import _cost_matrix

    return _cost_matrix(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(bounding_matrix, dtype=np.float64),
    )


def _monotone_path(m1, m2):
    """Unit-step monotone path from (0, 0) to (m1 - 1, m2 - 1)."""
    i = j = 0
    cells = [(0, 0)]
    while (i, j) != (m1 - 1, m2 - 1):
        if i < m1 - 1 and (j == m2 - 1 or (i + 1) * m2 <= (j + 1) * m1):
            i += 1
        else:
            j += 1
        cells.append((i, j))
    return cells


def _random_mask(m1, m2, seed, density=0.5):
    """Random (m1, m2) mask that always admits at least one warping path."""
    rng = np.random.RandomState(seed)
    bm = np.where(rng.random_sample((m1, m2)) < density, 0.0, np.inf)
    for i, j in _monotone_path(m1, m2):
        bm[i, j] = 0.0
    return np.ascontiguousarray(bm)


def test_known_values_1d():
    """Matches sktime's documented 1d doctest value."""
    x = np.array([1, 2, 3, 4])
    y = np.array([5, 6, 7, 8])
    assert dtw_distance(x, y) == pytest.approx(58.0)


def test_known_values_2d():
    """Matches sktime's documented multivariate (dependent DTW) doctest value."""
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    y = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])
    assert dtw_distance(x, y) == pytest.approx(512.0)


def test_identical_series_is_zero():
    """DTW of a series with itself is 0."""
    x = _series(0)
    assert dtw_distance(x, x) == pytest.approx(0.0)


def test_symmetry():
    """DTW is symmetric in its arguments."""
    x, y = _series(1), _series(2)
    assert dtw_distance(x, y) == pytest.approx(dtw_distance(y, x))


def test_distance_matches_cost_matrix_corner():
    """The rolling-buffer distance equals cost_matrix[-1, -1] (shared recurrence)."""
    x, y = _series(3), _series(4)
    cm = dtw_cost_matrix(x, y)
    assert dtw_distance(x, y) == pytest.approx(cm[-1, -1])


@pytest.mark.parametrize("d,m", [(1, 25), (3, 30), (5, 18)])
def test_cost_matrix_shape(d, m):
    """cost_matrix is (m1, m2) float64."""
    x, y = _series(5, d=d, m=m), _series(6, d=d, m=m)
    cm = dtw_cost_matrix(x, y)
    assert cm.shape == (m, m)
    assert cm.dtype == np.float64


def test_window_reduces_reachable_region():
    """A tight Sakoe-Chiba band masks off-diagonal cells to inf in the matrix."""
    x, y = _series(7, m=30), _series(8, m=30)
    cm = dtw_cost_matrix(x, y, window=0.1)
    # corners far from the diagonal are out of band -> inf
    assert not np.isfinite(cm[0, -1])
    assert not np.isfinite(cm[-1, 0])
    # the diagonal endpoint stays finite
    assert np.isfinite(cm[-1, -1])


def test_both_bounds_raises():
    """window and itakura_max_slope are mutually exclusive."""
    x, y = _series(0), _series(1)
    with pytest.raises(ValueError, match="only use one bounding matrix"):
        dtw_distance(x, y, window=0.2, itakura_max_slope=0.5)


def test_non_array_raises():
    """Inputs must be numpy arrays."""
    with pytest.raises(ValueError, match="must be a numpy array"):
        dtw_distance([1, 2, 3], np.array([1, 2, 3]))


def test_unbalanced_lengths():
    """DTW between time series of significantly different lengths."""
    x = _series(0, d=2, m=5)
    y = _series(1, d=2, m=30)
    cm = dtw_cost_matrix(x, y)
    assert cm.shape == (5, 30)
    assert dtw_distance(x, y) == pytest.approx(cm[-1, -1])


def test_minimal_length():
    """1-sample time series."""
    x = np.array([[2.0]])
    y = np.array([[5.0]])
    assert dtw_distance(x, y) == pytest.approx(9.0)


def test_non_c_contiguous_input():
    """Handles non-C-contiguous arrays (Fortran-ordered or sliced)."""
    x = np.asfortranarray(_series(0, m=20))
    y = _series(1, m=40)[:, ::2]  # Strided slice
    # Should complete without error or segfault
    res = dtw_distance(x, y)
    assert np.isfinite(res)


def test_distance_and_cost_matrix_parity_with_window():
    """Verify rolling buffer distance equals cost matrix corner WITH windowing."""
    x, y = _series(10, m=30), _series(11, m=35)
    kw = {"window": 0.15}
    cm = dtw_cost_matrix(x, y, **kw)
    dist = dtw_distance(x, y, **kw)
    assert dist == pytest.approx(cm[-1, -1])


def test_fully_masked_window_returns_inf():
    """If the window masks out the path to (m1, m2), distance should be inf."""
    x, y = _series(0, m=10), _series(1, m=10)
    bm = np.full((10, 10), np.inf)
    # Top-left cell valid, but no path to end
    bm[0, 0] = 0.0
    assert np.isinf(dtw_distance(x, y, bounding_matrix=bm))


@pytest.mark.parametrize("func", [dtw_distance, dtw_cost_matrix])
@pytest.mark.parametrize("dx,dy", [(2, 1), (1, 2), (3, 2)])
def test_mismatched_channels_raises(func, dx, dy):
    """Unequal channel counts are rejected, not read past the end of y."""
    x, y = _series(0, d=dx, m=4), _series(1, d=dy, m=4)
    with pytest.raises(ValueError, match="same number of channels"):
        func(x, y)


@pytest.mark.parametrize("func", [dtw_distance, dtw_cost_matrix])
@pytest.mark.parametrize("shape", [(4, 6), (6, 4), (5, 4), (4, 5), (5,), (5, 5, 1)])
def test_bad_bounding_matrix_shape_raises(func, shape):
    """A bounding matrix that is not (m1, m2) is rejected."""
    x, y = _series(0, m=5), _series(1, m=5)
    with pytest.raises(ValueError, match="bounding matrix must have shape"):
        func(x, y, bounding_matrix=np.zeros(shape))


@pytest.mark.parametrize("func", [dtw_distance, dtw_cost_matrix])
def test_transposed_bounding_matrix_raises_for_unequal_lengths(func):
    """An (m2, m1) mask - the old Itakura shape - is rejected."""
    x, y = _series(0, m=8), _series(1, m=13)
    with pytest.raises(ValueError, match="bounding matrix must have shape"):
        func(x, y, bounding_matrix=np.zeros((13, 8)))


@pytest.mark.parametrize("kernel", ["distance", "cost_matrix"])
def test_kernel_validates_shapes_directly(kernel):
    """The Cython kernels validate their own arguments when called directly."""
    func = getattr(_cy, kernel)
    x = np.ascontiguousarray(_series(0, d=2, m=4))
    y = np.ascontiguousarray(_series(1, d=1, m=4))
    with pytest.raises(ValueError, match="same number of channels"):
        func(x, y, np.zeros((4, 4)))

    y2 = np.ascontiguousarray(_series(1, d=2, m=6))
    with pytest.raises(ValueError, match="bounding matrix must have shape"):
        func(x, y2, np.zeros((6, 4)))


@pytest.mark.parametrize("m1,m2", UNEQUAL_LENGTHS)
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"window": 0.0}, {"window": 0.2}, {"window": 1.0}, {"itakura_max_slope": 0.5}],
)
def test_auto_bounding_matrix_is_m1_by_m2(m1, m2, kwargs):
    """Every bounding mode yields an (m1, m2) mask both kernels can index.

    Before the orientation fix, ``window`` raised IndexError whenever
    ``len(x) > len(y)`` and ``itakura_max_slope`` handed the kernels an
    ``(m2, m1)`` mask to read out of bounds.
    """
    x, y = _series(0, d=2, m=m1), _series(1, d=2, m=m2)
    cm = dtw_cost_matrix(x, y, **kwargs)
    assert cm.shape == (m1, m2)
    # every band starts at the origin; a narrow one need not reach the far
    # corner, since unit warping steps cannot follow a steep zero-width line
    assert np.isfinite(cm[0, 0])
    np.testing.assert_allclose(dtw_distance(x, y, **kwargs), cm[-1, -1])


@pytest.mark.parametrize("m1,m2", UNEQUAL_LENGTHS)
@pytest.mark.parametrize("kwargs", [{}, {"window": 1.0}])
def test_wide_band_reaches_far_corner(m1, m2, kwargs):
    """A band wide enough to admit a path gives a finite unequal-length DTW."""
    x, y = _series(2, d=2, m=m1), _series(3, d=2, m=m2)
    assert np.isfinite(dtw_distance(x, y, **kwargs))


@pytest.mark.parametrize("m1,m2", UNEQUAL_LENGTHS)
@pytest.mark.parametrize(
    "builder,arg", [(_sakoe_chiba, 0.2), (_itakura_parallelogram, 0.5)]
)
def test_builders_are_x_by_y_oriented(m1, m2, builder, arg):
    """The mask builders index [i_x, j_y], matching what the kernels read."""
    x, y = _to_timeseries(_series(0, m=m1)), _to_timeseries(_series(1, m=m2))
    bm = builder(x, y, arg)
    assert bm.shape == (m1, m2)
    assert bm.flags["C_CONTIGUOUS"]
    # both ends of the alignment are in band
    assert np.isfinite(bm[0, 0])
    assert np.isfinite(bm[-1, -1])


# equivalence against sktime's numba implementation (groundtruth)
@needs_sktime
@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("m", [10, 37, 50])
@pytest.mark.parametrize(
    "kwargs", [{}, {"window": 0.0}, {"window": 0.2}, {"window": 0.5}, {"window": 1.0}]
)
def test_cython_matches_numba(d, m, kwargs):
    """Cython DTW must match sktime's numba implementation (groundtruth).

    Covers the bounding modes whose masks are bit-identical to numba's: no
    bounding at any lengths, and Sakoe-Chiba at equal lengths (that band is
    symmetric, so numba's missing transpose is invisible). Itakura is covered
    by ``test_cython_matches_numba_with_shared_mask``.
    """
    x, y = _series(11, d=d, m=m), _series(12, d=d, m=m)

    expected = _numba_dtw_distance(x, y, **kwargs)
    got = dtw_distance(x, y, **kwargs)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


@needs_sktime
@pytest.mark.parametrize("m1,m2", UNEQUAL_LENGTHS)
@pytest.mark.parametrize("d", [1, 3])
def test_cython_matches_numba_unequal_lengths(m1, m2, d):
    """Unbounded DTW matches numba in both length orientations."""
    x, y = _series(13, d=d, m=m1), _series(14, d=d, m=m2)

    np.testing.assert_allclose(
        dtw_distance(x, y), _numba_dtw_distance(x, y), rtol=1e-9, atol=1e-9
    )


@needs_sktime
@pytest.mark.parametrize("m1,m2", [(20, 20), *UNEQUAL_LENGTHS])
@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize(
    "mask",
    ["sakoe_chiba", "itakura", "random_sparse", "random_dense"],
)
def test_cython_matches_numba_with_shared_mask(m1, m2, d, mask):
    """The kernel matches numba cell-for-cell given the same bounding matrix.

    Passing the mask explicitly to both implementations isolates the recurrence
    from the mask builders, which is what lets Itakura and unequal-length
    Sakoe-Chiba be compared at all: numba's ``lower_bounding`` would build a
    transposed (and for unequal lengths wrongly shaped) mask of its own.
    """
    x, y = _series(15, d=d, m=m1), _series(16, d=d, m=m2)
    _x, _y = _to_timeseries(x), _to_timeseries(y)
    if mask == "sakoe_chiba":
        bm = _sakoe_chiba(_x, _y, 0.2)
    elif mask == "itakura":
        bm = _itakura_parallelogram(_x, _y, 0.5)
    elif mask == "random_sparse":
        bm = _random_mask(m1, m2, seed=17, density=0.25)
    else:
        bm = _random_mask(m1, m2, seed=18, density=0.9)

    expected = _numba_dtw_distance(x, y, bounding_matrix=bm)
    got = dtw_distance(x, y, bounding_matrix=bm)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


@needs_sktime
@pytest.mark.parametrize("m1,m2", [(20, 20), *UNEQUAL_LENGTHS])
@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("mask", ["none", "sakoe_chiba", "itakura", "random"])
def test_cost_matrix_matches_numba(m1, m2, d, mask):
    """The full cost matrix matches numba's, inf cells included."""
    x, y = _series(19, d=d, m=m1), _series(20, d=d, m=m2)
    _x, _y = _to_timeseries(x), _to_timeseries(y)
    if mask == "none":
        bm = np.zeros((m1, m2))
    elif mask == "sakoe_chiba":
        bm = _sakoe_chiba(_x, _y, 0.3)
    elif mask == "itakura":
        bm = _itakura_parallelogram(_x, _y, 0.5)
    else:
        bm = _random_mask(m1, m2, seed=21, density=0.4)

    expected = _numba_cost_matrix(x, y, bm)
    got = dtw_cost_matrix(x, y, bounding_matrix=bm)
    assert got.shape == expected.shape == (m1, m2)
    # assert_allclose treats inf == inf as equal, so masked cells are compared too
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


@needs_sktime
@pytest.mark.parametrize("m", [10, 31, 50])
@pytest.mark.parametrize("window", [0.0, 0.2, 0.5, 1.0])
def test_sakoe_chiba_mask_matches_numba(m, window):
    """The Sakoe-Chiba mask itself is bit-identical to numba's at equal lengths."""
    from sktime.dists_kernels._numba_distances._lower_bounding_numba import sakoe_chiba

    x, y = _to_timeseries(_series(21, m=m)), _to_timeseries(_series(22, m=m))
    np.testing.assert_array_equal(_sakoe_chiba(x, y, window), sakoe_chiba(x, y, window))


@needs_sktime
@pytest.mark.parametrize("m1,m2", [(20, 20), (7, 19), (19, 7)])
@pytest.mark.parametrize("slope", [0.1, 0.5, 1.0])
def test_itakura_mask_is_numbas_transposed(m1, m2, slope):
    """Documents the deliberate divergence from numba's Itakura mask.

    numba's ``itakura_parallelogram`` skips the transpose out of the
    ``[y, x]``-indexed builder, so its mask is the mirror of the correct band
    (and is shaped ``(m2, m1)`` for unequal lengths, which the DTW kernel then
    indexes out of bounds). This port transposes it.
    """
    from sktime.dists_kernels._numba_distances._lower_bounding_numba import (
        itakura_parallelogram,
    )

    x, y = _to_timeseries(_series(23, m=m1)), _to_timeseries(_series(24, m=m2))
    ours = _itakura_parallelogram(x, y, slope)
    theirs = itakura_parallelogram(x, y, slope)
    assert ours.shape == (m1, m2)
    assert theirs.shape == (m2, m1)
    np.testing.assert_array_equal(ours, theirs.T)
