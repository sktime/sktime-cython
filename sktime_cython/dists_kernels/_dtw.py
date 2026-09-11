"""Numba-free dynamic time warping (pure-numpy scaffolding + Cython core).

Compute layer with no sktime dependency: numpy arrays in, scalar / numpy arrays
out. The hot dynamic-programming recurrence lives in the compiled Cython kernel
``_dtw_cython``; everything here (input reshaping and Sakoe-Chiba / Itakura
bounding-matrix construction) is pure numpy, ported from sktime so the bounding
masks are bit-identical and the results match sktime's numba DTW exactly, with
one deliberate exception.

That exception: the ported mask builders transpose their output into the
``(m1, m2)`` layout the kernels index (see ``_to_xy_orientation``). numba's
``lower_bounding`` does not, which is harmless for Sakoe-Chiba on equal-length
series but mirrors the Itakura parallelogram and produces an out-of-bounds mask
shape for unequal-length series. So ``itakura_max_slope`` results differ from
numba's; every other mode stays bit-identical.

Ports (BSD-3-Clause, authors chrisholder and TonyBagnall):
* ``_to_timeseries`` <- ``sktime...._numba_utils.to_numba_timeseries``
* ``_resolve_bounding_matrix`` <- ``sktime...lower_bounding.resolve_bounding_matrix``
* ``_sakoe_chiba`` / ``_itakura_parallelogram`` / ``_create_shape_on_matrix`` /
  ``_check_line_steps`` <- ``sktime...._lower_bounding_numba``

Algorithmic references
----------------------
Sakoe H. and Chiba S., "Dynamic programming algorithm optimization for spoken
word recognition," IEEE Trans. ASSP 26(1):43-49, 1978.
Itakura F., "Minimum prediction residual principle applied to speech
recognition," IEEE Trans. ASSP 23(1):67-72, 1975.
Shokoohi-Yekta M. et al., "Generalizing DTW to the multi-dimensional case
requires an adaptive approach," DMKD 31:1-31, 2017 (dependent DTW_D).
"""

import math

import numpy as np

from sktime_cython.dists_kernels import _dtw_cython as _cy

__all__ = ["dtw_distance", "dtw_cost_matrix"]


def _to_timeseries(x):
    """Reshape a 1d/2d series to channels-first ``(d, m)`` float64.

    Port of ``to_numba_timeseries``: a 1d array or a column vector ``(m, 1)``
    (with ``m != 1``) becomes a single-channel ``(1, m)`` series; a ``(d, m)``
    array is kept as-is.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("The input time series must be a numpy array.")
    _x = np.array(x, copy=True, dtype=np.float64)
    if _x.ndim == 1 or (_x.ndim == 2 and _x.shape[1] == 1 and _x.shape[0] != 1):
        _x = np.reshape(_x, (1, _x.shape[0]))
    elif _x.ndim > 2:
        raise ValueError(
            "The matrix provided has more than 2 dimensions. This is not supported."
        )
    return np.ascontiguousarray(_x, dtype=np.float64)


def _check_series_pair(x, y):
    """Reshape both inputs and check they have the same number of channels.

    The dependent local cost loops over ``x``'s channels and reads ``y[k, j]``
    for each one with bounds checking disabled, so a mismatch would read past
    the end of ``y``.
    """
    _x = _to_timeseries(x)
    _y = _to_timeseries(y)
    if _x.shape[0] != _y.shape[0]:
        raise ValueError(
            "The two time series must have the same number of channels, but "
            f"x has {_x.shape[0]} and y has {_y.shape[0]}."
        )
    return _x, _y


def _check_line_steps(line):
    """Clamp each step of a line to +/- 1 of the previous value."""
    prev = line[0]
    for i in range(1, len(line)):
        curr_val = line[i]
        if curr_val > (prev + 1):
            line[i] = prev + 1
        elif curr_val < (prev - 1):
            line[i] = prev - 1
        prev = curr_val
    return line


def _create_shape_on_matrix(bounding_matrix, y_upper_line, y_lower_line=None):
    """Set the band between upper and lower lines to 0.0 (in-bound).

    ``bounding_matrix`` is indexed ``[y_index, x_index]`` here, i.e. it must be
    shaped ``(m2, m1)``: the lines are sampled once per ``x`` index (columns)
    and give the ``y`` range (rows) reachable from it. Callers transpose the
    result, since the kernels index the mask as ``bm[i_x, j_y]``.
    """
    y_size = bounding_matrix.shape[0]
    if y_lower_line is None:
        y_lower_line = y_upper_line
    if y_upper_line.shape[0] != y_lower_line.shape[0]:
        raise ValueError(
            "The number of upper line values must equal the number of lower line values"
        )
    for i in range(y_upper_line.shape[0]):
        x = i  # x_step_size = 1, start_val = 0
        upper_y = max(0, min(y_size - 1, math.ceil(y_upper_line[i])))
        lower_y = max(0, min(y_size - 1, math.floor(y_lower_line[i])))
        if upper_y == lower_y:
            bounding_matrix[upper_y, x] = 0.0
        else:
            bounding_matrix[upper_y : (lower_y + 1), x] = 0.0
    return bounding_matrix


def _to_xy_orientation(mask):
    """Transpose a ``[y, x]``-indexed mask into the kernels' ``(m1, m2)`` layout.

    ``_create_shape_on_matrix`` writes ``mask[y_index, x_index]``, so the band it
    builds is shaped ``(m2, m1)``. The kernels read ``bm[i, j]`` with ``i`` over
    ``x`` and ``j`` over ``y``, so the mask has to be transposed on the way out.

    sktime's numba ``lower_bounding`` skips this transpose, which is only
    invisible for Sakoe-Chiba on equal-length series (that band is symmetric).
    For the Itakura parallelogram it mirrors the band, and for unequal lengths
    it yields a mask the DTW kernel indexes out of bounds. This port therefore
    deliberately diverges from numba's ``itakura_parallelogram`` output.
    """
    return np.ascontiguousarray(mask.T)


def _no_bounding(x, y):
    """All-zero (fully in-bound) mask of shape ``(m1, m2)``."""
    return np.zeros((x.shape[1], y.shape[1]))


def _sakoe_chiba(x, y, window):
    """Sakoe-Chiba band mask of shape ``(m1, m2)``.

    ``window`` in [0, 1] is the fractional radius.
    """
    if window < 0 or window > 1:
        raise ValueError("Window must between 0 and 1")
    x_size = x.shape[1]
    y_size = y.shape[1]
    bounding_matrix = np.full((y_size, x_size), np.inf)
    radius = ((x_size / 100) * window) * 100

    upper = np.interp(
        list(range(x_size)),
        [0, x_size - 1],
        [0 - radius, y_size - radius - 1],
    )
    lower = np.interp(
        list(range(x_size)),
        [0, x_size - 1],
        [0 + radius, y_size + radius - 1],
    )
    return _to_xy_orientation(_create_shape_on_matrix(bounding_matrix, upper, lower))


def _itakura_parallelogram(x, y, itakura_max_slope):
    """Itakura parallelogram mask of shape ``(m1, m2)``.

    ``itakura_max_slope`` in [0, 1].
    """
    if itakura_max_slope < 0 or itakura_max_slope > 1:
        raise ValueError("Window must between 0 and 1")
    x_size = x.shape[1]
    y_size = y.shape[1]
    bounding_matrix = np.full((y_size, x_size), np.inf)
    itakura_max_slope = math.floor(((x_size / 100) * itakura_max_slope) * 100) / 2

    middle_x_upper = math.ceil(x_size / 2)
    middle_x_lower = math.floor(x_size / 2)
    if middle_x_lower == middle_x_upper:
        middle_x_lower = middle_x_lower - 1
    middle_y = math.floor(y_size / 2)

    diff = abs((middle_x_lower * itakura_max_slope) - middle_y)
    middle_y_lower = middle_y + diff
    middle_y_upper = middle_y - diff

    upper = np.interp(
        list(range(x_size)),
        [0, middle_x_lower, middle_x_upper, x_size - 1],
        [0, middle_y_upper, middle_y_upper, y_size - 1],
    )
    lower = np.interp(
        list(range(x_size)),
        [0, middle_x_lower, middle_x_upper, x_size - 1],
        [0, middle_y_lower, middle_y_lower, y_size - 1],
    )
    if np.array_equal(upper, lower):
        upper = _check_line_steps(upper)
    return _to_xy_orientation(_create_shape_on_matrix(bounding_matrix, upper, lower))


def _resolve_bounding_matrix(
    x, y, window=None, itakura_max_slope=None, bounding_matrix=None
):
    """Pick / build the bounding mask (finite = in-bound, inf = out-of-bound).

    The returned mask is always ``(m1, m2)`` contiguous float64: the Cython
    kernels read ``bm[i, j]`` for every ``i < m1``, ``j < m2`` with bounds
    checking disabled, so any other shape is rejected here rather than read
    out of bounds.
    """
    expected = (x.shape[1], y.shape[1])
    if bounding_matrix is not None:
        bm = np.ascontiguousarray(bounding_matrix, dtype=np.float64)
        if bm.shape != expected:
            raise ValueError(
                f"The bounding matrix must have shape {expected} "
                f"(len(x), len(y)), but has shape {bm.shape}."
            )
        return bm
    if window is not None and itakura_max_slope is not None:
        raise ValueError(
            "You can only use one bounding matrix at once. You have set both "
            "window and itakura_max_slope parameter."
        )
    if window is not None:
        if not isinstance(window, float):
            raise ValueError("The sakoe chiba window must be a float.")
        bm = _sakoe_chiba(x, y, window)
    elif itakura_max_slope is not None:
        if not isinstance(itakura_max_slope, float):
            raise ValueError("The itakura max slope must be a float.")
        bm = _itakura_parallelogram(x, y, itakura_max_slope)
    else:
        bm = _no_bounding(x, y)
    return np.ascontiguousarray(bm, dtype=np.float64)


def dtw_cost_matrix(x, y, window=None, itakura_max_slope=None, bounding_matrix=None):
    """Full DTW cost matrix between two series.

    Parameters
    ----------
    x, y : np.ndarray, 1d ``(m,)`` or 2d ``(d, m)`` channels-first
        Input series; cast to contiguous float64 internally.
    window : float or None, default=None
        Sakoe-Chiba band radius as a fraction in [0, 1].
    itakura_max_slope : float or None, default=None
        Itakura parallelogram max slope in [0, 1]. Mutually exclusive
        with ``window``. Results differ from sktime's numba DTW, whose
        Itakura mask is transposed; see the module docstring.
    bounding_matrix : np.ndarray or None, default=None
        Custom ``(m1, m2)`` mask (finite = in-bound, inf = out-of-bound).
        Overrides ``window`` / ``itakura_max_slope`` when given.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different channel counts, or if
        ``bounding_matrix`` is given and is not ``(m1, m2)``.

    Returns
    -------
    np.ndarray, shape ``(m1, m2)``, float64
        Cost matrix; its bottom-right entry is the DTW distance.
    """
    _x, _y = _check_series_pair(x, y)
    bm = _resolve_bounding_matrix(_x, _y, window, itakura_max_slope, bounding_matrix)
    return _cy.cost_matrix(_x, _y, bm)


def dtw_distance(x, y, window=None, itakura_max_slope=None, bounding_matrix=None):
    """Dependent multivariate DTW distance (squared-Euclidean local cost).

    Parameters
    ----------
    x, y : np.ndarray, 1d ``(m,)`` or 2d ``(d, m)`` channels-first
        Input series; cast to contiguous float64 internally.
    window : float or None, default=None
        Sakoe-Chiba band radius as a fraction in [0, 1].
    itakura_max_slope : float or None, default=None
        Itakura parallelogram max slope in [0, 1]. Mutually exclusive
        with ``window``. Results differ from sktime's numba DTW, whose
        Itakura mask is transposed; see the module docstring.
    bounding_matrix : np.ndarray or None, default=None
        Custom ``(m1, m2)`` mask (finite = in-bound, inf = out-of-bound).
        Overrides ``window`` / ``itakura_max_slope`` when given.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different channel counts, or if
        ``bounding_matrix`` is given and is not ``(m1, m2)``.

    Returns
    -------
    float
        DTW distance between ``x`` and ``y``. Not square-rooted, matching
        sktime's convention (sum of squared local costs along the path).
    """
    _x, _y = _check_series_pair(x, y)
    bm = _resolve_bounding_matrix(_x, _y, window, itakura_max_slope, bounding_matrix)
    return float(_cy.distance(_x, _y, bm))
