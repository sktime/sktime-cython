# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython dynamic time warping (DTW) kernels.

Ahead-of-time compiled port of sktime's numba ``_cost_matrix`` kernel
(``sktime.dists_kernels._numba_distances._dtw_numba._cost_matrix``). Same math,
no numba JIT warmup. Two entry points share one recurrence:

* ``cost_matrix`` fills the full dynamic-programming matrix (the reusable
  kernel; foundation for alignment-path recovery in later phases);
* ``distance`` computes only the final DTW cost via a rolling two-row buffer,
  avoiding the O(m1 * m2) allocation when the cost matrix is not needed.

Both use the "dependent" multivariate squared-Euclidean local cost
(DTW_D, Shokoohi-Yekta et al. 2017) and are numerically equivalent to the
numba implementation (verified against it as groundtruth in tests).

The ``bounding_matrix`` follows sktime's sentinel convention: an in-bound
cell holds a finite value (0.0) and an out-of-bound cell holds ``inf``.

Both kernels index ``x``, ``y`` and the bounding matrix with bounds checking
disabled, so both validate their argument shapes up front: ``x`` and ``y`` must
share a channel count and the bounding matrix must be exactly ``(m1, m2)``.
These checks run once per call, outside the ``nogil`` recurrence.

The recurrence blocks cells with ``INFINITY`` and relies on ``inf`` propagating
through ``+`` and ``fmin``, which is incompatible with finite-math assumptions;
``setup.py`` therefore compiles this extension without ``-ffast-math``.

References
----------
Port of the numba kernel by chrisholder and TonyBagnall in sktime
(BSD-3-Clause), ``sktime/dists_kernels/_numba_distances/_dtw_numba.py``.
Original DTW algorithm: Sakoe & Chiba, IEEE TASSP 26(1):43-49, 1978.
"""

import numpy as np

cimport numpy as cnp
from libc.math cimport INFINITY, fmin
cnp.import_array()

cdef inline double _min3(double a, double b, double c) noexcept nogil:
    return fmin(fmin(a, b), c)


cdef inline double _local_cost(
    double[:, ::1] x, double[:, ::1] y, Py_ssize_t i, Py_ssize_t j, Py_ssize_t d
) noexcept nogil:
    """Dependent multivariate squared-Euclidean cost between x[:, i] and y[:, j]."""
    cdef Py_ssize_t k
    cdef double diff, s = 0.0
    for k in range(d):
        diff = x[k, i] - y[k, j]
        s += diff * diff
    return s


cdef _check_shapes(
    cnp.ndarray x, cnp.ndarray y, cnp.ndarray bounding_matrix
):
    """Reject argument shapes the unchecked kernel loops would read past.

    ``_local_cost`` walks ``x``'s channels and reads ``y[k, j]`` for each, and
    the recurrence reads ``bounding_matrix[i, j]`` over the full ``(m1, m2)``
    grid. Both do so with ``boundscheck=False``, so the shapes are checked here
    instead. Callers in ``_dtw.py`` normally guarantee this; the check keeps the
    kernels safe when they are called directly.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "The two time series must have the same number of channels, but "
            f"x has {x.shape[0]} and y has {y.shape[0]}."
        )
    if bounding_matrix.shape[0] != x.shape[1] or bounding_matrix.shape[1] != y.shape[1]:
        raise ValueError(
            f"The bounding matrix must have shape ({x.shape[1]}, {y.shape[1]}) "
            f"(len(x), len(y)), but has shape "
            f"({bounding_matrix.shape[0]}, {bounding_matrix.shape[1]})."
        )


def cost_matrix(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] y,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] bounding_matrix,
):
    """Full DTW cost matrix.

    Port of ``_cost_matrix``. ``x`` and ``y`` are ``(d, m)`` channels-first
    series (possibly different lengths). Returns the ``(m1, m2)`` cost matrix
    (the numba code's ``cost_matrix[1:, 1:]`` slice); the DTW distance is its
    bottom-right entry.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different channel counts, or if
        ``bounding_matrix`` is not ``(m1, m2)``.
    """
    _check_shapes(x, y, bounding_matrix)
    cdef Py_ssize_t d = x.shape[0]
    cdef Py_ssize_t m1 = x.shape[1]
    cdef Py_ssize_t m2 = y.shape[1]
    cdef double[:, ::1] xv = x
    cdef double[:, ::1] yv = y
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] bm_mask_arr = np.asarray(
        np.isfinite(bounding_matrix), dtype=np.uint8
    )
    cdef unsigned char[:, ::1] bm_mask = bm_mask_arr

    # (m1 + 1, m2 + 1) padded with an inf border; [0, 0] = 0 seeds the path.
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] full = np.full(
        (m1 + 1, m2 + 1), INFINITY, dtype=np.float64
    )
    cdef double[:, ::1] c = full
    c[0, 0] = 0.0

    cdef Py_ssize_t i, j
    with nogil:
        for i in range(m1):
            for j in range(m2):
                if bm_mask[i, j]:
                    c[i + 1, j + 1] = _local_cost(xv, yv, i, j, d) + _min3(
                        c[i, j + 1], c[i + 1, j], c[i, j]
                    )

    return full[1:, 1:]


def distance(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] y,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] bounding_matrix,
):
    """DTW distance only, via a rolling two-row buffer.

    Numerically identical to ``cost_matrix(x, y, bm)[-1, -1]`` but uses
    ``O(m2)`` scratch instead of ``O(m1 * m2)``.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different channel counts, or if
        ``bounding_matrix`` is not ``(m1, m2)``.
    """
    _check_shapes(x, y, bounding_matrix)
    cdef Py_ssize_t d = x.shape[0]
    cdef Py_ssize_t m1 = x.shape[1]
    cdef Py_ssize_t m2 = y.shape[1]
    cdef double[:, ::1] xv = x
    cdef double[:, ::1] yv = y
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] bm_mask_arr = np.asarray(
        np.isfinite(bounding_matrix), dtype=np.uint8
    )
    cdef unsigned char[:, ::1] bm_mask = bm_mask_arr

    # prev = row i (padded), curr = row i + 1 (padded); length m2 + 1.
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] prev_a = np.full(
        m2 + 1, INFINITY, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] curr_a = np.empty(
        m2 + 1, dtype=np.float64
    )
    cdef double[::1] prev = prev_a
    cdef double[::1] curr = curr_a
    prev[0] = 0.0

    cdef Py_ssize_t i, j
    cdef double[::1] tmp
    cdef double result
    with nogil:
        for i in range(m1):
            curr[0] = INFINITY
            for j in range(m2):
                if bm_mask[i, j]:
                    curr[j + 1] = _local_cost(xv, yv, i, j, d) + _min3(
                        prev[j + 1], curr[j], prev[j]
                    )
                else:
                    curr[j + 1] = INFINITY
            tmp = prev
            prev = curr
            curr = tmp
        result = prev[m2]

    return result
