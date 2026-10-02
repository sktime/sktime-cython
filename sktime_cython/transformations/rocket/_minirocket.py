"""Numba-free MiniRocket transform (pure-numpy + Cython core).

Handles univariate and multivariate panels with one code path: a univariate
panel is the single-channel case, matching both ``MiniRocket`` and
``MiniRocketMultivariate`` in sktime.

Compute layer with no sktime dependency: numpy arrays in, numpy arrays out.
``fit`` returns a parameter tuple; ``transform`` applies it. An sktime
``BaseTransformer`` wrapper delegates to these two functions.
"""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sktime_cython.transformations.rocket import _minirocket_multivariate_cython as _cy
from sktime_cython.transformations.rocket._common import (
    _NUM_KERNELS,
    _biases_from_C,
    _check_dilations,
    _fit_dilations,
    _quantiles,
)

__all__ = ["rocket_fit", "rocket_transform"]


def _as_panel(X):
    """Return X as contiguous 3D float32/float64; 2D is a univariate panel."""
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[:, None, :]
    elif X.ndim != 3:
        raise ValueError(f"X must be 2D or 3D, but found {X.ndim} dimensions")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one instance")
    dtype = X.dtype if X.dtype in (np.float32, np.float64) else np.float64
    return np.ascontiguousarray(X, dtype=dtype)


def rocket_fit(X, num_kernels=10_000, max_dilations_per_kernel=32, random_state=None):
    """Fit dilations, channel selections, and biases.

    Parameters
    ----------
    X : np.ndarray, shape (n_instances, n_columns, n_timepoints)
        or (n_instances, n_timepoints) for univariate series. float32 and
        float64 are read without a copy, other dtypes are cast to float64;
        computation is in float32, as in sktime.
    num_kernels : int, default=10000
        number of kernels; rounded down to a multiple of 84 (min 84).
    max_dilations_per_kernel : int, default=32
        maximum number of dilations per kernel.
    random_state : int or None, default=None
        seed for reproducibility.

    Returns
    -------
    parameters : tuple of np.ndarray
        (num_channels_per_combination, channel_indices, dilations,
        num_features_per_dilation, biases), ready to pass to ``transform``.
    """
    if random_state is not None and not isinstance(random_state, (int, np.integer)):
        raise ValueError(
            f"random_state must be int or None, but found {type(random_state)}"
        )
    seed = np.int32(random_state) if random_state is not None else None

    X = _as_panel(X)
    n_instances, n_columns, n_timepoints = X.shape
    if n_timepoints < 9:
        raise ValueError(
            f"n_timepoints must be >= 9, but found {n_timepoints};"
            " zero pad shorter series so that n_timepoints == 9"
        )

    if seed is not None:
        np.random.seed(seed)

    num_kernels_ = _NUM_KERNELS
    dilations, num_features_per_dilation = _fit_dilations(
        n_timepoints, num_kernels, max_dilations_per_kernel
    )
    num_features_per_kernel = np.sum(num_features_per_dilation)
    quantiles = _quantiles(num_kernels_ * num_features_per_kernel)

    num_dilations = len(dilations)
    num_combinations = num_kernels_ * num_dilations

    max_num_channels = min(n_columns, 9)
    max_exponent = np.log2(max_num_channels + 1)

    num_channels_per_combination = (
        2 ** np.random.uniform(0, max_exponent, num_combinations)
    ).astype(np.int32)

    channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)
    num_channels_start = 0
    for combination_index in range(num_combinations):
        n_this = num_channels_per_combination[combination_index]
        num_channels_end = num_channels_start + n_this
        channel_indices[num_channels_start:num_channels_end] = np.random.choice(
            n_columns, n_this, replace=False
        )
        num_channels_start = num_channels_end

    # biases: re-seed (matching numba _fit_biases_multi), draw one instance
    # index per combination, build C in Cython, then quantile per combination.
    if seed is not None:
        np.random.seed(seed)
    instance_indices = np.array(
        [np.random.randint(n_instances) for _ in range(num_combinations)],
        dtype=np.int32,
    )
    C = _cy.fit_biases(
        X,
        num_channels_per_combination,
        channel_indices,
        dilations.astype(np.int32),
        num_features_per_dilation.astype(np.int32),
        instance_indices,
    )

    biases = _biases_from_C(C, quantiles, num_features_per_dilation, num_kernels_)

    return (
        num_channels_per_combination,
        channel_indices,
        dilations.astype(np.int32),
        num_features_per_dilation.astype(np.int32),
        biases,
    )


def rocket_transform(X, parameters, n_jobs=1):
    """Apply a fitted MiniRocket transform.

    Parameters
    ----------
    X : np.ndarray, shape (n_instances, n_columns, n_timepoints)
        or (n_instances, n_timepoints) for univariate series. float32 and
        float64 are read without a copy, other dtypes are cast to float64;
        computation is in float32, as in sktime.
    parameters : tuple
        the tuple returned by ``fit``.
    n_jobs : int, default=1
        threads for the GIL-releasing Cython kernel over disjoint instance
        chunks. ``-1`` (or out of range) uses all processors.

    Returns
    -------
    np.ndarray, shape (n_instances, n_features), float32
    """
    X = _as_panel(X)
    _check_dilations(parameters[2], X.shape[2], "dilations")
    n_instances = X.shape[0]

    if n_jobs < 1 or n_jobs > multiprocessing.cpu_count():
        n_jobs = multiprocessing.cpu_count()
    n_jobs = min(n_jobs, n_instances)

    if n_jobs <= 1:
        return _cy.transform(X, *parameters)

    # the Cython kernel releases the GIL, so plain threads run truly in
    # parallel across disjoint instance chunks.
    bounds = np.linspace(0, n_instances, n_jobs + 1).astype(int)
    chunks = [
        np.ascontiguousarray(X[bounds[i] : bounds[i + 1]])
        for i in range(n_jobs)
        if bounds[i + 1] > bounds[i]
    ]
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        parts = list(ex.map(lambda c: _cy.transform(c, *parameters), chunks))
    return np.vstack(parts)
