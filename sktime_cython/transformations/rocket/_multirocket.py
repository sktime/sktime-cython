"""Numba-free MultiRocket multivariate transform (pure-numpy + Cython core).

Compute layer with no sktime dependency: numpy arrays in, numpy arrays out.
``fit`` returns a parameter tuple; ``transform`` applies it. An sktime
``BaseTransformer`` wrapper delegates to these two functions.

MultiRocket applies the MiniRocket kernels to both the raw series and its
first-order difference, and takes four features per kernel rather than one.
"""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sktime_cython.transformations.rocket import _multirocket_multivariate_cython as _cy
from sktime_cython.transformations.rocket._common import (
    _NUM_KERNELS,
    _biases_from_C,
    _check_dilations,
    _fit_dilations,
    _quantiles,
)

__all__ = ["multirocket_fit", "multirocket_transform"]


def _normalise(X):
    """Normalise each series to zero mean and unit variance."""
    return (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)


def _get_parameter(X, num_kernels, max_dilations_per_kernel, seed):
    """Fit dilations, channel selections, and biases for one representation."""
    if seed is not None:
        np.random.seed(seed)

    n_instances, n_columns, n_timepoints = X.shape
    num_kernels_ = _NUM_KERNELS

    dilations, num_features_per_dilation = _fit_dilations(
        n_timepoints, num_kernels, max_dilations_per_kernel
    )
    num_features_per_kernel = np.sum(num_features_per_dilation)
    quantiles = _quantiles(num_kernels_ * num_features_per_kernel)

    num_combinations = num_kernels_ * len(dilations)

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

    # biases: re-seed (matching numba _fit_biases), draw one instance index per
    # combination, build C in Cython, then quantile per combination.
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


def multirocket_fit(
    X,
    num_kernels=6_250,
    max_dilations_per_kernel=32,
    normalise=False,
    random_state=None,
):
    """Fit dilations, channel selections, and biases, on X and its difference.

    Parameters
    ----------
    X : 3D np.ndarray, shape (n_instances, n_columns, n_timepoints)
        panel of time series; normalised in its own dtype, then cast to float64
        (as in sktime).
    num_kernels : int, default=6250
        number of kernels; rounded down to a multiple of 84 (min 84).
    max_dilations_per_kernel : int, default=32
        maximum number of dilations per kernel.
    normalise : bool, default=False
        whether to normalise each series before fitting.
    random_state : int or None, default=None
        seed for reproducibility.

    Returns
    -------
    parameters : flattened tuple of tuple of np.ndarray
        the parameters for the raw and the differenced representation, ready
        to pass to ``transform``.
    """
    if random_state is not None and not isinstance(random_state, (int, np.integer)):
        raise ValueError(
            f"random_state must be int or None, but found {type(random_state)}"
        )
    seed = np.int32(random_state) if random_state is not None else None

    # sktime's order: normalise in the input dtype, pad, then cast to float64
    if normalise:
        X = _normalise(X)

    if X.shape[2] < 10:
        # handling very short series (like PensDigit from the MTSC archive)
        # series have to be at least a length of 10 (including differencing)
        padded = np.zeros((X.shape[0], X.shape[1], 10), dtype=X.dtype)
        padded[:, :, : X.shape[2]] = X
        X = padded
    X = np.ascontiguousarray(X, dtype=np.float64)

    args = (num_kernels, max_dilations_per_kernel, seed)
    return (
        _get_parameter(X, *args),
        _get_parameter(np.ascontiguousarray(np.diff(X, 1)), *args),
    )


def multirocket_transform(
    X,
    parameters,
    normalise=False,
    n_jobs=1,
    original_implementation=False,
):
    """Apply a fitted MultiRocket transform.

    Parameters
    ----------
    X : 3D np.ndarray, shape (n_instances, n_columns, n_timepoints)
        panel of time series, made contiguous; float32 is kept and computed in
        float32, other dtypes are cast to float64. sktime casts all dtypes to
        float64, so float32 results differ slightly from sktime's.
    parameters : tuple
        the tuple returned by ``fit``.
    normalise : bool, default=False
        whether to normalise each series before transforming; must match the
        value used in ``fit``.
    n_jobs : int, default=1
        threads for the GIL-releasing Cython kernel over disjoint instance
        chunks. ``-1`` (or out of range) uses all processors.
    original_implementation : bool, default=False
        whether to reproduce sktime's reference behaviour for the differenced
        pass rather than the corrected transform; see sktime#11291.

    Returns
    -------
    np.ndarray, shape (n_instances, n_features), float32
    """
    X = np.ascontiguousarray(
        X, dtype=np.float32 if X.dtype == np.float32 else np.float64
    )
    if normalise:
        X = _normalise(X)
    X1 = np.ascontiguousarray(np.diff(X, 1))

    _check_dilations(parameters[0][2], X.shape[2], "dilations")
    _check_dilations(parameters[1][2], X1.shape[2], "dilations1")

    args = (
        *parameters[0],
        *parameters[1],
        original_implementation,
    )
    n_instances = X.shape[0]

    if n_jobs < 1 or n_jobs > multiprocessing.cpu_count():
        n_jobs = multiprocessing.cpu_count()
    n_jobs = min(n_jobs, n_instances)

    # nan_to_num in place: the kernel's output is freshly allocated, and per
    # chunk it runs in the worker threads and avoids a second full copy.
    if n_jobs <= 1:
        return np.nan_to_num(_cy.transform(X, X1, *args), copy=False)

    # the Cython kernel releases the GIL, so plain threads run truly in
    # parallel across disjoint instance chunks.
    bounds = np.linspace(0, n_instances, n_jobs + 1).astype(int)
    chunks = [
        (
            np.ascontiguousarray(X[bounds[i] : bounds[i + 1]]),
            np.ascontiguousarray(X1[bounds[i] : bounds[i + 1]]),
        )
        for i in range(n_jobs)
        if bounds[i + 1] > bounds[i]
    ]
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        parts = list(
            ex.map(
                lambda c: np.nan_to_num(_cy.transform(*c, *args), copy=False), chunks
            )
        )
    return np.vstack(parts)
