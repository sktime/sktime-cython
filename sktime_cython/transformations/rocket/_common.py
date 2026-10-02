"""Scaffolding shared by the MiniRocket and MultiRocket transforms.

The dilation schedule and quantile points are identical for both transforms;
the per-transform feature accounting happens in the callers. Pure numpy: these
run once per fit on small arrays, so there is nothing here worth compiling.
"""

import numpy as np

_NUM_KERNELS = 84


def _fit_dilations(n_timepoints, num_features, max_dilations_per_kernel):
    """Dilation schedule (pure numpy, copied from sktime numba module)."""
    num_kernels = _NUM_KERNELS
    if num_features < num_kernels:
        num_features = num_kernels

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel
    )
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((n_timepoints - 1) / (9 - 1))
    dilations, num_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
            np.int32
        ),
        return_counts=True,
    )
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
        np.int32
    )

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


def _quantiles(n):
    """Evenly-spaced low-discrepancy quantile points (copied from sktime)."""
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )


def _biases_from_C(C, quantiles, num_features_per_dilation, num_kernels):
    """Take the per-combination quantiles of C, vectorized.

    Equivalent to looping over combinations in dilation-major order and
    calling np.quantile on each row with its own slice of quantiles.

    Parameters
    ----------
    C : per-combination convolution outputs, one row per combination
    quantiles : one quantile per output feature, in feature order
    num_features_per_dilation : number of features per dilation
    num_kernels : number of kernels per dilation

    Returns
    -------
    the bias per output feature
    """
    features_per_combination = np.repeat(num_features_per_dilation, num_kernels)
    combination = np.repeat(
        np.arange(len(features_per_combination)), features_per_combination
    )
    # numba's np.quantile, op for op (numba/np/arraymath.py): it promotes both
    # operands to float64, scales the quantile to a percentile and back, and
    # interpolates between ranks f-1 and f. Matching it exactly matters because
    # a bias landing on a repeated convolution value would otherwise
    # reclassify a whole tied group of samples.
    S = np.sort(C, axis=1).astype(np.float64)
    n_timepoints = S.shape[1]
    percentile = quantiles.astype(np.float64) * 100.0
    rank = 1.0 + (n_timepoints - 1) * (percentile / 100.0)
    floor = np.floor(rank)
    weight = rank - floor
    lower = (floor - 1.0).astype(np.intp)
    upper = np.minimum(lower + 1, n_timepoints - 1)
    return (
        S[combination, lower] * (1 - weight) + S[combination, upper] * weight
    ).astype(np.float32)


def _check_dilations(dilations, n_timepoints, name):
    """Check the in-bounds invariants the kernel relies on.

    The Cython kernel runs with bounds checking off and integer division
    unguarded, so a hand-built parameter tuple could otherwise corrupt memory
    or divide by zero. ``_fit_dilations`` guarantees both conditions.

    Parameters
    ----------
    dilations : the dilations for one representation
    n_timepoints : length of the series that representation is applied to
    name : parameter name, for the error message

    Raises
    ------
    ValueError
        if a dilation is below 1, or too large for the series length
    """
    if dilations.min() < 1:
        raise ValueError(f"{name} must be >= 1, but found {dilations.min()}")
    if 8 * dilations.max() >= n_timepoints:
        raise ValueError(
            f"{name} must satisfy 8 * dilation < n_timepoints={n_timepoints}, "
            f"but found {dilations.max()}"
        )
