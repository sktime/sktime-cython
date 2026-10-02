"""Tests for the scaffolding shared by MiniRocket and MultiRocket.

Self-contained (numpy + pytest only) so they run in cibuildwheel's isolated
wheel-test env.
"""

import numpy as np
import pytest

from sktime_cython.transformations.rocket._common import (
    _biases_from_C,
    _fit_dilations,
    _quantiles,
)


def _biases_from_C_loop(C, quantiles, num_features_per_dilation, num_kernels):
    """Reference: numba's np.quantile, per combination, in a plain loop.

    numba/np/arraymath.py promotes to float64, scales the quantile to a
    percentile and back, and interpolates between ranks f-1 and f. Spelled out
    here so the vectorized version can be checked against something readable,
    without needing numba installed.
    """
    biases = np.zeros(
        num_kernels * int(np.sum(num_features_per_dilation)), dtype=np.float32
    )
    feature_index_start = 0
    combination_index = 0
    for dilation_index in range(len(num_features_per_dilation)):
        nfd = num_features_per_dilation[dilation_index]
        for _kernel_index in range(num_kernels):
            feature_index_end = feature_index_start + nfd
            a = np.sort(C[combination_index]).astype(np.float64)
            for i in range(feature_index_start, feature_index_end):
                percentile = np.float64(quantiles[i]) * 100.0
                rank = 1.0 + (len(a) - 1) * (percentile / 100.0)
                floor = np.floor(rank)
                weight = rank - floor
                lower = int(floor) - 1
                biases[i] = np.float32(
                    a[lower] * (1 - weight) + a[min(lower + 1, len(a) - 1)] * weight
                )
            feature_index_start = feature_index_end
            combination_index += 1
    return biases


@pytest.mark.parametrize("n_timepoints", [9, 60, 137])
@pytest.mark.parametrize(
    "num_features_per_dilation", [[1], [4, 2, 1], [32, 16, 8, 4, 2, 1, 1]]
)
def test_biases_from_C_matches_loop(num_features_per_dilation, n_timepoints):
    """Vectorized quantiles must match the per-combination loop exactly."""
    num_kernels = 84
    nfpd = np.array(num_features_per_dilation, dtype=np.int32)
    rng = np.random.RandomState(0)
    C = rng.normal(size=(num_kernels * len(nfpd), n_timepoints)).astype(np.float32)
    quantiles = _quantiles(num_kernels * int(nfpd.sum()))

    np.testing.assert_array_equal(
        _biases_from_C(C, quantiles, nfpd, num_kernels),
        _biases_from_C_loop(C, quantiles, nfpd, num_kernels),
    )


@pytest.mark.parametrize("num_features", [84, 168, 10_000])
def test_fit_dilations_features_add_up(num_features):
    """Features per dilation sum to the per-kernel budget, one per dilation."""
    dilations, nfpd = _fit_dilations(137, num_features, 32)
    assert len(dilations) == len(nfpd)
    assert nfpd.sum() == max(num_features, 84) // 84
