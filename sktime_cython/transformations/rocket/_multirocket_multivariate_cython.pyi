"""Type stubs for the compiled MultiRocketMultivariate Cython kernels."""

import numpy as np
from numpy.typing import NDArray

_Real = NDArray[np.float32] | NDArray[np.float64]

def transform(
    X: _Real,
    X1: _Real,
    num_channels_per_combination: NDArray[np.int32],
    channel_indices: NDArray[np.int32],
    dilations: NDArray[np.int32],
    num_features_per_dilation: NDArray[np.int32],
    biases: NDArray[np.float32],
    num_channels_per_combination1: NDArray[np.int32],
    channel_indices1: NDArray[np.int32],
    dilations1: NDArray[np.int32],
    num_features_per_dilation1: NDArray[np.int32],
    biases1: NDArray[np.float32],
    num_features_per_kernel: int,
    original_implementation: bool,
) -> NDArray[np.float32]: ...
def fit_biases(
    X: _Real,
    num_channels_per_combination: NDArray[np.int32],
    channel_indices: NDArray[np.int32],
    dilations: NDArray[np.int32],
    num_features_per_dilation: NDArray[np.int32],
    instance_indices: NDArray[np.int32],
) -> NDArray[np.float32]: ...
