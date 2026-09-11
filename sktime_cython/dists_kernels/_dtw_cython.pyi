"""Type stubs for the compiled DTW Cython kernels."""

import numpy as np
from numpy.typing import NDArray

def cost_matrix(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bounding_matrix: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def distance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bounding_matrix: NDArray[np.float64],
) -> float: ...
