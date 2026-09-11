"""Setup script for sktime-cython Cython extensions."""

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# fast-math is the bulk of the speedup vs numba; flags are platform-specific.
if sys.platform == "win32":
    _fast = ["/O2", "/fp:fast"]
    _strict = ["/O2", "/fp:precise"]
else:
    _fast = ["-O3", "-ffast-math"]
    _strict = ["-O3"]

# The DTW recurrence blocks out-of-band cells with `inf` and relies on it
# propagating through `+` and `fmin`. fast-math lets the compiler assume no
# infinities (clang warns "use of infinity via a macro is undefined behavior"),
# so the DTW extension opts out; it is min/add-bound and gains little from it
# anyway. MiniRocket is pure finite arithmetic and keeps fast-math.

extensions = [
    Extension(
        "sktime_cython.transformations.rocket._minirocket_multivariate_cython",
        sources=[
            "sktime_cython/transformations/rocket/_minirocket_multivariate_cython.pyx"
        ],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=_fast,
    ),
    Extension(
        "sktime_cython.dists_kernels._dtw_cython",
        sources=["sktime_cython/dists_kernels/_dtw_cython.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=_strict,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
