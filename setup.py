"""Setup script for sktime-cython Cython extensions."""

from setuptools import Extension, setup

from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "sktime_cython._cython.example",
        sources=["sktime_cython/_cython/example.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
