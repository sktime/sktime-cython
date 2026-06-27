"""sktime-cython - Cython extensions for sktime.

Each estimator lives in its own submodule and exposes its own compute API,
e.g. ``from sktime_cython.minirocket import fit, transform``. Nothing is
re-exported at the top level, so estimators never collide on common names.
"""
