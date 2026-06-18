"""Tests for the example Cython extension module."""

import pytest


def test_add():
    """Test the Cython-compiled add function."""
    from sktime_cython._cython.example import add

    assert add(1, 2) == 3
    assert add(0, 0) == 0
    assert add(-1, 1) == 0


def test_multiply():
    """Test the Cython-compiled multiply function."""
    from sktime_cython._cython.example import multiply

    assert multiply(2.0, 3.0) == pytest.approx(6.0)
    assert multiply(0.0, 5.0) == pytest.approx(0.0)
    assert multiply(-1.0, 4.0) == pytest.approx(-4.0)
