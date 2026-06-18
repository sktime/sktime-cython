# cython: language_level=3
"""Example Cython extension module demonstrating basic Cython usage."""


def add(int x, int y):
    """Add two integers using Cython-compiled code.

    Parameters
    ----------
    x : int
        First integer.
    y : int
        Second integer.

    Returns
    -------
    int
        Sum of x and y.
    """
    return x + y


def multiply(double x, double y):
    """Multiply two doubles using Cython-compiled code.

    Parameters
    ----------
    x : double
        First number.
    y : double
        Second number.

    Returns
    -------
    double
        Product of x and y.
    """
    return x * y
