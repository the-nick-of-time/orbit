from math import sqrt

import pytest
from pytest import approx

from orbits.solvers import newtons_method


def test_newtons_method():
    zero_crossing = newtons_method(lambda x: x ** 2 - 2, lambda x: 2 * x, 2)
    assert zero_crossing == approx(sqrt(2), abs=1e-6)


def test_newtons_method_failure():
    with pytest.raises(ValueError):
        zero_crossing = newtons_method(lambda x: sqrt(x), lambda x: 1 / (2 * sqrt(x)), 2)
