"""Regression tests for the single brentq wrapper.

Three legacy brentq blocks were collapsed into `tilting._solvers.brentq_with_doubling`;
these tests pin the contract: bracket doubling, direction handling, and
explicit failure when no root is bracketable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from frasian.tilting._solvers import BracketingFailed, brentq_with_doubling


@pytest.mark.L0
class TestBrentqWithDoubling:
    def test_finds_root_in_initial_bracket(self):
        # f(x) = x - 2; root at x=2, initial bracket [0, 4].
        x = brentq_with_doubling(
            lambda x: x - 2.0, midpoint=0.0, initial_half_width=4.0,
            direction=+1,
        )
        assert math.isclose(x, 2.0, abs_tol=1e-9)

    def test_doubles_when_initial_too_small(self):
        # Root is at x=10 but initial half-width is 1.
        x = brentq_with_doubling(
            lambda x: x - 10.0, midpoint=0.0, initial_half_width=1.0,
            direction=+1,
        )
        assert math.isclose(x, 10.0, abs_tol=1e-9)

    def test_negative_direction(self):
        # Root at -5; midpoint 0, search downward.
        x = brentq_with_doubling(
            lambda x: x + 5.0, midpoint=0.0, initial_half_width=1.0,
            direction=-1,
        )
        assert math.isclose(x, -5.0, abs_tol=1e-9)

    def test_raises_when_unbracketable(self):
        with pytest.raises(BracketingFailed):
            brentq_with_doubling(
                lambda x: x ** 2 + 1.0,  # no real root
                midpoint=0.0, initial_half_width=1.0, direction=+1,
                max_doublings=4,
            )

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            brentq_with_doubling(
                lambda x: x, midpoint=0.0, initial_half_width=1.0, direction=0,
            )

    def test_invalid_half_width_raises(self):
        with pytest.raises(ValueError):
            brentq_with_doubling(
                lambda x: x, midpoint=0.0, initial_half_width=0.0,
                direction=+1,
            )
