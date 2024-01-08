"""Unit tests for the `yotse.optimization.fitting` package's `FuncFit` class."""
import unittest
from typing import Any

import numpy as np

from yotse.optimization.fitting import FuncFit


class TestFitting(unittest.TestCase):
    @staticmethod
    def dummy_func(x: Any) -> Any:
        """Dummy function."""
        # func = x*x*x - 1.0/x
        func = x * x
        func[func == -np.inf] = 0
        return func

    def test_best_fit(self) -> None:
        """Test that the fitting methods work."""

        # Fit polynomial functions
        ff = FuncFit()

        x = np.arange(-3, 3, 0.1, dtype=int)
        y = [self.dummy_func(x)]

        all_fits = ff.find_poly_fit(x, y)
        best_fit = all_fits[0][0]

        order = best_fit.order
        # error = all_fits[0][1]

        # test output types
        self.assertIsInstance(order, int)
        self.assertIsInstance(all_fits, list)
        self.assertEqual(len(all_fits), 41)
        # test correct value
        self.assertEqual(order, 3)
