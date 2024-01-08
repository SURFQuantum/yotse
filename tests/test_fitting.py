import unittest
from typing import List
from typing import Tuple
from unittest.mock import Mock
import numpy as np

from yotse.optimization.fitting import FuncFit


class TestFitting(unittest.TestCase):

    def dummy_func(self, x):
        # func = x*x*x - 1.0/x
        func = x * x
        func[func == -np.inf] = 0
        return func

    def test_best_fit(self) -> None:
        """Test that the fitting methods work."""

        # Fit polynomial functions
        ff = FuncFit()

        x = np.arange(-3, 3, 0.1, dtype=int)
        y = dummy_func(x)

        all_fits = ff.find_poly_fit(x, y)
        best_fit = all_fits[0][0]

        order = best_fit.order
        error = all_fits[0][1]

        # test output types
        self.assertIsInstance(order, int)
        self.assertIsInstance(all_fits, list)
        self.assertEqual(len(all_fits), 41)
        # test correct value
        self.assertEqual(order, 3)
