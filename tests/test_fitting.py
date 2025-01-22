"""Unit tests for the `yotse.optimization.fitting` package's `FuncFit` class."""

import unittest
from typing import Any

import numpy as np

from yotse.optimization.fitting import FuncFit


class TestFitting(unittest.TestCase):
    """Validate FuncFit class methods."""

    @staticmethod
    def dummy_func_quadratic(x: Any) -> Any:
        """Dummy function."""
        func = x * x
        func[func == -np.inf] = 0
        return func

    @staticmethod
    def dummy_func_cubic(x: Any) -> Any:
        """Dummy cubic function."""
        return x * x * x

    @staticmethod
    def dummy_func_quartic(x: Any) -> Any:
        """Dummy quartic function."""
        return x * x * x * x

    @staticmethod
    def dummy_func_sinusoidal(x: Any) -> Any:
        """Dummy sinusoidal function."""
        return np.sin(x) + 0.5 * x

    def test_best_fit(self) -> None:
        """Test that the fitting methods work."""

        # Fit polynomial functions
        ff = FuncFit()

        # Test for Quadratic function (x^2)
        x_quadratic = np.arange(-3, 3, 0.1, dtype=float)
        y_quadratic = self.dummy_func_quadratic(x_quadratic)
        all_fits_quadratic = ff.find_poly_fit(list(x_quadratic), y_quadratic)
        best_fit_quadratic = all_fits_quadratic[0][0]
        order_quadratic = best_fit_quadratic.degree()

        # Test for Cubic function (x^3)
        x_cubic = np.arange(-3, 3, 0.1, dtype=float)
        y_cubic = self.dummy_func_cubic(x_cubic)
        all_fits_cubic = ff.find_poly_fit(list(x_cubic), y_cubic)
        best_fit_cubic = all_fits_cubic[0][0]
        order_cubic = best_fit_cubic.degree()

        # Test for Quartic function (x^4)
        x_quartic = np.arange(-3, 3, 0.1, dtype=float)
        y_quartic = self.dummy_func_quartic(x_quartic)
        all_fits_quartic = ff.find_poly_fit(list(x_quartic), y_quartic)
        best_fit_quartic = all_fits_quartic[0][0]
        order_quartic = best_fit_quartic.degree()

        # Test for Sinusoidal function (sin(x) + 0.5 * x)
        x_sinusoidal = np.arange(-3 * np.pi, 3 * np.pi, 0.1, dtype=float)
        y_sinusoidal = self.dummy_func_sinusoidal(x_sinusoidal)
        all_fits_sinusoidal = ff.find_poly_fit(list(x_sinusoidal), y_sinusoidal)
        best_fit_sinusoidal = all_fits_sinusoidal[0][0]
        order_sinusoidal = best_fit_sinusoidal.degree()

        # Test output types and correct values for each function
        self.assertIsInstance(order_quadratic, int)
        self.assertEqual(order_quadratic, 2)  # Expecting a quadratic function (x^2)

        self.assertIsInstance(order_cubic, int)
        self.assertEqual(order_cubic, 3)  # Expecting a cubic function (x^3)

        self.assertIsInstance(order_quartic, int)
        self.assertEqual(order_quartic, 4)  # Expecting a quartic function (x^4)

        self.assertIsInstance(order_sinusoidal, int)
        self.assertLessEqual(
            order_sinusoidal, 10
        )  # Adjust the expected order based on your function
