"""
Module: my_func_fit

This module provides a class, `FuncFit`, for fitting polynomials to data and calculating errors and chi-square values.

Classes:
    - FuncFit: A class for polynomial fitting and error/chi-square calculation.

Functions:
    - add_noise: Static method to add noise to an array.
    - chi_square: Static method to calculate chi-square.
    - err: Static method to calculate error.
    - find_poly_fit: Method to find the best-fitting polynomial for given data.

"""
import warnings
from typing import Any
from typing import List
from typing import Tuple

import numpy as np


class FuncFit:
    """FuncFit class for fitting polynomials, calculating errors, and chi-square values.

    Methods:
        - add_noise: Static method to add noise to an array.
        - chi_square: Static method to calculate chi-square.
        - err: Static method to calculate error.
        - find_poly_fit: Method to find the best-fitting polynomial for given data.
    """

    def __init__(self) -> None:
        """Initialize FuncFit."""
        pass

    @staticmethod
    def add_noise(arr: np.ndarray, level: float = 0.2) -> np.ndarray:
        """Add noise to the given array.

        Parameters
        ----------
        arr : numpy.ndarray
            Array of values.
        level : float, optional
            Noise level. Default is 0.2.

        Returns
        -------
        numpy.ndarray
            Input array with added noise.
        """
        return np.random.normal(2 * arr + 2, level)

    @staticmethod
    def chi_square(
        poly: np.polynomial.Polynomial, x: List[float], y: List[float]
    ) -> Any:
        """Calculate chi-square.

        Parameters
        ----------
        poly : numpy.polynomial.Polynomial
            Polynomial (numpy.poly1d object).
        x : list of float
            X-points of the original dataset.
        y : list of float
            Y-points of the original dataset.

        Returns
        -------
        Any
            Chi-square.
        """
        x_values = np.asarray(x, dtype=float)
        return np.sum((poly(x_values) - y) ** 2) / len(x)

    @staticmethod
    def err(poly: np.poly1d, x: List[float], y: List[float]) -> Any:
        """Calculate error.

        Parameters
        ----------
        poly : numpy.poly1d
            Polynomial (numpy.poly1d object).
        x : list of float
            X-points of the original dataset.
        y : list of float
            Y-points of the original dataset.

        Returns
        -------
        Any
            Error.
        """
        return np.sqrt(np.sum(np.power(poly(x) - y, 2.0)))

    def find_poly_fit(
        self, x: List[float], y: List[float], max_order: int = 41
    ) -> List[Tuple[Any, Any]]:
        """Find a polynomial that fits the given data best.

        Parameters
        ----------
        x : list of float
            X-points.
        y : list of float
            Y-points.
        max_order : int, optional
            Maximum order of a polynomial. Default is 41.

        Returns
        -------
        list of tuple
            A list of tuples containing polynomials and their corresponding errors.
            The list is sorted in ascending order using the error values.
        """
        poly_fits = []

        # Fit a polynomial
        for n in range(1, max_order + 1):
            # Try to fit the polynomial
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    # Fit polynomial function
                    poly, residuals = np.polynomial.polynomial.Polynomial.fit(
                        x, y, [n], full=True
                    )

                    # Get error
                    error = self.chi_square(poly, x, y)
                    print("Order: {}, Error: {} ".format(n, error))

                    poly_fits.append((poly, error))

            except Exception:
                pass
        return sorted(poly_fits, key=lambda err: err[1])
