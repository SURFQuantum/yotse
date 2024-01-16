import warnings
from typing import Any
from typing import List

import numpy as np


class FuncFit:
    def __init__(self) -> None:
        pass

    def add_noise(self, arr: np.ndarray, level: float = 0.2) -> np.ndarray:
        """Add noise to the given array :param arr: Array of values (numpy array) :param
        level: Noise level :return: Input array with added noise."""
        return np.random.normal(2 * arr + 2, level)

    def chi_square(self, poly: np.poly1d, x: List[float], y: List[float]) -> Any:
        """Calculate chi-square :param poly: Polynomial (numpy.poly1d object) :param x:

        X-points of the original dataset
        :param y: Y-points of the original dataset
        :return: Chi-square.
        """
        return np.sum((np.polyval(poly, x) - y) ** 2) / len(x)

    def err(self, poly: np.poly1d, x: List[float], y: List[float]) -> Any:
        """Calculate error :param poly: Polynomial (numpy.poly1d object) :param x:

        X-points of the original dataset
        :param y: Y-points of the original dataset
        :return: Error.
        """
        return np.sqrt(np.sum(np.power(poly(x) - y, 2.0)))

    def find_poly_fit(
        self, x: List[float], y: List[float], max_order: int = 41
    ) -> List[np.poly1d]:
        """Find a polynomial that fits the given data best.

        :param x: X-points
        :param y: Y-points
        :param max_order: Maximum order of a polynomial
        :return: A list of all polynomials and the corresponding errors. The list is
            sorted in ascending order using the error values.
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
                    poly, residuals, _, _, _ = np.polyfit(x, y, n, full=True)
                    poly_1d = np.poly1d(poly)

                    # Get error
                    # error = self.err(poly, x, y)
                    error = self.chi_square(poly, x, y)
                    print("Order: {}, Error: {} ".format(poly_1d.order, error))
                    # print('Order: {}, Error: {}, Coeffs: {}, '.format(trend.order, error, trend.coeffs))

                    poly_fits.append((poly_1d, error))

            except Exception:
                pass

        return sorted(poly_fits, key=lambda err: err[1])
