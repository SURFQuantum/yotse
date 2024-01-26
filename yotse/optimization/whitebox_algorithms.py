"""whitebox_algorithms.py.

This module provides classes for performing whitebox optimization, aka an optimization where the function is known.

Classes
-------
SciPyOpt:
    A class for optimization using the scipy.optimize.minimize function.
"""
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult

from yotse.optimization.generic_optimization import GenericOptimization


class SciPyOpt(GenericOptimization):
    """A class for optimization using the scipy.optimize.minimize function.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : array_like
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives (if any).
    method : str, optional
        Type of solver. Default is 'BFGS'.
    jac : callable or None, optional
        Jacobian (gradient) of the objective function. If None, it will be computed numerically.
    bounds : sequence or None, optional
        Bounds for variables (only for L-BFGS-B, TNC, COBYLA, and trust-constr methods).
    constraints : dict or sequence of dict, optional
        Constraints definition (only for COBYLA and trust-constr methods).
    tol : float or None, optional
        Tolerance for termination. For detailed control, use solver-specific options.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the current parameter vector.
    options : dict, optional
        A dictionary of solver options.
    """

    def __init__(
        self,
        fun: Callable[..., float],
        x0: Any,
        args: Optional[Tuple[Any]] = (),  # type: ignore [assignment]
        method: str = "BFGS",
        jac: Optional[Callable[..., Any]] = None,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        constraints: Optional[Union[dict, Sequence[dict]]] = (),  # type: ignore [type-arg]
        tol: Optional[float] = None,
        callback: Optional[Callable[..., Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the SciPyOpt object.

        Parameters
        ----------
        fun : callable
            The objective function to be minimized.
        x0 : array_like
            Initial guess.
        args : tuple, optional
            Extra arguments passed to the objective function and its derivatives (if any).
        method : str, optional
            Type of solver. Default is 'BFGS'.
        jac : callable or None, optional
            Jacobian (gradient) of the objective function. If None, it will be computed numerically.
        bounds : sequence or None, optional
            Bounds for variables (only for L-BFGS-B, TNC, COBYLA, and trust-constr methods).
        constraints : dict or sequence of dict, optional
            Constraints definition (only for COBYLA and trust-constr methods).
        tol : float or None, optional
            Tolerance for termination. For detailed control, use solver-specific options.
        callback : callable, optional
            Called after each iteration, as callback(xk), where xk is the current parameter vector.
        options : dict, optional
            A dictionary of solver options.

        Returns
        -------
        None
        """
        self.x0 = x0
        self.args = args
        self.method = method
        self.jac = jac
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options
        self.result: OptimizeResult = None

        super().__init__(function=fun)

    @property
    def current_datapoints(self) -> np.ndarray:
        """Return the current datapoints that will be used if an optimization is started
        now.

        In this case it is the initial guess.
        """
        return self.x0

    @property
    def max_iterations(self) -> int:
        """Return maximum number of iterations of SciPy optimization if specified."""
        if self.options is None:
            raise ValueError("Max iteration not specified for SciPy optimizer.")
        else:
            if "maxiter" in self.options:
                return int(self.options["maxiter"])
            elif "maxfun" in self.options:
                return int(self.options["maxfun"])
            else:
                raise ValueError("Max iteration not specified for SciPy optimizer.")

    def execute(self) -> None:
        """Execute the optimization using the specified parameters.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The optimization result.
        """
        result = minimize(
            self.function,
            self.x0,
            args=self.args,
            method=self.method,
            jac=self.jac,
            bounds=self.bounds,
            constraints=self.constraints,
            tol=self.tol,
            callback=self.callback,
            options=self.options,
        )
        self.result = result

    def get_best_solution(self) -> Tuple[List[float], float, int]:
        """Get the best solution. Should be implemented in every derived class.

        Returns
        -------
        solution, solution_fitness, solution_idx
            Solution, its fitness and its index in the list of data points.
        """
        if self.result is None:
            raise RuntimeError(
                "Trying to `get_best_solution` without result. Please make sure to call `execute` first."
            )
        return self.result.x.tolist(), self.result.fun, 0

    def get_new_points(self) -> np.ndarray:
        """Not needed, just passing."""
        pass

    def overwrite_internal_data_points(self, data_points: np.ndarray) -> None:
        """Not needed, just passing."""
        pass
