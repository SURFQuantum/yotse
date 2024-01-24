import unittest
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import pandas
from bayes_opt import UtilityFunction
from scipy.optimize import LinearConstraint

from yotse.optimization.blackbox_algorithms import BayesOpt
from yotse.optimization.blackbox_algorithms import GAOpt
from yotse.optimization.generic_optimization import GenericOptimization
from yotse.optimization.optimizer import Optimizer
from yotse.optimization.whitebox_algorithms import SciPyOpt
from yotse.pre import Experiment


def _paraboloid(params: List[float]) -> float:
    """A simple paraboloid function.

    Has one global minimum:
    f(x1, x2) = 0.0; (x1, x2) = (0.0, 0.0)

    Parameters
    ----------
    params : List[float]
        List of x and y variables.

    Returns
    -------
    float
        Value of the function.
    """
    x_loc = params[0]
    y_loc = params[1]
    return x_loc**2 + y_loc**2


def _sixhump(
    params: List[float],
) -> float:
    """The six-hump camelback function.

    Has two global minimums:
    f(x1, x2) = -1.0316; (x1, x2) = (-0.0898, 0.7126), (0.0898, -0.7126)

    Parameters
    ----------
    params : List[float]
        List of x and y variables.

    Returns
    -------
    float
        Value of the function.
    """
    x_loc = params[0]
    y_loc = params[1]
    return (
        (4 - 2.1 * x_loc**2 + (x_loc**4) / 3.0) * x_loc**2
        + x_loc * y_loc
        + (-4 + 4 * y_loc**2) * y_loc**2
    )


def _rosenbrock(params: List[float]) -> float:
    """The Rosenbrock function.

    Has one global minimum:
    f(x1, x2) = 0.0; (x1, x2) = (1.0, 1.0)

    Parameters
    ----------
    params : List[float]
        List of x and y variables.

    Returns
    -------
    float
        Value of the function.
    """
    x_loc = params[0]
    y_loc = params[1]
    return (1 - x_loc) ** 2 + 100 * (y_loc - x_loc**2) ** 2


def _rastrigin(params: List[float]) -> Any:
    """The Rastrigin function.

    Has one global minimum:
    f(x1, x2) = 0.0; (x1, x2) = (0.0, 0.0)

    Parameters
    ----------
    params : List[float]
        List of x and y variables.

    Returns
    -------
    float
        Value of the function.
    """
    x_loc = params[0]
    y_loc = params[1]
    return (
        (x_loc**2 - 10 * np.cos(2 * np.pi * x_loc))
        + (y_loc**2 - 10 * np.cos(2 * np.pi * y_loc))
        + 20
    )


def _execute_blackbox(
    optimization_algorithm: GenericOptimization,
    function: Callable[..., float],
) -> List[float]:
    """Execute blackbox optimization.

    Parameters
    ----------
    optimization_algorithm : GenericOptimization
        The optimizer to be executed.

    Returns
    -------
    List[float]
        Best solution found by the optimization.
    """
    optimizer = Optimizer(optimization_algorithm)
    # execute multiple steps
    for _ in range(optimizer.optimization_algorithm.num_iterations):  # type: ignore
        # generate points of our "blackbox"-function
        blackbox_solutions = [
            function(value)
            for value in optimizer.optimization_algorithm.current_datapoints
        ]
        x_values, y_values = zip(*optimizer.optimization_algorithm.current_datapoints)
        data = {"f(x,y)": blackbox_solutions, "x": x_values, "y": y_values}
        df = pandas.DataFrame(data)

        # input results of our "blackbox"-function to optimization
        optimizer.optimization_algorithm.update_internal_cost_data(data=df)
        # optimize now we have our data on this data
        optimizer.optimize()

        # construct new points for next iteration
        new_points = optimizer.optimization_algorithm.get_new_points()
        optimizer.optimization_algorithm.overwrite_internal_data_points(new_points)
        # repeat

    # return solution to test against
    return optimizer.suggest_best_solution()[0]


class TestBlackBoxOptimization(unittest.TestCase):
    """Test case for yotse.optimization.blackbox_algorithms."""

    @staticmethod
    def _setup_and_execute_ga_optimization(
        function: Callable[..., float], var_range: Tuple[float, float] = (1.2, 1.2)
    ) -> List[float]:
        """Setup and execute the genetic algorithm optimization."""
        var_step: float = 0.01
        # generate initial population
        x_vals = list(np.arange(-var_range[0], var_range[1], var_step))
        y_vals = list(np.arange(-var_range[0], var_range[1], var_step))
        initial_pop = []
        for i in range(len(x_vals)):
            initial_pop.append((float(x_vals[i]), float(y_vals[i])))
        # set up optimizer
        ga_opt = GAOpt(
            blackbox_optimization=True,
            initial_data_points=np.array(initial_pop),
            num_generations=100,
            num_parents_mating=10,
            # gene_type=float,
            mutation_probability=0.1,
        )

        return _execute_blackbox(ga_opt, function)

    @staticmethod
    def _setup_and_execute_bayesian_optimization(
        function: Callable[..., float], var_range: Tuple[float, float] = (1.2, 1.2)
    ) -> List[float]:
        """Setup and execute the bayesian optimization."""
        opt_parameters = {
            "utility_function": UtilityFunction(kind="ucb", kappa=2.5, xi=0.0),
            "n_iter": 10,
        }
        # set up optimizer
        bayes_opt = BayesOpt(
            blackbox_optimization=True,
            pbounds={
                "x": (
                    int(var_range[0]),
                    int(-var_range[1]),
                ),  # cast to int for bayesian opt, see comment in class
                "y": (
                    int(var_range[0]),
                    int(-var_range[1]),
                ),  # cast to int for bayesian opt, see comment in class
            },
            **opt_parameters,
        )

        return _execute_blackbox(bayes_opt, function)

    def test_optimize_paraboloid(self) -> None:
        """Test optimization of the paraboloid function."""
        solution = self._setup_and_execute_ga_optimization(_paraboloid)
        x_true = 0.0
        y_true = 0.0

        self.assertTrue(np.abs(solution[0] - x_true) <= 1e-12)
        self.assertTrue(np.abs(solution[1] - y_true) <= 1e-12)

    def test_optimize_sixhump(self) -> None:
        """Test optimization of the six-hump camelback function."""
        solution = self._setup_and_execute_ga_optimization(
            _sixhump, var_range=(0.8, 0.8)
        )
        x_true = -0.0898
        y_true = 0.7126

        # Check one of the possible solutions
        if solution[0] > 0:
            x_true *= -1
            y_true *= -1
        self.assertTrue(np.abs(solution[0] - x_true) <= 1e-2)
        self.assertTrue(np.abs(solution[1] - y_true) <= 1e-2)

    def test_optimize_rosenbrock(self) -> None:
        """Test optimization of the Rosenbrock function."""
        solution = self._setup_and_execute_ga_optimization(_rosenbrock)
        x_true = 1.0
        y_true = 1.0

        self.assertTrue(np.abs(solution[0] - x_true) <= 1e-12)
        self.assertTrue(np.abs(solution[1] - y_true) <= 1e-12)

    def test_optimize_rastrigin(self) -> None:
        """Test optimization of the Rastrigin function."""
        solution = self._setup_and_execute_ga_optimization(_rastrigin)
        x_true = 0.0
        y_true = 0.0

        self.assertTrue(np.abs(solution[0] - x_true) <= 1e-12)
        self.assertTrue(np.abs(solution[1] - y_true) <= 1e-12)

    # todo : write test for constraint: check that in different cases (e.g. multiple params, single param /
    #  starting params within/outside constraints) all points created are always inside constrains (multiple iterations)


class TestGenericOptimization(unittest.TestCase):
    """Test case for yotse.optimization.generic_optimization.GenericOptimization."""

    def test_get_function(self) -> None:
        """Test get_function."""

        def mock_function(x: float) -> float:
            """Mock function."""
            return x

        test_opt = GenericOptimization(function=mock_function)  # type: ignore[abstract]
        self.assertEqual(test_opt.get_function(), mock_function)

    def test_generic_optimization_update_internal_cost_data(self) -> None:
        """Combined test for input_params_to_cost_value & update_internal_cost_data
        (because otherwise the dict is empty)."""
        test_df = pandas.DataFrame({"f": [1, 2, 3], "x": [4, 5, 6], "y": [7, 8, 9]})
        test_points = np.array([[4, 7], [5, 8], [6, 9]])
        test_df2 = pandas.DataFrame(
            {"f": [0.01, 0.02, 0.03], "x": [0.04, 0.05, 0.06], "y": [0.07, 0.08, 0.09]}
        )
        test_points2 = np.array([[0.04, 0.07], [0.05, 0.08], [0.06, 0.09]])
        test_df2_unprecise = pandas.DataFrame(
            {
                "f": [0.01, 0.02, 0.03],
                "x": [0.04, 0.04999999999, 0.06],
                "y": [0.07, 0.07999999999, 0.09],
            }
        )

        test_exp = Experiment(experiment_name="test", system_setup=None)  # type: ignore[arg-type]
        test_optimization = GenericOptimization(function=None)  # type: ignore[abstract, arg-type]

        test_exp.data_points = test_points
        test_optimization.update_internal_cost_data(data=test_df)
        self.assertTrue(test_optimization.input_param_cost_df.equals(test_df))
        test_exp.data_points = test_points2
        test_optimization.update_internal_cost_data(data=test_df2)
        self.assertTrue(test_optimization.input_param_cost_df.equals(test_df2))
        # test float representation errors
        test_optimization.update_internal_cost_data(data=test_df2_unprecise)
        self.assertTrue(
            test_optimization.input_param_cost_df.equals(test_df2_unprecise)
        )

        # Test input_params_to_cost_value
        self.assertEqual(
            test_optimization.input_params_to_cost_value([0.05, 0.08], 1),
            0.02,
        )
        # test float representation error
        self.assertEqual(
            test_optimization.input_params_to_cost_value(
                [0.04999999999, 0.07999999999], 1
            ),
            0.02,
        )
        with self.assertRaises(ValueError):
            test_optimization.input_params_to_cost_value([0.049, 0.079], 0)


class TestWhiteboxOptimization(unittest.TestCase):
    """Test case for yotse.optimization.whitebox_algorithms.WhiteboxOptimization."""

    def test_scipy_quadratic_minimization(self) -> None:
        """Test minimization of a simple quadratic function.

        The quadratic function is f(x) = (x - 2)^2, and the test checks
        if the optimization successfully finds the minimum value close to 2.0.
        """

        def objective_function(x: float) -> float:
            """Simple shifted quadratic function."""
            return (x - 2) ** 2

        test_scipy = SciPyOpt(fun=objective_function, x0=0.0, method="BFGS")
        test_scipy.execute()

        self.assertTrue(test_scipy.result.success)
        self.assertAlmostEqual(test_scipy.result.x[0], 2.0, places=6)
        self.assertAlmostEqual(test_scipy.get_best_solution()[0][0], 2.0, places=6)

    def test_constraint_minimization(self) -> None:
        """Test minimization of a function with an inequality constraint.

        The objective function is f(x) = x^2, subject to the constraint x >= 3. The test
        checks if the optimization successfully satisfies the constraint and finds an
        optimized value greater than or equal to 3.0.
        """

        def objective_function(x: float) -> float:
            """Simple quadratic function around 0."""
            return x**2

        constraint = LinearConstraint([[1]], [3], [np.inf])

        test_scipy = SciPyOpt(
            fun=objective_function,
            x0=[4.0],
            constraints=[constraint],
            method="trust-constr",
        )
        test_scipy.execute()

        self.assertTrue(test_scipy.result.success)
        self.assertGreaterEqual(test_scipy.result.x[0], 3.0)
        self.assertGreaterEqual(test_scipy.get_best_solution()[0][0], 3.0)


if __name__ == "__main__":
    unittest.main()
