import unittest
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import pandas
from pygad.pygad import GA

from yotse.optimization.blackbox_algorithms import GAOpt
from yotse.optimization.optimizer import Optimizer
from yotse.pre import Experiment


class TestGAOpt(unittest.TestCase):
    @staticmethod
    def _paraboloid(ga_instance: GA, solution: List[float], sol_index: int) -> float:
        """A simple paraboloid function.

        Has one global minimum:
        f(x1,x2)=0.0; (x1,x2)=(0.0, 0.0)
        :param solution: List of x and y variables
        :param sol_index: index related to this solution
        :return: Value of the function
        """
        x_loc = solution[0]
        y_loc = solution[1]
        return x_loc**2 + y_loc**2

    @staticmethod
    def _sixhump(ga_instance: GA, solution: List[float], sol_index: int) -> float:
        """The six-hump camel back function.

        Has two global minimums:
        f(x1,x2)=-1.0316; (x1,x2)=(-0.0898,0.7126), (0.0898,-0.7126)
        :param solution: List of x and y variables
        :param sol_index: index related to this solution
        :return: Value of the function
        """
        x_loc = solution[0]
        y_loc = solution[1]
        return (
            (4 - 2.1 * x_loc**2 + (x_loc**4) / 3.0) * x_loc**2
            + x_loc * y_loc
            + (-4 + 4 * y_loc**2) * y_loc**2
        )

    @staticmethod
    def _rosenbrock(ga_instance: GA, solution: List[float], sol_index: int) -> float:
        """The Rosenbrock function.

        Has one global minimum:
        f(x1,x2)=0.0; (x1,x2)=(1.0, 1.0)
        :param solution: List of x and y variables
        :param sol_index: index related to this solution
        :return: Value of the function
        """
        x_loc = solution[0]
        y_loc = solution[1]
        return (1 - x_loc) ** 2 + 100 * (y_loc - x_loc**2) ** 2

    @staticmethod
    def _rastrigin(ga_instance: GA, solution: List[float], sol_index: int) -> Any:
        """The Rastrigin function.

        Has one global minimum:
        f(x1,x2)=0.0; (x1,x2)=(0.0, 0.0)
        :param solution: List of x and y variables
        :param sol_index: index related to this solution
        :return: Value of the function
        """
        x_loc = solution[0]
        y_loc = solution[1]
        return (
            (x_loc**2 - 10 * np.cos(2 * np.pi * x_loc))
            + (y_loc**2 - 10 * np.cos(2 * np.pi * y_loc))
            + 20
        )

    def _setup_and_execute(
        self,
        function: Callable[..., float],
        var_range: Tuple[float, float] = (1.2, 1.2),
        var_step: float = 0.01,
    ) -> List[float]:
        self.x = list(np.arange(-var_range[0], var_range[1], var_step))
        self.y = list(np.arange(-var_range[0], var_range[1], var_step))
        initial_pop = []
        for i in range(len(self.x)):
            initial_pop.append((float(self.x[i]), float(self.y[i])))

        ga_opt = GAOpt(
            blackbox_optimization=False,
            initial_data_points=np.array(initial_pop),
            num_generations=100,
            num_parents_mating=10,
            fitness_func=function,
            # gene_type=float,
            mutation_probability=0.1,
        )
        opt = Optimizer(ga_opt)

        for _ in range(ga_opt.optimization_instance.num_generations):
            opt.optimize()
            ga_opt.optimization_instance.initial_population = (
                ga_opt.optimization_instance.population
            )

        # matplotlib.use('Qt5Agg')
        # ga_opt.ga_instance.plot_fitness()
        return ga_opt.get_best_solution()[0]

    def test_optimize_paraboloid(self) -> None:
        solution = self._setup_and_execute(self._paraboloid)
        x_true = 0.0
        y_true = 0.0

        self.assertTrue(np.abs(solution[0] - x_true) <= 1e-12)
        self.assertTrue(np.abs(solution[1] - y_true) <= 1e-12)

    def test_optimize_sixhump(self) -> None:
        solution = self._setup_and_execute(
            self._sixhump, var_range=(0.8, 0.8), var_step=0.01
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
        solution = self._setup_and_execute(self._rosenbrock)
        x_true = 1.0
        y_true = 1.0

        self.assertTrue(np.abs(solution[0] - x_true) <= 1e-12)
        self.assertTrue(np.abs(solution[1] - y_true) <= 1e-12)

    def test_optimize_rastrigin(self) -> None:
        solution = self._setup_and_execute(self._rastrigin)
        x_true = 0.0
        y_true = 0.0

        self.assertTrue(np.abs(solution[0] - x_true) <= 1e-12)
        self.assertTrue(np.abs(solution[1] - y_true) <= 1e-12)

    # todo : write test for constraint: check that in different cases (e.g. multiple params, single param /
    #  starting params within/outside constraints) all points created are always inside constrains (multiple iterations)


class TestGenericOptimization(unittest.TestCase):
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
        test_optimization = GAOpt(
            blackbox_optimization=True,
            initial_data_points=np.array([[1], [1]]),
            num_generations=1,
            num_parents_mating=1,
        )

        # test update_internal_cost_data
        self.assertTrue(test_optimization.input_param_cost_df.empty)
        with self.assertRaises(ValueError):
            # update data will disagree with data_points in experiment
            test_optimization.update_internal_cost_data(
                experiment=test_exp, data=test_df
            )
        with self.assertRaises(ValueError):
            # length of update data will disagree
            test_exp.data_points = test_points[:2]
            test_optimization.update_internal_cost_data(
                experiment=test_exp, data=test_df
            )

        test_exp.data_points = test_points
        test_optimization.update_internal_cost_data(experiment=test_exp, data=test_df)
        self.assertTrue(test_optimization.input_param_cost_df.equals(test_df))
        test_exp.data_points = test_points2
        test_optimization.update_internal_cost_data(experiment=test_exp, data=test_df2)
        self.assertTrue(test_optimization.input_param_cost_df.equals(test_df2))
        # test float representation errors
        test_optimization.update_internal_cost_data(
            experiment=test_exp, data=test_df2_unprecise
        )
        self.assertTrue(
            test_optimization.input_param_cost_df.equals(test_df2_unprecise)
        )

        # Test update_internal_cost_data.
        ga_instance = test_optimization.optimization_instance
        self.assertEqual(
            test_optimization.input_params_to_cost_value(ga_instance, [0.05, 0.08], 1),
            0.02,
        )
        # test float representation error
        self.assertEqual(
            test_optimization.input_params_to_cost_value(
                ga_instance, [0.04999999999, 0.07999999999], 1
            ),
            0.02,
        )
        with self.assertRaises(ValueError):
            test_optimization.input_params_to_cost_value(ga_instance, [0.049, 0.079], 0)


if __name__ == "__main__":
    unittest.main()
