"""Unit tests for optimizer.py module."""
import unittest
from typing import List
from typing import Tuple
from unittest.mock import Mock

import numpy as np
import pandas

from yotse.optimization.generic_optimization import GenericOptimization
from yotse.optimization.optimizer import Optimizer
from yotse.pre import Experiment


class TestOptimizer(unittest.TestCase):
    """Set of tests to validate `Optimizer` class."""

    def test_init(self) -> None:
        """Test that Optimizer is initialized correctly."""
        mock_optimization = Mock(spec=GenericOptimization)
        test_optimizer = Optimizer(optimization_algorithm=mock_optimization)
        self.assertEqual(test_optimizer.optimization_algorithm, mock_optimization)
        self.assertFalse(test_optimizer._is_executed)
        with self.assertRaises(ValueError):
            _ = Optimizer(optimization_algorithm=42)  # type: ignore[arg-type]

    def test_best_solution(self) -> None:
        """Test that Optimizer correctly suggests the best solution."""

        class MockOpt(GenericOptimization):
            """Mock optimization class to test."""

            def get_best_solution(self) -> Tuple[List[float], float, int]:
                """Get mock solution."""
                return [1, 0.1, 0.01], 2.0, 3

        test_optimizer = Optimizer(
            optimization_algorithm=MockOpt(function=None, opt_instance=None)  # type: ignore
        )
        solution = test_optimizer.suggest_best_solution()
        # test output types
        self.assertIsInstance(solution, tuple)
        self.assertIsInstance(solution[0], list)
        self.assertIsInstance(solution[1], float)
        self.assertIsInstance(solution[2], int)
        self.assertEqual(len(solution), 3)
        # test correct value
        self.assertEqual(solution, ([1, 0.1, 0.01], 2.0, 3))

    def test_optimize(self) -> None:
        """Test that Optimizer correctly optimizes."""

        class MockOpt(GenericOptimization):
            """Mock optimization class to test."""

            def __init__(self) -> None:
                """Init mock object."""
                self.was_executed = 0

            def execute(self) -> None:
                """Execute mock object."""
                self.was_executed += 1

        test_go = MockOpt()  # type: ignore
        test_optimizer = Optimizer(test_go)
        self.assertEqual(test_go.was_executed, 0)
        test_optimizer.optimize()
        self.assertEqual(test_go.was_executed, 1)
        test_optimizer.optimize()
        test_optimizer.optimize()
        self.assertEqual(test_go.was_executed, 3)

    def test_update_blackbox_cost_data(self) -> None:
        """Test that the optimizer correctly checks data against the experiment."""
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
        test_optimization = GenericOptimization(function=None)  # type: ignore
        test_opt = Optimizer(test_optimization)
        # test update_internal_cost_data
        self.assertTrue(test_optimization.input_param_cost_df.empty)
        with self.assertRaises(ValueError):
            # update data will disagree with data_points in experiment
            test_opt.update_blackbox_cost_data(experiment=test_exp, data=test_df)
        with self.assertRaises(ValueError):
            # length of update data will disagree
            test_exp.data_points = test_points[:2]
            test_opt.update_blackbox_cost_data(experiment=test_exp, data=test_df)

        test_exp.data_points = test_points
        test_opt.update_blackbox_cost_data(experiment=test_exp, data=test_df)
        self.assertTrue(test_optimization.input_param_cost_df.equals(test_df))
        test_exp.data_points = test_points2
        test_opt.update_blackbox_cost_data(experiment=test_exp, data=test_df2)
        self.assertTrue(test_optimization.input_param_cost_df.equals(test_df2))
        # test float representation errors
        test_opt.update_blackbox_cost_data(experiment=test_exp, data=test_df2_unprecise)
        self.assertTrue(
            test_optimization.input_param_cost_df.equals(test_df2_unprecise)
        )
