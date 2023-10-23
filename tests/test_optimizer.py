import unittest
from unittest.mock import Mock

from yotse.optimization.generic_optimization import GenericOptimization
from yotse.optimization.optimizer import Optimizer


class TestOptimizer(unittest.TestCase):

    def test_init(self):
        """Test that Optimizer is initialized correctly."""
        mock_optimization = Mock(spec=GenericOptimization)
        test_optimizer = Optimizer(optimization_algorithm=mock_optimization)
        self.assertEqual(test_optimizer.optimization_algorithm, mock_optimization)
        self.assertFalse(test_optimizer._is_executed)
        with self.assertRaises(ValueError):
            _ = Optimizer(optimization_algorithm=42)

    def test_best_solution(self):
        """Test that Optimizer correctly suggests the best solution."""
        class MockOpt(GenericOptimization):
            def get_best_solution(self) -> (list, float, int):
                return [1, 0.1, 0.01], 2., 3

        test_optimizer = Optimizer(optimization_algorithm=MockOpt(function=None))
        solution = test_optimizer.suggest_best_solution()
        # test output types
        self.assertIsInstance(solution, tuple)
        self.assertIsInstance(solution[0], list)
        self.assertIsInstance(solution[1], float)
        self.assertIsInstance(solution[2], int)
        self.assertEqual(len(solution), 3)
        # test correct value
        self.assertEqual(solution, ([1, 0.1, 0.01], 2., 3))

    def test_optimize(self):
        """Test that Optimizer correctly optimizes."""
        class MockOpt(GenericOptimization):
            def __init__(self):
                self.was_executed = 0

            def execute(self) -> None:
                self.was_executed += 1
        test_go = MockOpt()
        test_optimizer = Optimizer(test_go)
        self.assertEqual(test_go.was_executed, 0)
        test_optimizer.optimize()
        self.assertEqual(test_go.was_executed, 1)
        test_optimizer.optimize()
        test_optimizer.optimize()
        self.assertEqual(test_go.was_executed, 3)
