import os
import unittest
import numpy as np
from qiaopt.pre import Parameter, ExperimentalSystemSetup, Experiment


class TestParameters(unittest.TestCase):
    """Test the parameters class."""
    @staticmethod
    def create_default_param(name="bright_state_parameter", parameter_range=[.1, .9], number_points=9,
                             distribution="linear", constraints=[None], custom_distribution=None):
        return Parameter(name=name, parameter_range=parameter_range, number_points=number_points,
                         distribution=distribution, constraints=constraints, custom_distribution=custom_distribution)

    def test_initialization(self):
        test_parameter = self.create_default_param()
        self.assertEqual(len(test_parameter.data_points), test_parameter.number_points)
        np.testing.assert_almost_equal(test_parameter.data_points, [.1, .2, .3, .4, .5, .6, .7, .8, .9])

    def test_invalid_distribution(self):
        with self.assertRaises(ValueError):
            self.create_default_param(distribution='invalid')

    def test_custom_distribution(self):
        @staticmethod
        def mock_distribution(min_value, max_value, number_points):
            return [.1, .5, .8]

        with self.assertRaises(ValueError):
            self.create_default_param(custom_distribution=mock_distribution)
        with self.assertRaises(ValueError):
            self.create_default_param(distribution='custom')
        with self.assertRaises(ValueError):
            self.create_default_param(distribution='custom', custom_distribution=mock_distribution)
        custom_param = self.create_default_param(number_points=3, distribution='custom',
                                                 custom_distribution=mock_distribution)
        self.assertListEqual(custom_param.data_points, [.1, .5, .8])

    def test_initial_data_points_within_range(self):
        linear_param = self.create_default_param(distribution='linear')
        self.assertEqual(len(linear_param.data_points), linear_param.number_points)
        self.assertAlmostEqual(linear_param.data_points[0], linear_param.range[0])
        self.assertAlmostEqual(linear_param.data_points[-1], linear_param.range[1])

        for dist in ['uniform', 'normal', 'log']:
            dist_param = self.create_default_param(distribution=dist)
            self.assertEqual(len(dist_param.data_points), dist_param.number_points)
            self.assertGreaterEqual(max(dist_param.data_points), dist_param.range[0])
            self.assertLessEqual(min(dist_param.data_points), dist_param.range[1])


class TestSystemSetup(unittest.TestCase):
    """Test the ExperimentalSystemSetup class."""

    def test_invalid_directory(self):
        invalid_directory = '/invalid/directory'
        with self.assertRaises(ValueError):
            ExperimentalSystemSetup(invalid_directory, 'experiment.py', {'arg1': 0.1, 'arg2': 'value2'})
        # test correct setup
        ExperimentalSystemSetup(os.getcwd(), 'experiment.py', {'arg1': 0.1, 'arg2': 'value2'}
)


class TestExperiment(unittest.TestCase):
    """Test the Experiment class."""

    def create_default_experiment(self, parameters=None, optimization_steps=None):
        return Experiment(experiment_name='default_exp',
                          system_setup=ExperimentalSystemSetup(os.getcwd(),
                                                               'experiment.py',
                                                               {'arg1': 0.1, 'arg2': 'value2'}),
                          parameters=parameters,
                          optimization_steps=optimization_steps)

    def test_add_parameter(self):
        test_exp = self.create_default_experiment()
        self.assertEqual(len(test_exp.parameters), 0)
        test_param = TestParameters.create_default_param()
        test_exp.add_parameter(test_param)
        self.assertEqual(len(test_exp.parameters), 1)

    def test_add_optimization_step(self):
        test_exp = self.create_default_experiment()
        self.assertEqual(len(test_exp.optimization_steps), 0)

        test_opt = self.create_default_experiment(optimization_steps=[('GA', 5)])
        test_opt.add_optimization_step(('GD', 3))
        self.assertEqual(len(test_opt.optimization_steps), 2)
        self.assertEqual(test_opt.optimization_steps[-1], ('GD', 3))


if __name__ == '__main__':
    unittest.main()
