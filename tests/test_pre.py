import os
import unittest
import numpy as np
import itertools
from qiaopt.pre import Parameter, SystemSetup, Experiment, OptimizationInfo

DUMMY_FILE = "experiment.py"


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
    """Test the SystemSetup class."""

    def setUp(self) -> None:
        with open(DUMMY_FILE, "wt") as f:
            f.write("This is a dummy experiment file for test_pre.")

    def tearDown(self) -> None:
        os.remove(DUMMY_FILE)

    def test_invalid_directory_or_files(self):
        """Test if an invalid source_directory will correctly be caught."""
        invalid_directory = '/invalid/source_directory'

        with self.assertRaises(ValueError):
            SystemSetup(invalid_directory, DUMMY_FILE, {'--arg1': 0.1, '--arg2': 'value2'})
        with self.assertRaises(ValueError):
            SystemSetup(os.getcwd(), DUMMY_FILE, analysis_script='non_existent_file.sh')
        # test correct setup
        SystemSetup(os.getcwd(), DUMMY_FILE, {'--arg1': 0.1, '--arg2': 'value2'}
                    )

    def test_init(self):
        test_system = SystemSetup(source_directory=os.getcwd(), program_name=DUMMY_FILE,
                                  command_line_arguments={'--arg1': 0.1, '--arg2': 'value2'},
                                  executor='bash', files_needed=['*.sh', 'important_text.txt', 'generic_readme.md'])
        assert test_system.source_directory == os.getcwd()
        assert test_system.program_name == DUMMY_FILE
        assert test_system.cmdline_arguments == {'--arg1': 0.1, '--arg2': 'value2'}
        assert test_system.analysis_script is None
        assert test_system.executor == 'bash'
        assert test_system.files_needed == ['*.sh', 'important_text.txt', 'generic_readme.md']

    def test_cmdline_to_list(self):
        """Test if the dict of cmdline args is correctly converted to a list."""

        test_setup = SystemSetup(os.getcwd(), DUMMY_FILE, {'--arg1': 0.1, '--arg2': 'value2', '--arg3': False})
        assert test_setup.cmdline_dict_to_list() == ['--arg1', 0.1, '--arg2', 'value2', '--arg3', False]


class TestExperiment(unittest.TestCase):
    """Test the Experiment class."""
    def setUp(self) -> None:
        with open(DUMMY_FILE, "wt") as f:
            f.write("This is a dummy experiment file for test_pre.")

    def tearDown(self) -> None:
        os.remove(DUMMY_FILE)

    @staticmethod
    def create_default_experiment(parameters=None, optimization_info=None):
        """Helper function to set up a default experiment for the tests."""
        return Experiment(experiment_name='default_exp',
                          system_setup=SystemSetup(
                              source_directory=os.getcwd(),
                              program_name=DUMMY_FILE,
                              command_line_arguments={'arg1': 0.1, 'arg2': 'value2'}),
                          parameters=parameters,
                          opt_info_list=optimization_info)

    def test_c_product(self):
        """Test whether Cartesian product is correctly formed from active Parameters."""
        test_exp = self.create_default_experiment()
        test_exp.add_parameter(Parameter(name='active_param1',
                                         parameter_range=[1, 3],
                                         number_points=3,
                                         distribution="linear",
                                         parameter_active=True))
        test_exp.add_parameter(Parameter(name='inactive_param',
                                         parameter_range=[11, 13],
                                         number_points=3,
                                         distribution="linear",
                                         parameter_active=False))
        test_exp.add_parameter(Parameter(name='active_param2',
                                         parameter_range=[21, 23],
                                         number_points=3,
                                         distribution="linear",
                                         parameter_active=True))
        test_exp.create_datapoint_c_product()
        assert test_exp.data_points == list(itertools.product([1., 2., 3.], [21., 22., 23.]))
        # now activate 'inactive_param' and regenerate points
        test_exp.parameters[1].parameter_active = True
        assert test_exp.parameters[1].is_active
        test_exp.create_datapoint_c_product()
        assert test_exp.data_points == list(itertools.product([1., 2., 3.], [11., 12., 13.], [21., 22., 23.]))
        # now deactivate 'active_param1' and 'active_param2' and regenerate points
        test_exp.parameters[0].parameter_active = False
        test_exp.parameters[2].parameter_active = False
        test_exp.create_datapoint_c_product()
        assert test_exp.parameters[0].is_active is False
        assert test_exp.parameters[2].is_active is False
        assert test_exp.data_points == list(itertools.product([11., 12., 13.]))

    def test_add_parameter(self):
        """Test adding Parameters to an Experiment."""
        test_exp = self.create_default_experiment()
        self.assertEqual(len(test_exp.parameters), 0)
        test_param = TestParameters.create_default_param()
        test_exp.add_parameter(test_param)
        self.assertEqual(len(test_exp.parameters), 1)

    def test_add_optimization_information(self):
        """Test adding OptimizationInfo to an Experiment."""
        test_exp = self.create_default_experiment()
        self.assertEqual(len(test_exp.optimization_information_list), 0)

        test_opt = self.create_default_experiment(optimization_info=[OptimizationInfo(name='GA',
                                                                                      opt_parameters={'pop_size': 5})])
        test_opt.add_optimization_info(OptimizationInfo(name='GD', opt_parameters={}))
        self.assertEqual(len(test_opt.optimization_information_list), 2)
        self.assertEqual(test_opt.optimization_information_list[-1].name, 'GD')


if __name__ == '__main__':
    unittest.main()
