import os
import unittest
import numpy as np
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

    def test_invalid_directory_or_files(self):
        invalid_directory = '/invalid/directory'
        with open(DUMMY_FILE, "wt") as f:
            f.write("This is a dummy experiment file for test_pre.")

        with self.assertRaises(ValueError):
            SystemSetup(invalid_directory, DUMMY_FILE, {'--arg1': 0.1, '--arg2': 'value2'})
        with self.assertRaises(ValueError):
            SystemSetup(os.getcwd(), DUMMY_FILE, analysis_script='non_existent_file.sh')
        # test correct setup
        SystemSetup(os.getcwd(), DUMMY_FILE, {'--arg1': 0.1, '--arg2': 'value2'}
                    )
        os.remove(DUMMY_FILE)
    
    def test_init(self):
        with open(DUMMY_FILE, "wt") as f:
            f.write("This is a dummy experiment file for test_pre.")

        test_system = SystemSetup(directory=os.getcwd(), program_name=DUMMY_FILE,
                                  command_line_arguments={'--arg1': 0.1, '--arg2': 'value2'},
                                  executor='bash', files_needed=['*.sh', 'important_text.txt', 'generic_readme.md'])
        assert test_system.directory == os.getcwd()
        assert test_system.program_name == DUMMY_FILE
        assert test_system.cmdline_arguments == {'--arg1': 0.1, '--arg2': 'value2'}
        assert test_system.analysis_script is None
        assert test_system.executor == 'bash'
        assert test_system.files_needed == ['*.sh', 'important_text.txt', 'generic_readme.md']
        os.remove(DUMMY_FILE)

    def test_cmdline_to_list(self):
        """Test if the dict of cmdline args is correctly converted to a list."""
        with open(DUMMY_FILE, "wt") as f:
            f.write("This is a dummy experiment file for test_pre.")
        test_setup = SystemSetup(os.getcwd(), DUMMY_FILE, {'--arg1': 0.1, '--arg2': 'value2', '--arg3': False})
        assert test_setup.cmdline_dict_to_list() == ['--arg1', 0.1, '--arg2', 'value2', '--arg3', False]
        os.remove(DUMMY_FILE)


class TestExperiment(unittest.TestCase):
    """Test the Experiment class."""

    @staticmethod
    def create_default_experiment(parameters=None, optimization_info=[]):
        with open(DUMMY_FILE, "wt") as f:
            f.write("This is a dummy experiment file for test_pre.")

        return Experiment(experiment_name='default_exp',
                          system_setup=SystemSetup(
                              directory=os.getcwd(),
                              program_name=DUMMY_FILE,
                              command_line_arguments={'arg1': 0.1, 'arg2': 'value2'}),
                          parameters=parameters,
                          opt_info_list=optimization_info)

    def test_add_parameter(self):
        test_exp = self.create_default_experiment()
        self.assertEqual(len(test_exp.parameters), 0)
        test_param = TestParameters.create_default_param()
        test_exp.add_parameter(test_param)
        self.assertEqual(len(test_exp.parameters), 1)

        os.remove(DUMMY_FILE)

    def test_add_optimization_information(self):
        test_exp = self.create_default_experiment()
        self.assertEqual(len(test_exp.optimization_information_list), 0)

        test_opt = self.create_default_experiment(optimization_info=[OptimizationInfo(name='GA',
                                                                                      opt_parameters={'pop_size': 5})])
        test_opt.add_optimization_info(OptimizationInfo(name='GD', opt_parameters={}))
        self.assertEqual(len(test_opt.optimization_information_list), 2)
        self.assertEqual(test_opt.optimization_information_list[-1].name, 'GD')

        os.remove(DUMMY_FILE)


if __name__ == '__main__':
        unittest.main()
