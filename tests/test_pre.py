import os
import unittest
import numpy as np
import itertools
import inspect
from qiaopt.pre import Parameter, SystemSetup, Experiment, OptimizationInfo

DUMMY_FILE = "experiment.py"


class TestParameters(unittest.TestCase):
    """Test the parameters class."""
    @staticmethod
    def create_default_param(name="bright_state_parameter", parameter_range=[.1, .9], number_points=9,
                             distribution="linear", constraints=None, custom_distribution=None,
                             param_type="continuous", parameter_active=True, depends_on=None):
        return Parameter(name=name, param_range=parameter_range, number_points=number_points,
                         distribution=distribution, constraints=constraints, custom_distribution=custom_distribution,
                         param_type=param_type, parameter_active=parameter_active, depends_on=depends_on)

    def test_initialization(self):
        test_parameter = self.create_default_param()
        self.assertEqual(len(test_parameter.data_points), test_parameter.number_points)
        np.testing.assert_almost_equal(test_parameter.data_points, [.1, .2, .3, .4, .5, .6, .7, .8, .9])

    def test_invalid_distribution(self):
        with self.assertRaises(ValueError):
            self.create_default_param(distribution='invalid')

    def test_custom_distribution(self):

        def mock_distribution(min_value, max_value, number_points):
            return [.1, .5, .8]

        for param_type in ["continuous", "discrete"]:
            with self.assertRaises(ValueError):
                self.create_default_param(custom_distribution=mock_distribution, param_type=param_type)
            with self.assertRaises(ValueError):
                self.create_default_param(distribution='custom', param_type=param_type)
            with self.assertRaises(ValueError):
                self.create_default_param(distribution='custom', custom_distribution=mock_distribution,
                                          param_type=param_type)
            custom_param = self.create_default_param(number_points=3, distribution='custom',
                                                     custom_distribution=mock_distribution,
                                                     param_type=param_type)
            self.assertListEqual(custom_param.data_points, [.1, .5, .8])
            with self.assertRaises(ValueError):
                self.create_default_param(param_type="something")

    def test_initial_data_points_within_range(self):
        for param_type in ["continuous", "discrete"]:
            linear_param = self.create_default_param(distribution='linear', param_type=param_type,
                                                     parameter_range=[1., 9.])
            self.assertEqual(len(linear_param.data_points), linear_param.number_points)
            self.assertAlmostEqual(linear_param.data_points[0], linear_param.range[0])
            self.assertAlmostEqual(linear_param.data_points[-1], linear_param.range[1])

            for dist in ['uniform', 'normal', 'log']:
                dist_param = self.create_default_param(distribution=dist, param_type=param_type,
                                                       parameter_range=[1., 9.])
                self.assertEqual(len(dist_param.data_points), dist_param.number_points)
                self.assertGreaterEqual(max(dist_param.data_points), dist_param.range[0])
                self.assertLessEqual(min(dist_param.data_points), dist_param.range[1])

    def test_generate_data_points(self):
        test_parameter = self.create_default_param(number_points=5)
        test_parameter.generate_data_points(num_points=3)
        self.assertEqual(len(test_parameter.data_points), 3)
        np.testing.assert_almost_equal(test_parameter.data_points, [.1, .5, .9])

    def test_generate_dependent_data_points(self):
        def linear_dep(x, y):
            return x * y

        param1 = self.create_default_param(name="param1", number_points=4, distribution="linear",
                                           parameter_range=[1, 4])
        param2 = self.create_default_param(name="param2", number_points=4, distribution="linear",
                                           parameter_range=[1, 4], depends_on={'name': "param1",
                                                                               'function': linear_dep})
        param_list = [param1, param2]
        param2.generate_dependent_data_points(param_list)
        self.assertListEqual(param2.data_points, [1, 4, 9, 16])

        def fancy_dep(x, y):
            return 2*x**y

        param3 = self.create_default_param(name="param3", number_points=4, distribution="linear",
                                           parameter_range=[1, 4], depends_on={'name': "param1",
                                                                               'function': fancy_dep})
        param_list = [param1, param3]
        param3.generate_dependent_data_points(param_list)
        self.assertListEqual(param3.data_points, [2, 8, 54, 512])

    def test_is_active_property(self):
        active_param = self.create_default_param(parameter_active=True)
        inactive_param = self.create_default_param(parameter_active=False)
        self.assertTrue(active_param.is_active)
        self.assertFalse(inactive_param.is_active)


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
                                  executor='bash', output_dir_name='out', output_extension='txt', venv='test/test-env/',
                                  num_nodes=42, alloc_time='115:00:00', slurm_args=['--exclusive'],
                                  modules=['PYTHON3.10']
                                  )
        assert test_system.source_directory == os.getcwd()
        assert test_system.program_name == os.path.join(os.getcwd(), DUMMY_FILE)
        assert test_system.cmdline_arguments == {'--arg1': 0.1, '--arg2': 'value2'}
        assert test_system.analysis_script is None
        assert test_system.job_args["exec"] == 'bash'
        assert test_system.output_dir_name == 'out'
        assert test_system.output_extension == 'txt'
        assert test_system.venv == 'test/test-env/'
        assert test_system.job_args["venv"] == 'test/test-env/'
        assert test_system.num_nodes == 42
        assert test_system.alloc_time == '115:00:00'
        assert test_system.slurm_args == ['--exclusive']
        assert test_system.modules == ['PYTHON3.10']

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
                                         param_range=[1, 3],
                                         number_points=3,
                                         distribution="linear",
                                         parameter_active=True))
        test_exp.add_parameter(Parameter(name='inactive_param',
                                         param_range=[11, 13],
                                         number_points=3,
                                         distribution="linear",
                                         parameter_active=False))
        test_exp.add_parameter(Parameter(name='active_param2',
                                         param_range=[21, 23],
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
                                                                                      opt_parameters={'pop_size': 5},
                                                                                      is_active=True)])
        test_opt.add_optimization_info(OptimizationInfo(name='GD', opt_parameters={}, is_active=False))
        self.assertEqual(len(test_opt.optimization_information_list), 2)
        self.assertEqual(test_opt.optimization_information_list[-1].name, 'GD')

    def test_generate_slurm_script(self):
        """Test generation of a default slurm script for the Experiment."""
        test_exp = self.create_default_experiment()
        test_exp.system_setup.num_nodes = 42
        test_exp.system_setup.alloc_time = "01:00:00"
        test_exp.system_setup.venv = "test/test-env/"
        test_exp.system_setup.slurm_args = ["--exclusive"]
        test_exp.system_setup.modules = ["2023", "Python/3.11.1"]

        test_exp.generate_slurm_script('test_pre.py')

        # Read the contents of the slurm.job file
        with open("slurm.job", "r") as file:
            script_contents = file.readlines()

        # Define the expected output
        expected_output = [
            "!/bin/bash\n",
            "#SBATCH --nodes=42\n",
            "#SBATCH --exclusive\n",
            "#SBATCH --time=01:00:00\n",
            "\n",
            "\n",
            "module purge\n",
            "module load 2023\n",
            "module load Python/3.11.1\n",
            "source test/test-env/bin/activate\n",
            "\n",
            "python test_pre.py\n"
        ]

        # Compare the generated contents with the expected output line-by-line
        for line_num, (generated_line, expected_line) in enumerate(zip(script_contents, expected_output), start=1):
            assert generated_line == expected_line, \
                f"Line {line_num} of the generated slurm.job file does not match the expected output."

        # Ensure that the number of lines in the generated file matches the expected number of lines
        assert len(script_contents) == len(
            expected_output), "The generated slurm.job file has a different number of lines than the expected output."


if __name__ == '__main__':
    unittest.main()
