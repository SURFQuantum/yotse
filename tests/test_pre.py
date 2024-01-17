"""Unit tests for the `yotse.pre` package's `Parameter`, `SystemSetup`, and `Experiment`
classes."""
import itertools
import os
import unittest
from io import StringIO
from typing import Callable
from typing import List
from typing import Optional
from typing import Union
from unittest.mock import patch

import numpy as np

from yotse.pre import ConstraintDict
from yotse.pre import Experiment
from yotse.pre import OptimizationInfo
from yotse.pre import Parameter
from yotse.pre import ParameterDependencyDict
from yotse.pre import SystemSetup

DUMMY_FILE = "experiment.py"


def valid_function() -> None:
    """Test function for TestExperiment.test_cost_function_setter_and_getter()"""
    pass


class TestParameters(unittest.TestCase):
    """A set of tests to validate the `Parameter` class from the `yotse.pre` package."""

    @staticmethod
    def create_default_param(
        name: str = "bright_state_parameter",
        parameter_range: List[Union[float, int]] = [0.1, 0.9],
        number_points: int = 9,
        distribution: str = "linear",
        constraints: Union[ConstraintDict, np.ndarray, None] = None,
        custom_distribution: Optional[Callable[[float, float, int], np.ndarray]] = None,
        param_type: str = "continuous",
        parameter_active: bool = True,
        depends_on: Optional[ParameterDependencyDict] = None,
    ) -> Parameter:
        """Static method that creates a default `Parameter` instance with specified
        properties."""
        return Parameter(
            name=name,
            param_range=parameter_range,
            number_points=number_points,
            distribution=distribution,
            constraints=constraints,
            custom_distribution=custom_distribution,
            param_type=param_type,
            parameter_active=parameter_active,
            depends_on=depends_on,
        )

    def test_initialization(self) -> None:
        """Test to verify correct initialization of `Parameter` instances."""
        test_parameter = self.create_default_param()
        self.assertEqual(len(test_parameter.data_points), test_parameter.number_points)
        np.testing.assert_almost_equal(
            test_parameter.data_points, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

    def test_invalid_distribution(self) -> None:
        """Test to verify that an exception is raised for invalid distributions."""
        with self.assertRaises(ValueError):
            self.create_default_param(distribution="invalid")

    def test_custom_distribution(self) -> None:
        """Test to check the functionality of providing a custom distribution for
        `Parameter` instances."""

        def mock_distribution(
            min_value: float, max_value: float, number_points: int
        ) -> np.ndarray:
            """Mock custom distribution that outputs fixed set of floats."""
            return np.array((0.1, 0.5, 0.8))

        for param_type in ["continuous", "discrete"]:
            with self.assertRaises(ValueError):
                self.create_default_param(
                    custom_distribution=mock_distribution, param_type=param_type
                )
            with self.assertRaises(ValueError):
                self.create_default_param(distribution="custom", param_type=param_type)
            with self.assertRaises(ValueError):
                self.create_default_param(
                    distribution="custom",
                    custom_distribution=mock_distribution,
                    param_type=param_type,
                )
            custom_param = self.create_default_param(
                number_points=3,
                distribution="custom",
                custom_distribution=mock_distribution,
                param_type=param_type,
            )
            assert np.array_equal(custom_param.data_points, np.array((0.1, 0.5, 0.8)))
            with self.assertRaises(ValueError):
                self.create_default_param(param_type="something")

    def test_initial_data_points_within_range(self) -> None:
        """Test to ensure that initial data points are within the specified range for
        different distributions."""
        for param_type in ["continuous", "discrete"]:
            linear_param = self.create_default_param(
                distribution="linear", param_type=param_type, parameter_range=[1.0, 9.0]
            )
            self.assertEqual(len(linear_param.data_points), linear_param.number_points)
            self.assertAlmostEqual(linear_param.data_points[0], linear_param.range[0])
            self.assertAlmostEqual(linear_param.data_points[-1], linear_param.range[1])

            for dist in ["uniform", "normal", "log"]:
                dist_param = self.create_default_param(
                    distribution=dist, param_type=param_type, parameter_range=[1.0, 9.0]
                )
                self.assertEqual(len(dist_param.data_points), dist_param.number_points)
                self.assertGreaterEqual(
                    max(dist_param.data_points), dist_param.range[0]
                )
                self.assertLessEqual(min(dist_param.data_points), dist_param.range[1])

    def test_generate_data_points(self) -> None:
        """Test the generation of data points for a `Parameter`."""
        test_parameter = self.create_default_param(number_points=5)
        test_parameter.data_points = test_parameter.generate_data_points(num_points=3)
        self.assertEqual(len(test_parameter.data_points), 3)
        np.testing.assert_almost_equal(test_parameter.data_points, [0.1, 0.5, 0.9])

    def test_generate_dependent_data_points(self) -> None:
        """Test the generation of data points for a `Parameter` that depends on another
        parameter."""

        def linear_dep(x: float, y: float) -> float:
            """Linear function for parameter dependency."""
            return x * y

        param1 = self.create_default_param(
            name="param1",
            number_points=4,
            distribution="linear",
            parameter_range=[1, 4],
        )
        param2 = self.create_default_param(
            name="param2",
            number_points=4,
            distribution="linear",
            parameter_range=[1, 4],
            depends_on={"name": "param1", "function": linear_dep},
        )
        param_list = [param1, param2]
        param2.update_parameter_through_dependency(param_list)
        assert np.array_equal(param2.data_points, np.array((1, 4, 9, 16)))

        def fancy_dep(x: float, y: float) -> float:
            """More 'fancy' function for parameter dependency."""
            return float(2 * x**y)

        param3 = self.create_default_param(
            name="param3",
            number_points=4,
            distribution="linear",
            parameter_range=[1, 4],
            depends_on={"name": "param1", "function": fancy_dep},
        )
        param_list = [param1, param3]
        param3.update_parameter_through_dependency(param_list)
        assert np.array_equal(param3.data_points, np.array((2, 8, 54, 512)))

    def test_update_parameter_through_dependency(self) -> None:
        """Test the update of a `Parameter` that depends on another parameter."""
        # Create a mock experiment with parameters
        param_a = Parameter(
            "A",
            param_range=[1, 3],
            number_points=3,
            distribution="linear",
            constraints={"low": 0, "high": 5},
        )
        param_b = Parameter(
            "B",
            param_range=[4, 6],
            number_points=3,
            distribution="linear",
            constraints={"low": 0, "high": 10},
        )

        # Test when depends_on is not set
        with self.assertRaises(ValueError):
            param_a.update_parameter_through_dependency([param_a, param_b])

        # Set up dependency
        param_a.depends_on = {"name": "B", "function": lambda x, y: x + y}

        # Test the update
        param_a.update_parameter_through_dependency([param_a, param_b])

        # Check if the data_points were updated correctly
        self.assertTrue(isinstance(param_a.data_points, np.ndarray))
        np.testing.assert_array_equal(param_a.data_points, np.array([5, 7, 9]))

        # Check if the constraints were updated correctly
        self.assertEqual(param_a.constraints, {"low": 0, "high": 15})

    def test_is_active_property(self) -> None:
        """Test to verify the `is_active` property of a `Parameter`."""
        active_param = self.create_default_param(parameter_active=True)
        inactive_param = self.create_default_param(parameter_active=False)
        self.assertTrue(active_param.is_active)
        self.assertFalse(inactive_param.is_active)


class TestSystemSetup(unittest.TestCase):
    """A set of tests to validate the `SystemSetup` class from the `yotse.pre`
    package."""

    def setUp(self) -> None:
        """Hook method for setting up the test fixture before exercising it."""
        with open(DUMMY_FILE, "wt") as f:
            f.write("This is a dummy experiment file for test_pre.")

    def tearDown(self) -> None:
        """Hook method for deconstructing the test fixture after testing it."""
        os.remove(DUMMY_FILE)

    def test_invalid_directory_or_files(self) -> None:
        """Test to ensure that invalid directories or files are properly handled."""
        invalid_directory = "/invalid/source_directory"

        with self.assertRaises(ValueError):
            SystemSetup(
                invalid_directory, DUMMY_FILE, {"--arg1": 0.1, "--arg2": "value2"}
            )
        with self.assertRaises(ValueError):
            SystemSetup(os.getcwd(), DUMMY_FILE, analysis_script="non_existent_file.sh")
        # test correct setup
        SystemSetup(os.getcwd(), DUMMY_FILE, {"--arg1": 0.1, "--arg2": "value2"})

    def test_init(self) -> None:
        """Test the initialization of a `SystemSetup` instance with various
        configurations."""
        test_system = SystemSetup(
            source_directory=os.getcwd(),
            program_name=DUMMY_FILE,
            command_line_arguments={"--arg1": 0.1, "--arg2": "value2"},
            executor="bash",
            output_dir_name="out",
            output_extension="json",
            venv="test/test-env/",
            num_nodes=42,
            alloc_time="115:00:00",
            slurm_args=["--exclusive"],
            qcg_cfg={"init_timeout": 420},
            modules=["PYTHON3.10"],
        )
        assert test_system.source_directory == os.getcwd()
        assert test_system.program_name == os.path.join(os.getcwd(), DUMMY_FILE)
        assert test_system.cmdline_arguments == {"--arg1": 0.1, "--arg2": "value2"}
        assert test_system.analysis_script is None
        assert test_system.job_args["exec"] == "bash"
        assert test_system.output_dir_name == "out"
        assert test_system.output_extension == "json"
        assert test_system.venv == "test/test-env/"
        assert test_system.job_args["venv"] == "test/test-env/"
        assert test_system.num_nodes == 42
        assert test_system.alloc_time == "115:00:00"
        assert test_system.slurm_args == ["--exclusive"]
        assert test_system.qcg_cfg == {"init_timeout": 420}
        assert test_system.modules == ["PYTHON3.10"]

    def test_cmdline_to_list(self) -> None:
        """Test conversion of command line arguments from a dictionary to a list
        format."""

        test_setup = SystemSetup(
            os.getcwd(),
            DUMMY_FILE,
            {"--arg1": 0.1, "--arg2": "value2", "--arg3": False},
        )
        assert test_setup.cmdline_dict_to_list() == [
            "--arg1",
            0.1,
            "--arg2",
            "value2",
            "--arg3",
            False,
        ]


class TestExperiment(unittest.TestCase):
    """A set of tests to validate the `Experiment` class from the `yotse.pre`
    package."""

    def setUp(self) -> None:
        """Hook method for setting up the test fixture before exercising it."""
        with open(DUMMY_FILE, "wt") as f:
            f.write("This is a dummy experiment file for test_pre.")

    def tearDown(self) -> None:
        """Hook method for deconstructing the test fixture after testing it."""
        os.remove(DUMMY_FILE)

    @staticmethod
    def create_default_experiment(
        parameters: Optional[List[Parameter]] = None,
        optimization_info: Optional[List[OptimizationInfo]] = None,
    ) -> Experiment:
        """Static method to set up a default `Experiment` instance for testing."""
        return Experiment(
            experiment_name="default_exp",
            system_setup=SystemSetup(
                source_directory=os.getcwd(),
                program_name=DUMMY_FILE,
                command_line_arguments={"arg1": 0.1, "arg2": "value2"},
            ),
            parameters=parameters,
            opt_info_list=optimization_info,
        )

    def test_cost_function_setter_and_getter(self) -> None:
        """Test setting and getting the cost function in an `Experiment`."""
        test_exp = self.create_default_experiment()

        test_exp.cost_function = valid_function  # type: ignore[assignment]
        self.assertEqual(test_exp.cost_function, valid_function)

        # Test setting and getting a local function
        def main() -> None:
            """Mocking main with local function."""

            def local_function() -> None:
                """Mocked local function."""
                pass

            with self.assertRaises(ValueError):
                test_exp.cost_function = local_function  # type: ignore[assignment]

        main()

        # Test setting a non-function value
        with self.assertRaises(ValueError) as context:
            test_exp.cost_function = "not a function"  # type: ignore[assignment]
        self.assertEqual(
            str(context.exception), "Input cost_function is not a function."
        )

    def test_c_product(self) -> None:
        """Test the creation of a Cartesian product of data points from active
        `Parameter` instances."""
        test_exp = self.create_default_experiment()
        test_exp.add_parameter(
            Parameter(
                name="active_param1",
                param_range=[1, 3],
                number_points=3,
                distribution="linear",
                parameter_active=True,
            )
        )
        test_exp.add_parameter(
            Parameter(
                name="inactive_param",
                param_range=[11, 13],
                number_points=3,
                distribution="linear",
                parameter_active=False,
            )
        )
        test_exp.add_parameter(
            Parameter(
                name="active_param2",
                param_range=[21, 23],
                number_points=3,
                distribution="linear",
                parameter_active=True,
            )
        )
        test_exp.data_points = test_exp.create_datapoint_c_product()

        assert np.array_equal(
            test_exp.data_points,
            np.array(
                list(
                    itertools.product(
                        np.array([1.0, 2.0, 3.0]), np.array([21.0, 22.0, 23.0])
                    )
                )
            ),
        )
        # now activate 'inactive_param' and regenerate points
        test_exp.parameters[1].parameter_active = True
        assert test_exp.parameters[1].is_active
        test_exp.data_points = test_exp.create_datapoint_c_product()
        assert np.array_equal(
            test_exp.data_points,
            np.array(
                list(
                    itertools.product(
                        np.array((1.0, 2.0, 3.0)),
                        np.array((11.0, 12.0, 13.0)),
                        np.array((21.0, 22.0, 23.0)),
                    )
                )
            ),
        )
        # now deactivate 'active_param1' and 'active_param2' and regenerate points
        test_exp.parameters[0].parameter_active = False
        test_exp.parameters[2].parameter_active = False
        test_exp.data_points = test_exp.create_datapoint_c_product()
        assert test_exp.parameters[0].is_active is False
        assert test_exp.parameters[2].is_active is False
        assert np.array_equal(test_exp.data_points, np.array([[11.0], [12.0], [13.0]]))

    def test_add_parameter(self) -> None:
        """Test the addition of `Parameter` instances to an `Experiment`."""
        test_exp = self.create_default_experiment()
        self.assertEqual(len(test_exp.parameters), 0)
        test_param = TestParameters.create_default_param()
        test_exp.add_parameter(test_param)
        self.assertEqual(len(test_exp.parameters), 1)

    def test_add_optimization_information(self) -> None:
        """Test the addition of `OptimizationInfo` to an `Experiment`."""
        test_exp = self.create_default_experiment()
        self.assertEqual(len(test_exp.opt_info_list), 0)

        def linear_dep(x: float, y: float) -> float:
            """Linear function for analytical optimization."""
            return x * y

        test_opt = self.create_default_experiment(
            optimization_info=[
                OptimizationInfo(
                    name="GA",
                    blackbox_optimization=True,
                    opt_parameters={"pop_size": 5},
                    is_active=True,
                )
            ]
        )
        test_opt.add_optimization_info(
            OptimizationInfo(
                name="GD",
                blackbox_optimization=False,
                opt_parameters={"fun": linear_dep},
                is_active=False,
            )
        )

        self.assertEqual(len(test_opt.opt_info_list), 2)
        self.assertEqual(test_opt.opt_info_list[-1].name, "GD")

    def test_parse_slurm_arg(self) -> None:
        """Test the argument parsing for the SLURM script for an `Experiment`."""
        test_exp = self.create_default_experiment()

        # Mock sys.argv to simulate command-line arguments
        with patch("sys.argv", ["test_script.py", "--slurm"]):
            # Capture the standard output to check printed messages
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                # Use assertRaises to catch the SystemExit exception
                with self.assertRaises(SystemExit) as _:
                    test_exp.parse_slurm_arg("test_slurm.job")

        # Check if the expected success message is printed
        output = mock_stdout.getvalue()
        expected_output = (
            "\n"
            + "=" * 80
            + "\n"
            + "\033[1;92mSLURM execution script for test_slurm.job successfully created. Execute with 'sbatch slurm.job'.\033[0m"
            + "\n"
            + "=" * 80
            + "\n"
        )
        # Instead of using self.assertEqual(output, expected_output)
        self.assertMultiLineEqual(output.strip(), expected_output.strip())

        os.remove("slurm.job")

    def test_generate_slurm_script(self) -> None:
        """Test the generation of a default SLURM script for an `Experiment`."""
        test_exp = self.create_default_experiment()
        test_exp.system_setup.num_nodes = 42
        test_exp.system_setup.alloc_time = "01:00:00"
        test_exp.system_setup.venv = "test/test-env/"
        test_exp.system_setup.slurm_venv = "test/yotse-test-env/"
        test_exp.system_setup.slurm_args = ["--exclusive"]
        test_exp.system_setup.modules = ["2023", "Python/3.11.1"]

        test_exp.generate_slurm_script("test_pre.py")

        # Read the contents of the slurm.job file
        with open("slurm.job", "r") as file:
            script_contents = file.readlines()

        # Define the expected output
        expected_output = [
            "#!/bin/bash\n",
            "#SBATCH --nodes=42\n",
            "#SBATCH --exclusive\n",
            "#SBATCH --time=01:00:00\n",
            "\n",
            "\n",
            "module purge\n",
            "module load 2023\n",
            "module load Python/3.11.1\n",
            "source test/yotse-test-env/bin/activate\n",
            "\n",
            "python test_pre.py\n",
        ]

        # Compare the generated contents with the expected output line-by-line
        for line_num, (generated_line, expected_line) in enumerate(
            zip(script_contents, expected_output), start=1
        ):
            assert (
                generated_line == expected_line
            ), f"Line {line_num} of the generated slurm.job file does not match the expected output."

        # Ensure that the number of lines in the generated file matches the expected number of lines
        assert (
            len(script_contents) == len(expected_output)
        ), "The generated slurm.job file has a different number of lines than the expected output."

        os.remove("slurm.job")


if __name__ == "__main__":
    unittest.main()
