"""Defines classes and functions for the setup of your experiment."""
from __future__ import annotations

import argparse
import inspect
import itertools
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TypedDict
from typing import Union

import numpy as np


class ParameterDependencyDict(TypedDict):
    """Data structure to explicitly specify how to define parameter dependencies.

    Parameters
    ----------
    name: str
        Name of the parameter the initial parameter depends on. E.g. if `fidelity` depends on `time`, name = 'time'.
    function : Callable
        Dependency function of the form function(parameter_value: float, parameter_it_depends_on_value: float) -> float
    """

    name: str
    function: Callable[[float, float], float]


class ConstraintDict(TypedDict, total=False):
    """Data structure to define constraints on parameter values.

    Parameters
    ----------
    low : float, optional
        The lower bound for the parameter.
    high : float, optional
        The upper bound for the parameter.
    step : float, optional
        The step size for the parameter.
    """

    low: float
    high: float
    step: Optional[float]


class Parameter:
    """Defines a class for any type of parameter we want to vary in our simulation.

    Parameters
    ----------
    name : str
        Name of the parameter.
    param_range : list(min, max)
        List with the min and max value of the parameter; min and max are floats.
    number_points: int
        Number of points to be explored.
    distribution : str
        Type of distribution of the points. Currently supports 'linear', 'uniform', 'normal', 'log' or 'custom'.
        If 'custom' is specified then parameter custom_distribution is required.
    constraints : dict or np.ndarray (optional)
        Dictionary with constraints. Keys can be 'low', 'high' and 'step'. Alternatively np.ndarray with acceptable
        values. Defaults to None.
    weights : list (optional)
        List of weights for the parameters, defaults to None.
    parameter_active: bool (optional)
        Whether this parameter should be used a varied parameter in this optimization step. Can be used to perform
        sequential optimization of different parameters with only one pre-step. Defaults to True.
    custom_distribution : function (optional if distribution!='custom')
        Custom distribution function that takes as arguments (min_value: float, max_value: float, number_points: int)
        and returns a Tuple of float points.
        Defaults to None.
    param_type: str (optional)
        Type of parameter: 'discrete' or 'continuous'.
         Defaults to 'continuous'.
    scale_factor : float (optional)
        Scale factor to apply to the parameter when generating new points.
        Defaults to 1.0.
    # todo : check if this is what we intend to do actually?
    depends_on : dict (optional)
        Dictionary containing the two keys 'name' and 'function', specifying the parameter name it depends on and a
        function of the form function(parameter_value: float, parameter_it_depends_on_value: float) -> float.
        Defaults to None.

    Attributes
    ----------
    data_points : np.ndarray (1D)
        Data points for this parameter, stored in an np.ndarray for efficient computation and memory usage.
    """

    def __init__(
        self,
        name: str,
        param_range: List[Union[float, int]],
        number_points: int,
        distribution: str,
        constraints: Union[ConstraintDict, np.ndarray, None] = None,
        weights: Optional[List[float]] = None,
        parameter_active: bool = True,
        custom_distribution: Optional[Callable[[float, float, int], np.ndarray]] = None,
        param_type: str = "continuous",
        scale_factor: float = 1.0,
        depends_on: Optional[ParameterDependencyDict] = None,
    ):
        """Initialize `Parameter` object."""
        self.name = name
        self.range = param_range
        self.range = [float(self.range[0]), float(self.range[1])]
        self.number_points = number_points
        self.weights = weights
        if weights is not None:
            raise NotImplementedError("weights not implemented...yet.")
        self.constraints = constraints
        self.parameter_active = parameter_active
        self.data_points: np.array = np.array(())
        if custom_distribution is not None and distribution != "custom":
            raise ValueError(
                f"Custom distribution supplied but distribution set to {distribution}!"
            )
        self.distribution = distribution
        self.custom_distribution = custom_distribution
        self.param_type = param_type
        if scale_factor != 1.0:
            raise NotImplementedError("scale_factor not implemented yet.")
        self.scale_factor = scale_factor
        self.depends_on = depends_on

        self.data_points = self.generate_initial_data_points()

    @property
    def is_active(self) -> bool:
        """Whether this parameter is active (=used for the current optimization)."""
        return self.parameter_active

    def generate_data_points(self, num_points: int) -> np.ndarray:
        """Generate set of n=num_points data points based on the specified distribution,
        range, and param_type of this parameter.

        Parameters
        ----------
        num_points : int
            Number of datapoints to generate.

        Notes
        -----
        - data_points are not sorted.
        - data_points are not guaranteed to be unique.
        """
        if self.param_type == "continuous":
            if self.distribution == "linear":
                data_points = np.linspace(self.range[0], self.range[1], num_points)
            elif self.distribution == "uniform":
                data_points = np.random.uniform(
                    self.range[0], self.range[1], num_points
                )
            elif self.distribution == "normal":
                data_points = np.random.normal(
                    (self.range[0] + self.range[1]) / 2,
                    abs(self.range[1] - self.range[0]) / 3,
                    num_points,
                )
            elif self.distribution == "log":
                data_points = np.logspace(
                    np.log10(self.range[0]), np.log10(self.range[1]), num_points
                )
            elif self.distribution == "custom" and self.custom_distribution is not None:
                data_points = self.custom_distribution(
                    self.range[0], self.range[1], num_points
                )
                if len(data_points) != num_points:
                    raise ValueError(
                        f"Custom distribution returned invalid number of points {len(data_points)}."
                    )
                assert min(data_points) >= self.range[0]
                assert max(data_points) <= self.range[1]
            else:
                raise ValueError(
                    f"Invalid distribution specified: {self.distribution} for continuous parameter."
                )
        elif self.param_type == "discrete":
            if self.distribution == "linear":
                data_points = np.linspace(
                    self.range[0], self.range[1], num_points, dtype=int
                )
            elif self.distribution == "uniform":
                data_points = np.random.randint(
                    int(self.range[0]), int(self.range[1]) + 1, num_points
                )
            elif self.distribution == "normal":
                data_points = np.random.normal(
                    (self.range[0] + self.range[1]) / 2,
                    abs(self.range[1] - self.range[0]) / 3,
                    num_points,
                )
                # for discrete normal distribution round floats to the nearest int
                data_points = np.round(data_points).astype(int)
            elif self.distribution == "log":
                # data_points = np.unique(
                #     np.geomspace(self.range[0], self.range[1], num_points, dtype=int))
                data_points = np.logspace(
                    np.log10(self.range[0]),
                    np.log10(self.range[1]),
                    num_points,
                    dtype=int,
                )
            elif self.distribution == "custom" and self.custom_distribution is not None:
                data_points = self.custom_distribution(
                    self.range[0], self.range[1], num_points
                )
                if len(data_points) != num_points:
                    raise ValueError(
                        f"Custom distribution returned invalid number of points {len(data_points)}."
                    )
                assert all([self.range[0] <= p <= self.range[1] for p in data_points])
            else:
                raise ValueError(
                    f"Invalid distribution specified: {self.distribution} for discrete parameter."
                )
        else:
            raise ValueError(f"Invalid parameter type specified: {self.param_type}")

        return data_points

    def generate_initial_data_points(self) -> np.ndarray:
        """Generate initial data points based on the specified distribution and
        range."""
        return self.generate_data_points(num_points=self.number_points)

    def update_parameter_through_dependency(
        self, parameter_list: List[Parameter]
    ) -> None:
        """Update data points and constraints for this parameter based on another
        parameter's data points and constraints.

        Parameters
        ----------
        parameter_list : list
            List of (all) Parameter objects in the experiment. Should at least contain the parameter that this
            parameter depends on.

        Notes
        -----
        # todo : this will only be applied once before the start of the experiment. Is that useful?
        """
        if self.depends_on is None:
            raise ValueError(
                "self.depends_on is None, but it needs to be a dict to proceed."
            )

        target_parameter = [
            param for param in parameter_list if param.name == self.depends_on["name"]
        ][0]
        # update data points
        new_data_points = [
            self.depends_on["function"](a, b)
            for a, b in zip(self.data_points, target_parameter.data_points)
        ]
        # update constraints
        if self.constraints is not None:
            if isinstance(self.constraints, dict) and isinstance(
                target_parameter.constraints, dict
            ):
                self.constraints["low"] = self.depends_on["function"](
                    self.constraints["low"], target_parameter.constraints["low"]
                )
                self.constraints["high"] = self.depends_on["function"](
                    self.constraints["high"], target_parameter.constraints["high"]
                )
                try:
                    if (
                        self.constraints["step"] is not None
                        and target_parameter.constraints["step"] is not None
                    ):
                        # step is not necessary to specify
                        self.constraints["step"] = self.depends_on["function"](
                            self.constraints["step"],
                            target_parameter.constraints["step"],
                        )
                except KeyError:
                    pass
            elif isinstance(self.constraints, tuple) and isinstance(
                target_parameter.constraints, tuple
            ):
                updated_constraints: List[float] = []
                if len(self.constraints) != len(target_parameter.constraints):
                    raise ValueError(
                        "Can not compute dependency for parameter with different number of allowed values."
                    )
                for i, value in enumerate(self.constraints):
                    updated_constraints.append(
                        self.depends_on["function"](
                            value, target_parameter.constraints[i]
                        )
                    )
                self.constraints = tuple(updated_constraints)
            else:
                raise ValueError(
                    f"For parameters that depend on each other the constraints must be of the same type and not "
                    f"{type(self.constraints)} and {type(target_parameter.constraints)}."
                )

        self.data_points = np.array(new_data_points)


class SystemSetup:
    """Defines a class for the setup of the system parameters.

    Parameters
    ----------
    source_directory : str
        Path of the source directory.
    program_name : str
        Name of the script that should be used for the experiment.
    command_line_arguments : dict (optional)
        Dictionaries containing as keys the reference of the line argument and as values their value. Defaults to an
        empty dictionary.
    analysis_script : str (optional)
        Name of the script that is used to analyse the output of the program script. Defaults to None.
    executor : str (optional)
        Executor to be passed when submitting jobs. Defaults to 'python'.
    output_dir_name : str (optional)
        Name of the directory the output should be stored in. Defaults to 'output'.
    output_extension : str (optional)
        Extension of the output files to be picked up by the analysis_script, e.g 'csv' or 'json'. Defaults to 'csv'.
    venv : str (optional)
        Path to the virtual environment that should be initialized before the QCGPilot job is started. Defaults to None.
    slurm_venv : str (optional)
        Path to the environment that slurm should activate before executing yotse. This needs to have yotse installed.
        Defaults to None.
    num_nodes : int (optional)
        Number of nodes to allocate on the HPC cluster. Defaults to 1.
    alloc_time : str (optional)
        Time to allocate on the HPC cluster in the format HH:MM:SS (or HHH:MM:SS and so forth). Defaults to '00:15:00'.
    slurm_args : list (optional)
        Additional arguments to pass to SLURM, e.g. '--exclusive'. Defaults to None
    qcg_cfg : dict (optional)
        Configuration to pass to the QCG-PilotJob manager. Dict with supported keys 'init_timeout', 'poll_delay',
        'log_file', 'log_level'. See docstring of `qcg.pilotjob.api.manager.LocalManager`. If None QCG defaults are
        used. Defaults to None.
    modules : list (optional)
        Modules to load on the HPC cluster. Defaults to None.

    Attributes:
    ----------
    stdout_basename : str
        Basename of the file that the standard output steam (stdout) of the script should be written to.
        The final filename will be of the form '<stdout_basename><unique_job_identifier>.txt'. O
    working_directory : str
        Name of the current working directory to be passed to QCGPilotJob.
    """

    def __init__(
        self,
        source_directory: str,
        program_name: str,
        command_line_arguments: Optional[Dict[str, Any]] = None,
        analysis_script: Optional[str] = None,
        executor: str = "python",
        output_dir_name: Optional[str] = None,
        output_extension: str = "csv",
        venv: Optional[str] = None,
        slurm_venv: Optional[str] = None,
        num_nodes: int = 1,
        alloc_time: str = "00:15:00",
        slurm_args: Optional[List[str]] = None,
        qcg_cfg: Optional[Dict[str, Union[str, int]]] = None,
        modules: Optional[List[str]] = None,
    ):
        """Initialize `SystemSetup` object."""
        if not os.path.exists(source_directory):
            raise ValueError(f"Invalid source_directory path: {source_directory}")
        if not os.path.exists(os.path.join(source_directory, program_name)):
            raise ValueError(
                f"Invalid program_name: {os.path.join(source_directory, program_name)} is not a file."
            )
        if analysis_script is not None:
            if not os.path.exists(os.path.join(source_directory, analysis_script)):
                raise ValueError(
                    f"Invalid analysis_script:"
                    f" {os.path.join(source_directory, analysis_script)} is not a file."
                )
        if output_extension not in ["csv", "json", "pickle"]:
            raise NotImplementedError(
                f"`output_extension`={output_extension} not implemented yet."
            )

        self.source_directory = source_directory
        self.program_name = os.path.join(source_directory, program_name)
        self.cmdline_arguments = command_line_arguments or {}
        # replace paths in cmdline args with absolute paths
        for key, value in self.cmdline_arguments.items():
            if isinstance(value, str) and os.path.splitext(value)[1]:
                self.cmdline_arguments[key] = os.path.join(source_directory, value)
        self.analysis_script = (
            os.path.join(source_directory, analysis_script)
            if analysis_script is not None
            else None
        )
        self.job_args = {"exec": executor}
        self.output_dir_name = output_dir_name or "output"
        self.output_extension = output_extension
        self.stdout_basename = "stdout"
        self.working_directory: str = ""
        if venv is not None:
            self.job_args["venv"] = venv
            self.venv = venv
        self.num_nodes = num_nodes
        self.alloc_time = alloc_time
        self.slurm_venv = slurm_venv
        self.slurm_args = slurm_args
        self.qcg_cfg = qcg_cfg
        self.modules = modules

    @property
    def current_step_directory(self) -> str:
        """Returns the path of the current optimization step."""
        if self.working_directory is not None:
            return os.path.realpath(os.path.join(self.working_directory, ".."))
        else:
            raise RuntimeError(
                f"Could not get current step directory. Working directory is {self.working_directory}"
            )

    def cmdline_dict_to_list(self) -> List[Union[str, int, float]]:
        """Convert the dictionary of commandline arguments to a list for QCGPilot."""
        return [
            item
            for key_value_pair in self.cmdline_arguments.items()
            for item in key_value_pair
        ]


class OptimizationInfo:
    """Class that is optional as input to the Experiment, if the run is supposed to
    execute an optimization it will look here for the parameters.

    Parameters
    ----------
    name : str
        Name of the optimization algorithm to be used, e.g. "GA" (genetic algorithm), "GD" (gradient descent).
    blackbox_optimization : bool
        Whether the optimization should be a black-box optimization. (If False: a function must be supplied.)
    opt_parameters : dict
        Dictionary containing all necessary parameters for the optimization.
    is_active : bool
        Whether this is the currently active optimization algorithm. Can be used to perform sequential optimization with
        different optimization algorithms that can all be defined in a single Experiment.
    function : callable, optional
        The objective function to be optimized. Required if blackbox_optimization is False.
    """

    def __init__(
        self,
        name: str,
        blackbox_optimization: bool,
        opt_parameters: Dict[str, Any],
        is_active: bool,
    ):
        """Initialize `OptimizationInfo` object."""
        self.name = name
        self.blackbox_optimization = blackbox_optimization
        self.opt_parameters = opt_parameters if opt_parameters else {}
        self.is_active = is_active


class Experiment:
    """Class that contains the whole experiment, including ExperimentalSystemSetup, all
    Parameters and a list of optimization steps.

    Parameters
    ----------
    experiment_name : str
        Descriptive name for the experiment.
    system_setup : SystemSetup
        Instance of the SystemSetup class that contains the setup of the experimental system.
    parameters : list of Parameter, optional
        List of Parameter instances that define the parameters to be varied in the experiment.
        Defaults to an empty list.
        Note: If one wants to first optimize over a subset of parameters then set the remaining parameters as inactive
        `for param not in params_to_opt_over: param.parameter_active = False`. To later also optimize over the
        other subset just set them to active again.
    opt_info_list : list, optional
        List of OptimizationInfo objects describing the different optimization algorithms to be used and their
        parameters.
        Defaults to an empty list.

    Attributes
    ----------
    data_points : ndarray
        MxN array where M is the total number of combinations of parameter data points and N is the number of
        parameters. Each row represents a unique combination of parameter values to be explored in the experiment.
    """

    def __init__(
        self,
        experiment_name: str,
        system_setup: SystemSetup,
        parameters: Optional[List[Parameter]] = None,
        opt_info_list: Optional[List[OptimizationInfo]] = None,
    ):
        """Initialize `Experiment` object."""
        self.name = experiment_name
        self.system_setup = system_setup
        self.parameters = parameters or []
        self.opt_info_list = []
        if opt_info_list is not None:
            for item in opt_info_list:
                if not isinstance(item, OptimizationInfo):
                    raise ValueError(
                        f"Items in opt_info_list should be of type OptimizationInfo not {type(item)}."
                    )
            self.opt_info_list = list(opt_info_list)
        # todo: to avoid confusion maybe it is useful to call the datapoints of the exp different than those of params
        self.data_points: np.ndarray = self.create_datapoint_c_product()
        self._cost_function: Optional[Callable[..., float]] = None

    @property
    def cost_function(self) -> Union[Callable[..., float], None]:
        """Cost function of the experiment."""
        return self._cost_function

    @cost_function.setter
    def cost_function(self, func: Callable[..., float]) -> None:
        """Set cost function of the experiment."""
        if inspect.isfunction(func):
            if "<locals>" in func.__qualname__:
                raise ValueError(
                    "Local functions are not supported for serialization by pickle. Therefore your"
                    " optimization will not be able to save its state. Please define your cost_function"
                    " outside the main() function of your script."
                )
                # todo: or should we just switch to dill instead of pickle?
        else:
            raise ValueError("Input cost_function is not a function.")
        self._cost_function = func

    def create_datapoint_c_product(self) -> np.ndarray:
        """Create initial set of points as Cartesian product of all active parameters.

        Overwrite if other combination is needed.
        """
        if self.parameters:
            assert isinstance(self.parameters, list), "Parameters are not list."
            for param in self.parameters:
                if not isinstance(param, Parameter):
                    raise TypeError(
                        f"One of the parameters is not of correct type 'Parameter', but is {type(param)}"
                    )
                if param.depends_on is not None:
                    param.update_parameter_through_dependency(self.parameters)
            active_params = [param for param in self.parameters if param.is_active]
            if len(active_params) == 1:
                # single param -> reshape to 2D array where each row is a single data point
                data_points = active_params[0].data_points.reshape(-1, 1)
            else:
                # multiple params -> cartesian product and conversion to np.ndarray
                data_points_list = list(
                    itertools.product(*[param.data_points for param in active_params])
                )
                data_points = np.array(data_points_list)
            return data_points
        else:
            return np.array(())

    def add_parameter(self, parameter: Parameter) -> None:
        """Adds a parameter to the experiment.

        Parameters
        ----------
        parameter : Parameter
            The parameter to add to the experiment.
        """
        if not isinstance(parameter, Parameter):
            raise TypeError("Can not add parameter that is not of type Parameter.")
        self.parameters.append(parameter)

    def add_optimization_info(self, optimization_info: OptimizationInfo) -> None:
        """Adds OptimizationInfo to the experiment.

        Parameters
        ----------
        optimization_info : OptimizationInfo
            The optimization step to add to the experiment.
        """
        if not isinstance(optimization_info, OptimizationInfo):
            raise TypeError("Can not add parameter that is not of type Parameter.")
        self.opt_info_list.append(optimization_info)

    def generate_slurm_script(self, filename: str) -> None:
        """Generate slurm script to execute the file through slurm.

        Note: after the file has been created the process can be started by calling `sbatch slurm.job`

        Parameters
        ----------
        filename : str (optional)
            Name of the file to be executed through SLURM. Note this is not the filename of the SLURM script itself.
        """
        if self.system_setup.num_nodes is None:
            raise ValueError("Slurm script can not be generated without num_nodes.")
        if self.system_setup.alloc_time is None:
            raise ValueError("Slurm script can not be generated without alloc_time.")

        script = f"#!/bin/bash\n#SBATCH --nodes={self.system_setup.num_nodes}\n"
        if self.system_setup.slurm_args is not None:
            for slurm_arg in self.system_setup.slurm_args:
                script += f"#SBATCH {slurm_arg}\n"
        script += f"#SBATCH --time={self.system_setup.alloc_time}\n\n\n"
        script += "module purge\n"
        if self.system_setup.modules is not None:
            for module in self.system_setup.modules:
                script += f"module load {module}\n"
        if self.system_setup.slurm_venv is not None:
            script += f"source {os.path.join(self.system_setup.slurm_venv,'bin/activate')}\n\n"
        script += f"python {filename}\n"

        with open(
            os.path.join(self.system_setup.source_directory, "slurm.job"), "w"
        ) as file:
            file.write(script)

    def parse_slurm_arg(self, filename: str) -> None:
        """Parse command-line arguments to determine if a SLURM script should be
        generated.

        Parameters
        ----------
        filename : str
            The filename of the script to be executed with SLURM.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--slurm", action="store_true", help="Generate slurm.job file"
        )
        args = parser.parse_args()
        if args.slurm:
            self.generate_slurm_script(filename)
            border = "=" * 80
            print("\n" + border)
            print(
                f"\033[1;92mSLURM execution script for {filename} successfully created. Execute with 'sbatch slurm.job'.\033[0m"
            )
            print(border + "\n")
            exit()

    def qcgpilot_commandline(self, datapoint_item: List[Any]) -> List[Union[str, Any]]:
        """Creates a command line for the QCG-PilotJob executor based on the experiment
        configuration.

        Parameters
        ----------
        datapoint_item : List[float]
            Datapoint containing the specific values for each parameter e.g. (x1, y2, z1).

        Returns
        -------
        list
            A list of strings representing the command line arguments for the QCG-PilotJob executor.
        """
        cmdline = [
            os.path.join(
                self.system_setup.source_directory, self.system_setup.program_name
            )
        ]
        # add parameters
        inactive_params_skipped = 0
        for p, param in enumerate(self.parameters):
            if param.is_active:
                cmdline.append(f"--{param.name}")
                # datapoint item contains only entries of active parameters
                cmdline.append(datapoint_item[p - inactive_params_skipped])
            else:
                inactive_params_skipped += 1
        # add fixed cmdline arguments
        for key, value in self.system_setup.cmdline_arguments.items():
            cmdline.append(key)
            cmdline.append(str(value))
        return cmdline


def set_basic_directory_structure_for_job(
    experiment: Experiment, step_number: int, job_number: int
) -> None:
    """Creates a new directory for the given step number and updates the experiment's
    working directory accordingly.

    The basic directory structure is as follows::

        source_dir/
        ├── output_dir/
        │   ├── your_run_script.py
        │   ├── analysis_script.py
        │   └── step_{i}/
        │       ├── analysis_output.csv
        │       └── job_{j}/
        │           ├── output_of_your_run_script.extension
        │           └── stdout{j}.txt

    Parameters
    ----------
    experiment : Experiment
        The :obj:`Experiment` that is being run.
    step_number : int
        The number of the current step.
    job_number : int
        The number of the current job.
    """
    source_dir = experiment.system_setup.source_directory
    output_dir = experiment.system_setup.output_dir_name
    new_working_dir = os.path.join(
        source_dir, output_dir, f"step{step_number}", f"job{job_number}"
    )

    if not os.path.exists(new_working_dir):
        os.makedirs(new_working_dir)
    experiment.system_setup.working_directory = new_working_dir
