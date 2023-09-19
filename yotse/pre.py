"""Defines classes and functions for pre step."""
import argparse
import itertools
import os

import numpy as np


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
    constraints : dict or list(optional)
        Dictionary with constraints. Keys can be 'low', 'high' and 'step'. Alternatively list with acceptable values.
        Defaults to None.
    weights : list (optional)
        List of weights for the parameters, defaults to None.
    parameter_active: bool (optional)
        Whether this parameter should be used a varied parameter in this optimization step. Can be used to perform
        sequential optimization of different parameters with only one pre-step. Defaults to True.
    custom_distribution : function (optional if distribution!='custom')
        Custom distribution function that takes as arguments (min_value: float, max_value: float, number_points: int)
        and returns a list of float points.
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
    data_points : list
        Data points to be explored for this parameter.
    """
    def __init__(self, name: str, param_range: list, number_points: int, distribution: str, constraints=None,
                 weights=None, parameter_active=True, custom_distribution=None, param_type="continuous",
                 scale_factor=1., depends_on=None):
        self.name = name
        self.range = param_range
        self.range[0] = float(self.range[0])
        self.range[1] = float(self.range[1])
        self.number_points = number_points
        self.weights = weights
        if weights is not None:
            raise NotImplementedError("weights not implemented...yet.")
        self.constraints = constraints
        self.parameter_active = parameter_active
        self.data_points = []
        if custom_distribution is not None and distribution != 'custom':
            raise ValueError(f"Custom distribution supplied but distribution set to {distribution}!")
        self.distribution = distribution
        self.custom_distribution = custom_distribution
        self.param_type = param_type
        if scale_factor != 1.:
            raise NotImplementedError("scale_factor not implemented yet.")
        self.scale_factor = scale_factor
        self.depends_on = depends_on

        self.generate_initial_data_points()

    @property
    def is_active(self):
        return self.parameter_active

    def generate_data_points(self, num_points: int) -> None:
        """
        Generate set of n=num_points data points based on the specified distribution, range, and param_type of
        this parameter.

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
                self.data_points = np.linspace(self.range[0], self.range[1], num_points).tolist()
            elif self.distribution == "uniform":
                self.data_points = np.random.uniform(self.range[0], self.range[1], num_points).tolist()
            elif self.distribution == "normal":
                self.data_points = np.random.normal((self.range[0] + self.range[1]) / 2,
                                                    abs(self.range[1] - self.range[0]) / 3, num_points).tolist()
            elif self.distribution == "log":
                self.data_points = np.logspace(np.log10(self.range[0]), np.log10(self.range[1]), num_points).tolist()
            elif self.distribution == "custom" and self.custom_distribution is not None:
                self.data_points = self.custom_distribution(self.range[0], self.range[1], num_points)
                if len(self.data_points) != num_points:
                    raise ValueError(f'Custom distribution returned invalid number of points {len(self.data_points)}.')
                assert min(self.data_points) >= self.range[0]
                assert max(self.data_points) <= self.range[1]
            else:
                raise ValueError(f"Invalid distribution specified: {self.distribution} for continuous parameter.")
        elif self.param_type == "discrete":
            if self.distribution == "linear":
                self.data_points = np.linspace(self.range[0], self.range[1], num_points, dtype=int).tolist()
            elif self.distribution == "uniform":
                self.data_points = np.random.randint(self.range[0], self.range[1] + 1, num_points).tolist()
            elif self.distribution == "normal":
                self.data_points = np.random.normal((self.range[0] + self.range[1]) / 2,
                                                    abs(self.range[1] - self.range[0]) / 3, num_points)
                # for discrete normal distribution round floats to the nearest int
                self.data_points = np.round(self.data_points).astype(int).tolist()
            elif self.distribution == "log":
                # self.data_points = np.unique(
                #     np.geomspace(self.range[0], self.range[1], num_points, dtype=int)).tolist()
                self.data_points = np.logspace(np.log10(self.range[0]), np.log10(self.range[1]), num_points,
                                               dtype=int).tolist()
            elif self.distribution == "custom" and self.custom_distribution is not None:
                self.data_points = self.custom_distribution(self.range[0], self.range[1], num_points)
                if len(self.data_points) != num_points:
                    raise ValueError(
                        f'Custom distribution returned invalid number of points {len(self.data_points)}.')
                assert all([self.range[0] <= p <= self.range[1] for p in self.data_points])
            else:
                raise ValueError(f"Invalid distribution specified: {self.distribution} for discrete parameter.")
        else:
            raise ValueError(f"Invalid parameter type specified: {self.param_type}")

    def generate_initial_data_points(self) -> None:
        """Generate initial data points based on the specified distribution and range."""
        self.generate_data_points(num_points=self.number_points)

    def generate_dependent_data_points(self, parameter_list: list) -> None:
        """
        Generate data points for this parameter based on another parameter's data points and adjust constraints.

        Parameters
        ----------
        parameter_list : list
            List of (all) Parameter objects in the experiment. Should at least contain the parameter that this
            parameter depends on.

        Notes
        -----
        # todo : this will only be applied once before the start of the experiment. Is that useful?
        """
        target_parameter = [param for param in parameter_list if param.name == self.depends_on['name']][0]

        new_data_points = [self.depends_on['function'](a, b) for a, b in zip(self.data_points,
                                                                             target_parameter.data_points)]
        if self.constraints is not None:
            try:
                self.constraints['low'] = self.depends_on['function'](self.constraints['low'],
                                                                      target_parameter.constraints['low'])
                self.constraints['high'] = self.depends_on['function'](self.constraints['high'],
                                                                       target_parameter.constraints['high'])
            except KeyError:
                pass

        self.data_points = new_data_points


class SystemSetup:
    """Defines a class for the setup of the system parameters.

    Parameters:
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
    def __init__(self, source_directory: str, program_name: str, command_line_arguments: dict = None,
                 analysis_script: str = None, executor: str = "python", output_dir_name: str = None,
                 output_extension: str = 'csv', venv: str = None, num_nodes: int = 1, alloc_time: str = '00:15:00',
                 slurm_args: list = None, qcg_cfg: dict = None, modules: list = None):
        if not os.path.exists(source_directory):
            raise ValueError(f"Invalid source_directory path: {source_directory}")
        if not os.path.exists(os.path.join(source_directory, program_name)):
            raise ValueError(f"Invalid program_name: {os.path.join(source_directory, program_name)} is not a file.")
        if analysis_script is not None:
            if not os.path.exists(os.path.join(source_directory, analysis_script)):
                raise ValueError(f"Invalid analysis_script:"
                                 f" {os.path.join(source_directory, analysis_script)} is not a file.")

        self.source_directory = source_directory
        self.program_name = os.path.join(source_directory, program_name)
        self.cmdline_arguments = command_line_arguments or {}
        # replace paths in cmdline args with absolute paths
        for key, value in self.cmdline_arguments.items():
            if isinstance(value, str) and os.path.splitext(value)[1]:
                self.cmdline_arguments[key] = os.path.join(source_directory, value)
        self.analysis_script = os.path.join(source_directory, analysis_script) if analysis_script is not None else None
        self.job_args = {"exec": executor}
        self.output_dir_name = output_dir_name or 'output'
        self.output_extension = output_extension
        self.stdout_basename = 'stdout'
        self.working_directory = None
        if venv is not None:
            self.job_args["venv"] = venv
            self.venv = venv
        self.num_nodes = num_nodes
        self.alloc_time = alloc_time
        self.slurm_args = slurm_args
        self.qcg_cfg = qcg_cfg
        self.modules = modules

    @property
    def current_step_directory(self) -> str:
        """Returns the path of the current optimization step."""
        if self.working_directory is not None:
            return os.path.realpath(os.path.join(self.working_directory, '..'))
        else:
            raise RuntimeError(f"Could not get current step directory. Working directory is {self.working_directory}")

    def cmdline_dict_to_list(self) -> list:
        """Convert the dictionary of commandline arguments to a list for QCGPilot."""
        return [item for key_value_pair in self.cmdline_arguments.items() for item in key_value_pair]


class OptimizationInfo:
    """Class that is optional as input to the Experiment, if the run is supposed to execute an optimization it will
    look here for the parameters.
    Parameters
    ----------
    name : str
        Name of the optimization algorithm to be used, e.g. "GA" (genetic algorithm), "GD" (gradient descent).
    opt_parameters : dict
        Dictionary containing all necessary parameters for the optimization.
    is_active : bool
        Whether this is the currently active optimization algorithm. Can be used to perform sequential optimization with
        different optimization algorithms that can all be defined in a single Experiment.
    """

    def __init__(self, name, opt_parameters, is_active):
        self.name = name
        self.parameters = opt_parameters
        self.is_active = is_active


class Experiment:
    """Class that contains the whole experiment, including ExperimentalSystemSetup, all Parameters and a list of
    optimization steps.
    Parameters
    ----------
    experiment_name : str
        Descriptive name for the experiment.
    system_setup : SystemSetup
        Instance of the SystemSetup class that contains the setup of the experimental system.
    parameters : list[Parameter] (optional)
        List of Parameter instances that define the parameters to be varied in the experiment.
        Defaults to an empty list.
        Note: If one wants to first optimize over a subset of parameters then set the remaining parameters as inactive
        `for param not in params_to_opt_over: param.parameter_active = False`. To later also optimize over the other
        subset just set them to active again.
    opt_info_list : list (optional)
         List of :obj:OptimizationInfo describing the different optimization algorithms to be used and their parameters.
         Defaults to an empty list.

    Attributes
    ----------
    _current_optimization_step : int
        The current optimization step number.
    """

    def __init__(self, experiment_name: str, system_setup: SystemSetup, parameters=None, opt_info_list=None):
        self.name = experiment_name
        self.system_setup = system_setup
        self.parameters = parameters or []
        self.optimization_information_list = []
        if opt_info_list is not None:
            for item in opt_info_list:
                if not isinstance(item, OptimizationInfo):
                    raise ValueError(f"Items in opt_info_list should be of type OptimizationInfo not {type(item)}.")
            self.optimization_information_list = list(opt_info_list)
        self.data_points = []
        # set initial datapoints
        self.create_datapoint_c_product()
        self.cost_function = None

    def create_datapoint_c_product(self) -> None:
        """Create initial set of points as Cartesian product of all active parameters.

        Overwrite if other combination is needed."""
        if self.parameters:
            assert isinstance(self.parameters, list), "Parameters are not list."
            for param in self.parameters:
                if not isinstance(param, Parameter):
                    raise TypeError(f"One of the parameters is not of correct type 'Parameter', but is {type(param)}")
                if param.depends_on is not None:
                    param.generate_dependent_data_points(self.parameters)
            active_params = [param for param in self.parameters if param.is_active]
            if len(active_params) == 1:
                # single param -> no cartesian product
                self.data_points = active_params[0].data_points
            else:
                self.data_points = list(itertools.product(*[param.data_points for param in active_params]))

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
        self.optimization_information_list.append(optimization_info)

    def generate_slurm_script(self, filename):
        """Generate slurm script to execute the file through slurm.

        Note: after the file has been created the process can be started by calling `sbatch slurm.job`

        Parameters
        ----------
        filename : str (optional)
            Name of the file to be executed through SLURM
        """
        if self.system_setup.num_nodes is None:
            raise ValueError("Slurm script can not be generated without num_nodes.")
        if self.system_setup.alloc_time is None:
            raise ValueError("Slurm script can not be generated without alloc_time.")

        script = f"!/bin/bash\n#SBATCH --nodes={self.system_setup.num_nodes}\n"
        if self.system_setup.slurm_args is not None:
            for slurm_arg in self.system_setup.slurm_args:
                script += f"#SBATCH {slurm_arg}\n"
        script += f"#SBATCH --time={self.system_setup.alloc_time}\n\n\n"
        script += "module purge\n"
        if self.system_setup.modules is not None:
            for module in self.system_setup.modules:
                script += f"module load {module}\n"
        if self.system_setup.venv is not None:
            script += f"source {os.path.join(self.system_setup.venv,'bin/activate')}\n\n"
        script += f"python {filename}\n"

        with open(os.path.join(self.system_setup.source_directory, "slurm.job"), "w") as file:
            file.write(script)

    def parse_slurm_arg(self, filename):
        parser = argparse.ArgumentParser()
        parser.add_argument("--slurm", action="store_true", help="Generate slurm.job file")
        args = parser.parse_args()
        if args.slurm:
            self.generate_slurm_script(filename)
            exit()

    def qcgpilot_commandline(self, datapoint_item: list) -> list:
        """
         Creates a command line for the QCG-PilotJob executor based on the experiment configuration.

         Parameters:
         -----------
         experiment: Experiment
             The experiment to configure the command line for.
        datapoint_item : list or float #todo : fix this so it always gets a list?
            Datapoint containing the specific values for each parameter e.g. (x1, y2, z1).

         Returns:
         --------
         list
             A list of strings representing the command line arguments for the QCG-PilotJob executor.
         """
        cmdline = [os.path.join(self.system_setup.source_directory, self.system_setup.program_name)]
        # add parameters
        for p, param in enumerate(self.parameters):
            if param.is_active:
                cmdline.append(f"--{param.name}")
                if len(self.parameters) == 1:
                    # single parameter
                    cmdline.append(datapoint_item)
                else:
                    cmdline.append(datapoint_item[p])
        # add fixed cmdline arguments
        for key, value in self.system_setup.cmdline_arguments.items():
            cmdline.append(key)
            cmdline.append(str(value))
        return cmdline


def set_basic_directory_structure_for_job(experiment: Experiment, step_number: int, job_number: int) -> None:
    """
    Creates a new directory for the given step number and updates the experiment's working directory accordingly.

    The basic directory structure is as follows
    source_dir
        - output_dir
            your_run_script.py
            analysis_script.py
            - step_{i}
                 analysis_output.csv
                - job_{j}
                    output_of_your_run_script.extension
                    stdout{j}.txt

    Parameters:
    ----------
    experiment : Experiment
        The :obj:Experiment that is being run.
    step_number : int
        The number of the current step.
    job_number : int
        The number of the current job.
    """
    source_dir = experiment.system_setup.source_directory
    output_dir = experiment.system_setup.output_dir_name
    new_working_dir = os.path.join(source_dir, output_dir, f'step{step_number}', f'job{job_number}')

    if not os.path.exists(new_working_dir):
        os.makedirs(new_working_dir)
    experiment.system_setup.working_directory = new_working_dir
