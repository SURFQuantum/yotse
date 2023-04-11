"""Defines classes and functions for pre step."""
import os
import numpy as np
import itertools


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
    constraints : list (optional)
        List of constraints, defaults to None.
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

    Attributes
    ----------
    data_points : list
        Data points to be explored for this parameter.
    """
    def __init__(self, name: str, param_range: list, number_points: int, distribution: str, constraints=None,
                 weights=None, parameter_active=True, custom_distribution=None, param_type="continuous",
                 scale_factor=1.):
        self.name = name
        self.range = param_range
        self.range[0] = float(self.range[0])
        self.range[1] = float(self.range[1])
        self.number_points = number_points
        self.weights = weights
        if weights is not None:
            raise NotImplementedError("weights not implemented...yet.")
        self.constraints = constraints
        if constraints is not None:
            raise NotImplementedError("constraints not implemented..yet")
            # todo: note in principle all the functionality is here already just needs to be uncommented and tested
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

        self.generate_initial_data_points()

    @property
    def is_active(self):
        return self.parameter_active

    def generate_data_points(self, num_points):
        """Generate set of n=num_points data points based on the specified distribution, range and param_type of
        this parameter.

         Parameters
         ----------
         num_points : int
            Number of datapoints to generate.

        Note:
        - data_points are not sorted.
        - data_points are not guaranteed to be unique.
        # todo : should they be?
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

    def generate_initial_data_points(self):
        """Generate initial data points based on the specified distribution and range."""
        self.generate_data_points(num_points=self.number_points)


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
    files_needed : list (optional)
        List of files that are needed to run the experiment and should be copied to the run location.
        Defaults to ("*.py",).
    output_directory: str (optional)
        Name of the directory the output should be stored in. Defaults to 'output'.
    output_extension: str (optional)
        Extension of the output files to be picked up by the analysis_script, e.g 'csv' or 'json'. Defaults to 'csv'.
    venv : str (optional)
        Path to the virtual environment that should be initialized before the QCGPilot job is started.

    Attributes:
    ----------
    stdout_basename : str
        Basename of the file that the standard output steam (stdout) of the script should be written to.
        The final filename will be of the form '<stdout_basename><unique_job_identifier>.txt'. O
    working_directory : str
        Name of the current working directory to be passed to QCGPilotJob.
    """
    def __init__(self, source_directory: str, program_name: str, command_line_arguments=None, analysis_script=None,
                 executor="python", files_needed=("*.py",), output_directory=None, output_extension='csv', venv=None):
        if not os.path.exists(source_directory):
            raise ValueError(f"Invalid source_directory path: {source_directory}")
        if not os.path.exists(program_name):
            raise ValueError(f"Invalid program_name: {program_name} is not a file.")
        if analysis_script is not None and not os.path.exists(analysis_script):
            raise ValueError(f"Invalid analysis_script: {analysis_script} is not a file.")

        self.source_directory = source_directory
        self.program_name = program_name
        self.cmdline_arguments = command_line_arguments or {}
        self.analysis_script = analysis_script
        self.job_args = {"exec": executor}
        self.files_needed = files_needed
        self.output_directory = output_directory or 'output'
        self.output_extension = output_extension
        self.stdout_basename = 'stdout'
        self.working_directory = None
        if venv is not None:
            self.job_args["venv"] = venv

    @property
    def current_step_directory(self):
        """Returns the path of the current optimization step."""
        if self.working_directory is not None:
            return os.path.join(self.working_directory, '..')
        else:
            raise RuntimeError(f"Could not get current step directory. Working directory is {self.working_directory}")

    def cmdline_dict_to_list(self):
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
        self._current_optimization_step = None  # TODO: what would this mean if we have various different kinds of opts?

    def create_datapoint_c_product(self):
        """Create initial set of points as Cartesian product of all active parameters.

        Overwrite if other combination is needed."""
        if self.parameters:
            assert isinstance(self.parameters, list), "Parameters are not list."
            for param in self.parameters:
                if not isinstance(param, Parameter):
                    raise ValueError(f"One of the parameters is not of correct type 'Parameter', but is {type(param)}")
            self.data_points = list(itertools.product(*[param.data_points for param in self.parameters
                                                        if param.is_active]))

    @property
    def current_optimization_step(self):
        """Returns the current optimization step."""
        return self._current_optimization_step

    def add_parameter(self, parameter):
        """Adds a parameter to the experiment.

        Parameters
        ----------
        parameter : Parameter
            The parameter to add to the experiment.
        """
        if not isinstance(parameter, Parameter):
            raise ValueError("Can not add parameter that is not of type Parameter.")
        self.parameters.append(parameter)

    def add_optimization_info(self, optimization_info):
        """Adds OptimizationInfo to the experiment.

        Parameters
        ----------
        optimization_info : OptimizationStep
            The optimization step to add to the experiment.
        """
        self.optimization_information_list.append(optimization_info)
