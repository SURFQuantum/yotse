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
    parameter_range : list(min, max)
        List with the min and max value of the parameter; min and max are floats.
    number_points: int
        Number of points to be explored.
    distribution : str
        Type of distribution of the points. Currently supports 'linear', 'uniform', 'normal', 'log' or 'custom'.
        If 'custom' is specified then parameter custom_distribution must be set.
    constraints : list (optional)
        List of constraints, defaults to None.
    weights : list (optional)
        List of weights for the parameters, defaults to None.
    parameter_active: bool (optional)
        Whether this parameter should be used a varied parameter in this optimization step. Can be used to perform
        sequential optimization of different parameters with only one pre-step. Defaults to True.
    custom_distribution : function
        Custom distribution function that takes as arguments (min_value: float, max_value: float, number_points: int)
        and returns a list of float points.

    #TODO: keep those?
    data_type: str (optional)
        Type of variable: discrete or continuous, defaults to continuous.

    Attributes
    ----------
    data_points : list
        Data points to be explored for this parameter.
    """
    def __init__(self, name: str, parameter_range: list, number_points: int, distribution: str, constraints: list,
                 weights=None, parameter_active=True, custom_distribution=None, data_type="continuous"):
        self.name = name
        self.range = parameter_range
        self.range[0] = float(self.range[0])
        self.range[1] = float(self.range[1])
        self.number_points = number_points
        self.weights = weights
        self.constraints = constraints
        self.parameter_active = parameter_active
        self.data_points = []
        if custom_distribution is not None and distribution != 'custom':
            raise ValueError(f"Custom distribution supplied but distribution set to {distribution}!")
        self.distribution = distribution
        self.custom_distribution = custom_distribution
        self.data_type = data_type

        self.generate_initial_data_points()

    @property
    def is_active(self):
        return self.parameter_active

    def generate_initial_data_points(self):
        """Generate initial data points based on the specified distribution and range."""
        if self.distribution == "linear":
            self.data_points = np.linspace(self.range[0], self.range[1], self.number_points).tolist()
        elif self.distribution == "uniform":
            self.data_points = np.random.uniform(self.range[0], self.range[1], self.number_points).tolist()
        elif self.distribution == "normal":
            self.data_points = np.random.normal((self.range[0] + self.range[1]) / 2,
                                                abs(self.range[1] - self.range[0]) / 3, self.number_points).tolist()
        elif self.distribution == "log":
            self.data_points = np.logspace(np.log10(self.range[0]), np.log10(self.range[1]),
                                           self.number_points).tolist()
        elif self.distribution == "custom" and self.custom_distribution is not None:
            self.data_points = self.custom_distribution(self.range[0], self.range[1], self.number_points)
            if len(self.data_points) != self.number_points:
                raise ValueError(f'Custom distribution returned invalid number of points {len(self.data_points)}.')
            assert min(self.data_points) >= self.range[0]
            assert max(self.data_points) <= self.range[1]
        else:
            raise ValueError(f"Invalid distribution specified: {self.distribution}")


class SystemSetup:
    """Defines a class for the setup of the system parameters.
    Parameters
    ----------
    directory : str
        Path of the working directory.
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
        Defaults to ["*.py"].
    """
    def __init__(self, directory: str, program_name: str, command_line_arguments={}, analysis_script=None,
                 executor="python", files_needed=["*.py"]):
        if not os.path.isdir(directory):
            raise ValueError(f"Invalid directory path: {directory}")
        if not os.path.isfile(program_name):
            raise ValueError(f"Invalid program_name: {program_name} is not a file.")
        if analysis_script is not None and not os.path.isfile(analysis_script):
            raise ValueError(f"Invalid analysis_script: {analysis_script} is not a file.")

        self.directory = directory
        self.program_name = program_name
        self.cmdline_arguments = command_line_arguments
        self.analysis_script = analysis_script
        self.executor = executor
        self.files_needed = files_needed

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
    """

    def __init__(self, name, opt_parameters):
        self.name = name
        self.parameters = opt_parameters


class Experiment:
    """Class that contains the whole experiment, including ExperimentalSystemSetup, all Parameters and a list of
    optimization steps.
    Parameters
    ----------
    experiment_name : str
        Descriptive name for the experiment
    system_setup : SystemSetup
        Instance of the ExperimentalSystemSetup class that contains the setup of the experimental system.
    parameters : list[Parameter] (optional)
        List of Parameter instances that define the parameters to be varied in the experiment. Defaults to None.
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

    def __init__(self, experiment_name, system_setup, parameters=None, opt_info_list=[]):
        self.name = experiment_name
        self.system_setup = system_setup
        self.parameters = []
        if parameters is not None:
            self.parameters = parameters
        if opt_info_list:
            assert isinstance(opt_info_list, list)
            for item in opt_info_list:
                assert isinstance(item, OptimizationInfo)
        self.optimization_information_list = opt_info_list
        self.data_points = []
        self._current_optimization_step = None  # TODO: what would this mean if we have various different kinds of opts?

    def create_datapoint_c_product(self):  # TODO: better name?
        """Create initial set of points as Cartesian product of all active parameters.

        Overwrite if other combination is needed."""
        self.data_points = itertools.product([param.data_points for param in self.parameters if param.is_active])

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
        assert isinstance(parameter, Parameter)
        self.parameters.append(parameter)

    def add_optimization_info(self, optimization_info):
        """Adds OptimizationInfo to the experiment.

        Parameters
        ----------
        optimization_info : OptimizationStep
            The optimization step to add to the experiment.
        """
        self.optimization_information_list.append(optimization_info)
