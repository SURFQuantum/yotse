"""Defines classes and functions for pre step."""
import os
import numpy as np


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
    constraints : list
        List of constraints.
    weights : list (optional)
        List of weights for the parameters, defaults to all 1.
    parameter_active: bool (optional)
        Whether this parameter should be used a varied parameter in this optimization step. Can be used to perform
        sequential optimization of different parameters with only one pre-step. Defaults to true.
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
    def __init__(self, name, parameter_range, number_points, distribution, weights, constraints, parameter_active,
                 custom_distribution=None, data_type="continuous"):
        self.name = name
        self.range = parameter_range
        self.range[0] = float(self.range[0])
        self.range[1] = float(self.range[1])
        self.number_points = number_points
        self.weights = weights
        self.constraints = constraints
        self.parameter_active = parameter_active
        self.data_points = []
        self.distribution = distribution
        self.custom_distribution = custom_distribution
        self.data_type = data_type

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
        else:
            raise ValueError(f"Invalid distribution specified: {self.distribution}")


class ExperimentalSystemSetup:
    """Defines a class for the setup of the system parameters.
    Parameters
    ----------
    directory : str
        Path of the working directory.
    program_name : str
        Name of the script that should be used for the experiment.
    command_line_arguments : dict
        Dictionaries containing as keys the reference of the line argument and as values their value.

    """
    def __init__(self, directory, program_name, command_line_arguments):
        if not os.path.isdir(directory):
            print(f"Invalid directory path: {directory}")

        self.directory = directory
        self.program_name = program_name
        self.line_arguments = command_line_arguments


class Experiment:
    """Class that contains the whole experiment, including ExperimentalSystemSetup, all Parameters and a list of
    optimization steps.
    Parameters
    ----------
    system_setup : ExperimentalSystemSetup
        Instance of the ExperimentalSystemSetup class that contains the setup of the experimental system.
    parameters : list[Parameter]
        List of Parameter instances that define the parameters to be varied in the experiment.
    optimization_steps : list[OptimizationStep]
        List of OptimizationStep instances that define the optimization steps to be performed.

    Attributes
    ----------
    _current_optimization_step : int
        The current optimization step number.
    """

    def __init__(self, system_setup, parameters, optimization_steps):
        self.system_setup = system_setup
        self.parameters = parameters
        self.optimization_steps = optimization_steps
        self._current_optimization_step = 0

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
        self.parameters.append(parameter)

    def add_optimization_step(self, optimization_step):
        """Adds an optimization step to the experiment.

        Parameters
        ----------
        optimization_step : OptimizationStep
            The optimization step to add to the experiment.
        """
        self.optimization_steps.append(optimization_step)
