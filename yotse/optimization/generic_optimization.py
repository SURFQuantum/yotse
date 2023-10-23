import inspect
import math
from abc import ABCMeta
from abc import abstractmethod

import pandas

from yotse.pre import Experiment


class GenericOptimization:
    """
    Base class for optimization algorithms.

    Parameters:
    ----------
    function : function
        Cost function used for optimization.
    refinement_factors : list (optional)
        Refinement factors for all parameters. If specified must be list of length = #params.
        Defaults to None.
    logging_level : int (optional)
        Level of logging: 1 - only essential data; 2 - include plots; 3 - dump everything.
        Defaults to 1.
    extrema: str (optional)
        Define what type of problem to solve. 'extrema' can be equal to either MINIMUM or MAXIMUM. The
        optimization algorithm will look for minimum and maximum values respectively.
        Defaults to MINIMUM.
    """
    __metaclass__ = ABCMeta

    MAXIMUM = 0
    MINIMUM = 1

    def __init__(self, function, opt_instance=None, refinement_factors=None, logging_level=1, extrema=MINIMUM,
                 evolutionary=False):
        self.logging_level = logging_level
        self.extrema = extrema
        self.function = function
        self.refinement_factors = refinement_factors
        self.data = None
        self.can_create_points_evolutionary = evolutionary
        self.optimization_instance = opt_instance
        self.input_param_cost_df = None

    def get_function(self):
        """Returns the cost function."""
        return self.function

    @abstractmethod
    def execute(self) -> None:
        """
        Execute method should be implemented in every derived class.

        """
        raise NotImplementedError('The \'{}\' method is not implemented'.format(inspect.currentframe().f_code.co_name))

    def get_best_solution(self) -> (list, float, int):
        """
        Get the best solution. Should be implemented in every derived class.

        Returns:
        -------
        solution, solution_fitness, solution_idx
            Solution its fitness and its index in the list of data points.
        """
        raise NotImplementedError('The \'{}\' method is not implemented'.format(inspect.currentframe().f_code.co_name))

    @abstractmethod
    def get_new_points(self) -> list:
        """
        Get new points. Should be implemented in every evolutional algorithm.

        Returns:
        -------
        new_points : list of tuples
            New points for the next iteration of the optimization.
        """
        # todo: test output type here in tests
        raise NotImplementedError('The \'{}\' method is not implemented'.format(inspect.currentframe().f_code.co_name))

    @abstractmethod
    def overwrite_internal_data_points(self, data_points: list):
        """
        Overwrite the internal set of data points with one externally generated. E.g. when manually passing new points
        to an evolutionary optimization algorithm.

        Parameters:
        ----------
        data_points : list
            List containing all new data points that should be passed to the optimization.
        """
        raise NotImplementedError('The \'{}\' method is not implemented'.format(inspect.currentframe().f_code.co_name))

    def update_internal_cost_data(self, experiment: Experiment, data: pandas.DataFrame) -> None:
        """Update internal dataframe mapping input parameters to the associated cost from input data.
        It also checks that the ordering of the entries is the same as the data_points of the experiment.

        Parameters:
        ----------
        data : pandas.Dataframe
            A pandas dataframe containing the collected data in the format cost_value init_param_1 ... init_param_n.
        """
        # check ordering of data versus initial datapoints to avoid mistakes when fetching corresponding cost by index
        if len(data) != len(experiment.data_points):
            raise ValueError("Data has a different number of rows than the list of datapoints.")
        for i, values in enumerate(experiment.data_points):
            row = data.iloc[i]
            if any(not math.isclose(row[j + 1], values[j]) for j in range(len(values))):
                raise ValueError(f"Position of {values} is different between data and original data_points")

        self.input_param_cost_df = data

    def input_params_to_cost_value(self, *args, **kwargs) -> float:
        raise NotImplementedError('The \'{}\' method is not implemented'.format(inspect.currentframe().f_code.co_name))
