"""generic_optimization.py.

This module defines the GenericOptimization class, a base class for optimization algorithms.

Classes
-------
GenericOptimization:
    A base class for optimization algorithms.
"""
import inspect
import math
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas
from bayes_opt import BayesianOptimization

from yotse.optimization.modded_pygad_ga import ModGA  # type: ignore[attr-defined]


class GenericOptimization:
    """Base class for optimization algorithms.

    Parameters
    ----------
    function : Callable[..., float]
        Fitness function to be used for optimization. This can either be a discrete mapping between input parameters
         and associated cost (in the case of blackbox optimization) or a known analytical function (in the case of
         whitebox optimization).
    opt_instance: Union[ModGA, BayesianOptimization, None] (optional)
        Instance of the optimization engine.
        Defaults to None.
    refinement_factors : list (optional)
        Refinement factors for all parameters. If specified must be list of length = #params.
        Defaults to None.
    logging_level : int (optional)
        Level of logging: 1 - only essential data; 2 - include plots; 3 - dump everything.
        Defaults to 1.
    extrema: int (optional)
        Define what type of problem to solve. 'extrema' can be equal to either MINIMUM or MAXIMUM. The
        optimization algorithm will look for minimum and maximum values respectively.
        Defaults to MINIMUM.
    evolutionary : bool (optional)
        Whether the optimization algorithm allows for evolutionary optimization. Defaults to False.
    """

    __metaclass__ = ABCMeta

    MAXIMUM = 0
    MINIMUM = 1

    def __init__(
        self,
        function: Callable[..., float],
        opt_instance: Union[ModGA, BayesianOptimization] = None,
        refinement_factors: Optional[List[float]] = None,
        logging_level: int = 1,
        extrema: int = MINIMUM,
        evolutionary: bool = False,
    ):
        """Initialize the GenericOptimization object."""
        self.logging_level = logging_level
        self.extrema = extrema
        self.function = function
        self.refinement_factors = refinement_factors
        self.data = None
        self.can_create_points_evolutionary = evolutionary
        self.optimization_instance = opt_instance
        self.input_param_cost_df: pandas.DataFrame = pandas.DataFrame()

    @property
    @abstractmethod
    def current_datapoints(self) -> np.ndarray:
        """Return the current datapoints that will be used if an optimization is started
        now."""
        frame = inspect.currentframe()
        assert frame is not None, "Failed to get the current frame"
        raise NotImplementedError(
            f"The '{frame.f_code.co_name}' method is not implemented"
        )

    @property
    @abstractmethod
    def max_iterations(self) -> int:
        """Return the maximum number of iterations of the optimization if applicable."""
        frame = inspect.currentframe()
        assert frame is not None, "Failed to get the current frame"
        raise NotImplementedError(
            f"The '{frame.f_code.co_name}' method is not implemented"
        )

    def get_function(self) -> Callable[..., float]:
        """Returns the cost function."""
        return self.function

    @abstractmethod
    def execute(self) -> None:
        """Execute method should be implemented in every derived class."""
        frame = inspect.currentframe()
        assert frame is not None, "Failed to get the current frame"
        raise NotImplementedError(
            f"The '{frame.f_code.co_name}' method is not implemented"
        )

    def get_best_solution(self) -> Tuple[List[float], float, int]:
        """Get the best solution (aka the best set of input parameters). Should be
        implemented in every derived class.

        Returns
        -------
        solution, solution_fitness, solution_idx
            Solution (set of params), its fitness (associated cost) and its index in the list of data points.
            # todo: why do we care about the index?
        """
        frame = inspect.currentframe()
        assert frame is not None, "Failed to get the current frame"
        raise NotImplementedError(
            f"The '{frame.f_code.co_name}' method is not implemented"
        )

    @abstractmethod
    def get_new_points(self) -> np.ndarray:
        """Get new points. Should be implemented in every evolutional algorithm.

        Returns
        -------
        new_points : np.ndarray
            New points for the next iteration of the optimization.
        """
        # todo: test output type here in tests
        frame = inspect.currentframe()
        assert frame is not None, "Failed to get the current frame"
        raise NotImplementedError(
            f"The '{frame.f_code.co_name}' method is not implemented"
        )

    @abstractmethod
    def overwrite_internal_data_points(self, data_points: np.ndarray) -> None:
        """Overwrite the internal set of data points with one externally generated. E.g.
        when manually passing new points to an evolutionary optimization algorithm.

        Parameters
        ----------
        data_points : np.ndarray
            Array containing all new data points that should be passed to the optimization.
        """
        # todo: is this generic or should this be in a specific implementation?
        frame = inspect.currentframe()
        assert frame is not None, "Failed to get the current frame"
        raise NotImplementedError(
            f"The '{frame.f_code.co_name}' method is not implemented"
        )

    def update_internal_cost_data(self, data: pandas.DataFrame) -> None:
        """Update internal dataframe mapping input parameters to the associated cost
        from input data.

        Parameters
        ----------
        data : pandas.Dataframe
            A pandas dataframe containing the collected data in the format cost_value init_param_1 ... init_param_n.
        """

        self.input_param_cost_df = data

    def input_params_to_cost_value(
        self, solution: List[float], solution_idx: int
    ) -> Any:
        """Return value of cost function for given set of input parameter values and
        their index in the set of points.

        Parameters
        ----------
        solution : list
            Set of input parameter values of shape [param_1, param_2, .., param_n].
        solution_idx : int
            Index of the solution within the set of points.
        """
        row = self.input_param_cost_df.iloc[solution_idx]
        if all(
            math.isclose(row.iloc[i + 1], solution[i]) for i in range(len(solution))
        ):
            return row.iloc[0]
        else:
            print(self.input_param_cost_df)
            print(
                f"Solution {solution} was not found in internal dataframe row {solution_idx}."
            )
            raise ValueError(
                f"Solution {solution} was not found in internal dataframe row {solution_idx}."
            )
