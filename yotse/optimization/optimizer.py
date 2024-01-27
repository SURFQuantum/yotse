"""optimizer.py.

This module defines the Optimizer class, which serves as a facilitator for running optimization algorithms.
It ensures that the provided optimization algorithm is valid and manages the execution state of the optimization process.

Classes
-------
Optimizer:
    A class that wraps around a generic optimization algorithm.
"""
import math
from typing import List
from typing import Optional
from typing import Tuple

import pandas

from yotse.optimization.generic_optimization import GenericOptimization
from yotse.pre import Experiment


class Optimizer:
    """Optimizer class that wraps around a generic optimization algorithm.

    This class serves as a facilitator for running optimization algorithms
    which are defined as subclasses of `GenericOptimization`. It ensures that
    the provided optimization algorithm is valid and manages the execution
    state of the optimization process.

    Attributes
    ----------
    optimization_algorithm : GenericOptimization
        The optimization algorithm instance that will be executed.
    _is_executed : bool
        Internal flag to track whether the optimization has been executed.

    Raises
    ------
    ValueError
        If the optimization_algorithm is not an instance of GenericOptimization.
    """

    def __init__(self, optimization_algorithm: GenericOptimization):
        """Initializes the Optimizer with the given optimization algorithm.

        Parameters
        ----------
        optimization_algorithm : GenericOptimization
            An instance of GenericOptimization or its subclass that defines
            the optimization algorithm to be executed.

        Raises
        ------
        ValueError
            If the given optimization_algorithm is not an instance or subclass
            of GenericOptimization.
        """
        if optimization_algorithm is not None:
            if not isinstance(optimization_algorithm, GenericOptimization):
                raise ValueError(
                    "Optimization algorithm must be a subclass of GenericOptimization."
                )
        self.optimization_algorithm = optimization_algorithm
        self._is_executed = False
        self.num_executions = 0

    def optimize(self) -> None:
        """Executes the optimization algorithm.

        This method calls the `execute` method of the optimization_algorithm and
        sets the `_is_executed` flag to True upon successful completion. This method
        should be invoked to perform the optimization process.

        Returns
        -------
        None
        """
        self.optimization_algorithm.execute()
        self.num_executions += 1
        self._is_executed = True

    def suggest_best_solution(self) -> Tuple[List[float], float, int]:
        """Suggest the best solution found by the optimization algorithm.

        This method queries the underlying optimization algorithm for the best solution
        it has discovered so far, along with the corresponding fitness value and the
        index of the solution.

        Returns
        -------
        Tuple[List[float], float, int]
            A tuple containing the best solution as a list of floats, the fitness value
            of this solution as a float, and the index of the solution within the
            population as an integer.

        Notes
        -----
        The 'best_solution' represents the variables of the optimum result according to
        the objective function used in the optimization algorithm. The 'solution_fitness'
        is a numerical value representing how 'good' the solution is - the higher/lower the
        better depending on the problem. The 'solution_index' indicates the
        position of the best solution in the population if applicable.
        """
        return self.optimization_algorithm.get_best_solution()

    def construct_points(
        self,
        experiment: Experiment,
        evolutionary: bool,
        points_per_param: Optional[int] = None,
    ) -> None:
        """Constructs new set of values around the solution and write them to the
        Experiment.

        Parameters
        ----------
        experiment : Experiment
            Object of Experiment that the points should be constructed for.
        evolutionary : bool
            True if the optimization algorithm is evolutionary and generates a new set of points.
            False if the optimization algorithm generates a best point and points should be constructed around it.
        points_per_param : int (optional)
            Number of points to construct for each parameter. If None then for each parameter the initially specified
            number of points `Parameter.number_points` will be created. Only used when `evolutionary=False`.
            Defaults to None.
        """
        if not self._is_executed:
            raise RuntimeError(
                "construct_points was called before the optimization was executed."
            )
        if evolutionary:
            if not self.optimization_algorithm.can_create_points_evolutionary:
                raise RuntimeError(
                    "trying to construct_points evolutionary for an algorithm that does not support it."
                )
            experiment.data_points = self.optimization_algorithm.get_new_points()
        else:
            print(
                "Warning: Grid based point generation currently not supporting constraints!"
            )
            self.grid_based_point_creation(
                experiment=experiment, points_per_param=points_per_param
            )

    def grid_based_point_creation(
        self, experiment: Experiment, points_per_param: Optional[int] = None
    ) -> None:
        """Refines the parameter search space based on the best solution and creates new
        data points for the next round of optimization.

        This method uses the best solution found so far to refine the parameter
        ranges and generate a new grid of data points for further exploration.
        New points are created around the best solution using the refinement
        factors defined in the optimization algorithm.

        Parameters
        ----------
        experiment : Experiment
            The experiment object containing the parameters and current data points.
        points_per_param : Optional[int], optional
            The number of points to generate for each parameter. If None, the
            number specified in each parameter object is used, by default None.

        Raises
        ------
        AssertionError
            If the refinement factors are not defined in the optimization algorithm.
        ValueError
            If the number of refinement factors does not match the number of
            parameters in the experiment.

        Notes
        -----
        The method first retrieves the best solution from the optimization algorithm
        and uses the associated fitness value to guide the creation of a refined
        parameter space. It then updates the ranges of each active parameter based
        on the refinement factors and generates a new set of data points accordingly.
        The new data points are then used to overwrite the internal data points
        of the optimization algorithm.

        This method must be called after the optimization algorithm has found an
        initial best solution. It modifies the `experiment` object's parameters
        in-place, adjusting their ranges and data points without returning any value.
        """
        solution, solution_fitness, _ = self.optimization_algorithm.get_best_solution()

        if self.optimization_algorithm.logging_level >= 1:
            print("\n")
            print("Solution:     ", solution)
            print(f"Fitness value: {solution_fitness}")
        ref_factors = self.optimization_algorithm.refinement_factors
        assert (
            ref_factors is not None
        ), "refinement factors can not be None for grid_bases_point_creation."
        if len(ref_factors) != len(experiment.parameters):
            raise ValueError(
                f"Length of refinement factors {len(ref_factors)} "
                f"should be the same as number of parameters {len(experiment.parameters)}."
            )

        # todo make absolutely sure the index of the solution corresponds with the job number
        # opt_input_datapoint = experiment.data_points[solution_index]  # (x, y, z)
        opt_input_datapoint = solution
        for p, param in enumerate(experiment.parameters):
            if param.is_active:
                # calculate new ranges for each active param
                delta_param = ref_factors[p] * (param.range[1] - param.range[0]) * 0.5
                opt_range = [
                    opt_input_datapoint[p] - delta_param,
                    opt_input_datapoint[p] + delta_param,
                ]
                # write updated parameter range to each active param
                param.range = opt_range
                # create new points on each active param
                if points_per_param is not None:
                    # generate same amount of points for each param
                    param.data_points = param.generate_data_points(
                        num_points=points_per_param
                    )
                else:
                    # generate different amounts of points for each param
                    param.data_points = param.generate_data_points(
                        num_points=param.number_points
                    )
        # update data points points on the experiment
        experiment.data_points = experiment.create_datapoint_c_product()
        self.optimization_algorithm.overwrite_internal_data_points(
            experiment.data_points
        )
        self._is_executed = False

    def update_blackbox_cost_data(
        self, experiment: Experiment, data: pandas.DataFrame
    ) -> None:
        """Update internal dataframe of the optimization algorihtm, mapping input
        parameters to the associated cost from input data.

        Note: This also checks that the ordering of the entries is the same as the data_points of the experiment.

        Parameters
        ----------
        experiment : Experiment
            The experiment object containing the parameters and current data points.
        data : pandas.Dataframe
            A pandas dataframe containing the collected data in the format cost_value init_param_1 ... init_param_n.
        """
        # check ordering of data versus initial datapoints to avoid mistakes when fetching corresponding cost by index
        if len(data) != len(experiment.data_points):
            raise ValueError(
                "Data has a different number of rows than the list of datapoints."
            )
        for i, values in enumerate(experiment.data_points):
            row = data.iloc[i]
            if any(
                not math.isclose(row.iloc[j + 1], values[j]) for j in range(len(values))
            ):
                raise ValueError(
                    f"Position of {values} is different between data and original data_points"
                )
        self.optimization_algorithm.update_internal_cost_data(data=data)
