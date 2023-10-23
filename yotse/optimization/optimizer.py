from typing import List
from typing import Tuple

from yotse.optimization.generic_optimization import GenericOptimization
from yotse.pre import Experiment


class Optimizer:
    """
    Optimizer class
    """
    def __init__(self, optimization_algorithm: GenericOptimization):
        """
        Default constructor.

        Parameters:
        ----------
        optimization_algorithm: Object of GenericOptimization
            Optimization algorithm to be executed by this optimizer.
        """
        if optimization_algorithm is not None:
            if not isinstance(optimization_algorithm, GenericOptimization):
                raise ValueError("Optimization algorithm must be subclass of GenericOptimization.")
        self.optimization_algorithm = optimization_algorithm
        self._is_executed = False

    def optimize(self) -> None:
        """
        Optimization step
        """
        self.optimization_algorithm.execute()
        self._is_executed = True

    def suggest_best_solution(self) -> Tuple[List[float], float, int]:
        """Suggest the best solution and return: [best_solution, solution_fitness, solution_index]."""
        return self.optimization_algorithm.get_best_solution()

    def construct_points(self, experiment: Experiment, evolutionary: bool, points_per_param=None) -> None:
        """
        Constructs new set of values around the solution and write them to the experiment.

        Parameters:
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
            raise RuntimeError("construct_points was called before the optimization was executed.")
        if evolutionary:
            if not self.optimization_algorithm.can_create_points_evolutionary:
                raise RuntimeError("trying to construct_points evolutionary for an algorithm that does not support it.")
            experiment.data_points = self.optimization_algorithm.get_new_points()
        else:
            print("Warning: Grid based point generation currently not supporting constraints!")
            self.grid_based_point_creation(experiment=experiment, points_per_param=points_per_param)

    def grid_based_point_creation(self, experiment: Experiment, points_per_param: int) -> None:
        solution, solution_fitness, _ = self.optimization_algorithm.get_best_solution()

        if self.optimization_algorithm.logging_level >= 1:
            print('\n')
            print('Solution:     ', solution)
            print(f'Fitness value: {solution_fitness}')
        ref_factors = self.optimization_algorithm.refinement_factors
        if len(ref_factors) != len(experiment.parameters):
            raise ValueError(f"Length of refinement factors {len(ref_factors)} "
                             f"should be the same as number of parameters {len(experiment.parameters)}.")

        # todo make absolutely sure the index of the solution corresponds with the job number
        # opt_input_datapoint = experiment.data_points[solution_index]  # (x, y, z)
        opt_input_datapoint = solution
        i = 0
        for p, param in enumerate(experiment.parameters):
            if param.is_active:
                # calculate new ranges for each active param
                delta_param = ref_factors[p] * (param.range[1] - param.range[0]) * .5
                opt_range = opt_input_datapoint[i] - delta_param, opt_input_datapoint[i] + delta_param
                i += 1
                # write updated parameter range to each active param
                param.range = opt_range
                # create new points on each active param
                if points_per_param is not None:
                    # generate same amount of points for each param
                    param.generate_data_points(num_points=points_per_param)
                else:
                    # generate different amounts of points for each param
                    param.generate_data_points(num_points=param.number_points)
        # call create points on the experiment
        experiment.create_datapoint_c_product()
        self.optimization_algorithm.overwrite_internal_data_points(experiment.data_points)
        self._is_executed = False
