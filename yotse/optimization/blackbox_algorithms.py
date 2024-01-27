"""Collection of Subclasses of :class:GenericOptimization implementing different
optimization algorithms."""
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from bayes_opt import BayesianOptimization
from pygad.pygad import GA

from yotse.optimization.generic_optimization import GenericOptimization
from yotse.optimization.modded_pygad_ga import ModGA  # type: ignore[attr-defined]
from yotse.pre import ConstraintDict
from yotse.utils.utils import ndarray_to_list


class GAOpt(GenericOptimization):
    """Genetic algorithm.

    Parameters
    ----------
    blackbox_optimization: bool
        Whether this is used as a blackbox optimization.
    initial_data_points: np.ndarray
        Initial population of data points to start the optimization with.
    num_generations : int
        Number of generations in the genetic algorithm.
    num_parents_mating : int
        Number of solutions to be selected as parents in the genetic algorithm.
    fitness_func : function (optional)
        Fitness/objective/cost function/function to optimize. Only needed if `blackbox_optimization=False`.
        Default is None.
    gene_space : dict or list (optional)
        Dictionary with constraints. Keys can be 'low', 'high' and 'step'. Alternatively list with acceptable values or
        list of dicts. If only single object is passed it will be applied for all input parameters, otherwise a
        separate list or dict has to be supplied for each parameter.
        Defaults to None.
    refinement_factors : list (optional)
        Refinement factors for each active parameter in the optimization in range [0.,1.] to be used for manual
        grid point generation.
        Defaults to None.
    logging_level : int (optional)
        Level of logging: 1 - only essential data; 2 - include plots; 3 - dump everything.
        Defaults to 1.
    allow_duplicate_genes : bool (optional)
        If True, then a solution/chromosome may have duplicate gene values.
        If False, then each gene will have a unique value in its solution.
        Defaults to False.
    pygad_kwargs : (optional)
        Optional pygad arguments to be passed to `pygad.GA`.
        See https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class for documentation.

    Attributes
    ----------
    constraints : dict or list
        Constraints to check for during generation of new points.
    """

    def __init__(
        self,
        blackbox_optimization: bool,
        initial_data_points: np.ndarray,
        num_generations: int,
        num_parents_mating: int,
        gene_space: Optional[ConstraintDict] = None,
        refinement_factors: Optional[List[float]] = None,
        logging_level: int = 1,
        allow_duplicate_genes: bool = False,
        fitness_func: Optional[Callable[..., float]] = None,
        **pygad_kwargs: Any,
    ):
        """Initialize the GAOpt object."""
        if blackbox_optimization:
            if fitness_func is not None:
                raise ValueError(
                    "blackbox_optimization set to True, but fitness_func is not None."
                )
            fitness_func = self.input_params_to_cost_value

        # Note: number of new points is determined by initial population
        # gene_space to limit space in which new genes are formed = constraints
        ga_instance = ModGA(
            fitness_func=self._objective_func,
            initial_population=ndarray_to_list(initial_data_points),
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            # todo : gene_type/_space are exactly data_type/constraints of the params, see core.py
            # gene_type=gene_type,
            gene_space=gene_space,
            save_best_solutions=True,
            allow_duplicate_genes=allow_duplicate_genes,
            mutation_by_replacement=True,
            **pygad_kwargs,
        )
        self.blackbox_optimization = blackbox_optimization
        self.constraints = gene_space
        # todo : why if save_solutions=True the optimization doesn't converge anymore?
        super().__init__(
            function=fitness_func,  # type: ignore [arg-type]
            opt_instance=ga_instance,
            refinement_factors=refinement_factors,
            logging_level=logging_level,
            extrema=self.MINIMUM,
            evolutionary=True,
        )

    @property
    def current_datapoints(self) -> np.ndarray:
        """Return the current datapoints that will be used if an optimization is started
        now.

        In this case it is the population.
        """
        return self.optimization_instance.population

    @property
    def max_iterations(self) -> int:
        """Return maximum number of iterations of GA = number of generation."""
        return int(self.optimization_instance.num_generations)

    def _objective_func(
        self, ga_instance: GA, solution: List[float], solution_idx: int
    ) -> float:
        """Fitness function to be called from PyGAD.

        Wrapper around the actual function to give pygad some more functionality.
        First, it adds the possibility to choose whether to max-/minimize the fitness.
        Second, it removes the necessity to pass the ga_instance to the function, thus making the implementation
        more general.

        Parameters
        ----------
        ga_instance
            Instance of pygad.GA.
        solution : List[float]
            List of solutions.
        solution_idx : int
            Index of solution.

        Returns
        -------
        Fitness value.
        """
        # Invert function to find the minimum, if needed
        factor = 1.0
        if self.extrema == self.MINIMUM:
            factor = -1.0
        if self.blackbox_optimization:
            # passing params to self.input_params_to_cost_value
            fitness = factor * self.function(solution, solution_idx)
        else:
            # passing params to the function of type Callable[...,float]
            fitness = factor * self.function(solution)

        if self.logging_level >= 3:
            print(solution, solution_idx, fitness)

        return fitness

    def execute(self) -> None:
        """Execute single step in the genetic algorithm."""
        self.optimization_instance.run_single_generation()

        # Report convergence
        if self.logging_level >= 2:
            self.optimization_instance.plot_fitness()

    def get_best_solution(self) -> Tuple[List[float], None, None]:  # type: ignore[override]
        """Get the best solution. We don't yet know the fitness for the solution
        (because we have not run the simulation for those values yet), so just return
        the point.

        Returns
        -------
        solution, solution_fitness, solution_idx
            Solution its fitness and its index in the list of cost function solutions.
        """
        if self.optimization_instance is None:
            raise ValueError(
                "Trying to `get_best_solution`, but GA instance not initialized."
            )
        best_solution = self.optimization_instance.best_solutions.tolist()[-1]
        # solution_idx = self.ga_instance.population.tolist().index(best_solution)
        return best_solution, None, None
        # todo: this could also instead return solution and fitness of the best solution one generation back?

    def get_new_points(self) -> np.ndarray:
        """Get new points from the GA (aka return the next population).

        Returns
        -------
        new_points : np.ndarray
            New points for the next iteration of the optimization.
        """
        new_points = self.optimization_instance.population
        # todo: see if we check constraints somewhere else, might be redundant
        if self.constraints is not None:
            # double check constraints are kept
            for point in new_points:
                for index, value in enumerate(point):
                    if isinstance(self.constraints, list):
                        if self.constraints[index] is not None:
                            assert (
                                self.constraints[index]["low"]
                                <= value
                                <= self.constraints[index]["high"]
                            )
                    elif isinstance(self.constraints, dict):
                        assert (
                            self.constraints["low"] <= value <= self.constraints["high"]
                        )
                    else:
                        raise TypeError(f"Unacceptable type {type} for constraints.")
        new_points = [tuple(point) for point in new_points]

        return np.array(new_points)

    def overwrite_internal_data_points(self, data_points: np.ndarray) -> None:
        """Overwrite internal GA population with new datapoints from experiment."""
        self.optimization_instance.population = data_points


class BayesOpt(GenericOptimization):
    """Bayesian optimization.

    Parameters
    ----------
    blackbox_optimization: bool
        Whether this is used as a blackbox optimization.
    pbounds: dict
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values.
    initial_data_points: np.ndarray (optional)
        Initial population of data points to start the optimization with.
        If none are specified lets the algorithm suggest a point. Defaults to None.
    fitness_func : function (optional)
        Fitness/objective/cost function/function to optimize. Only needed if `blackbox_optimization=False`.
        Default is None.
    logging_level : int (optional)
        Level of logging: 1 - only essential data; 2 - include plots; 3 - dump everything.
        Defaults to 1.
    bayesopt_kwargs : (optional)
        Optional arguments to be passed to `bayes_opt.BayesianOptimization`.
        See the documentation of that class for more info.
    """

    def __init__(
        self,
        blackbox_optimization: bool,
        pbounds: Dict[Any, Tuple[int, int]],
        initial_data_points: Optional[np.ndarray] = None,
        naive_parallelization: bool = False,
        grid_size: int = 1,
        refinement_factors: Optional[List[float]] = None,
        fitness_func: Optional[Callable[..., float]] = None,
        logging_level: int = 1,
        **bayesopt_kwargs: Any,
    ) -> None:
        """Initialize Bayesian optimization."""
        if blackbox_optimization:
            if fitness_func is not None:
                raise ValueError(
                    "blackbox_optimization set to True, but fitness_func is not None."
                )
        else:
            raise NotImplementedError(
                "Whitebox optimization with BayesOpt not implemented...yet."
            )

        if "utility_function" not in bayesopt_kwargs:
            raise ValueError(
                "utility_function must be specified for Bayesian Optimization."
            )
        self.utility_function = bayesopt_kwargs["utility_function"]
        bayesopt_kwargs.pop("utility_function")
        if "n_iter" not in bayesopt_kwargs:
            raise ValueError("n_iter must be specified for Bayesian Optimization.")
        self._max_iterations = bayesopt_kwargs["n_iter"]
        bayesopt_kwargs.pop("n_iter")
        self.naive_parallelization = naive_parallelization
        self.grid_size = grid_size
        self.pbounds = pbounds

        optimizer = BayesianOptimization(
            f=fitness_func,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
            **bayesopt_kwargs,
        )
        # warn about discrete values
        print(
            "WARNING: Bayesian Optimization does not currently implement discrete variables. See "
            "https://github.com/bayesian-optimization/BayesianOptimization/blob/master/examples/advanced-tour.ipynb"
        )
        super().__init__(
            function=self.utility_function,
            refinement_factors=refinement_factors,
            opt_instance=optimizer,
            logging_level=logging_level,
            extrema=self.MINIMUM,
            evolutionary=True,
        )

        # set initial point to investigate
        if initial_data_points is None:
            self.next_point_to_probe = optimizer.suggest(self.utility_function)
            if self.naive_parallelization:
                self.overwrite_internal_data_points(
                    self.create_points_around_suggestion(self.next_point_to_probe)
                )
        else:
            self.overwrite_internal_data_points(initial_data_points)

    @property
    def current_datapoints(self) -> np.ndarray:
        """Return the current datapoints that will be used if an optimization is started
        now.

        In this case it is the currently suggested point.
        """
        input_data = self.next_point_to_probe
        # handle single or multiple points
        input_data = {
            key: [value] if not isinstance(value, list) else value
            for key, value in input_data.items()
        }
        next_point = np.array(list(input_data.values()))
        # transpose to get desired shape (num_data_points, num_params)
        next_point = next_point.T

        return next_point

    @property
    def max_iterations(self) -> int:
        """Return maximum number of iterations of bayesian optimization."""
        return int(self._max_iterations)

    def execute(self) -> None:
        """Execute single step in the bayesian optimization."""
        # Note this should be run after the user script has been executed with input next_point_to_probe
        target_column = self.input_param_cost_df.columns[0]

        for index, row in self.input_param_cost_df.iterrows():
            target_point = row[target_column]

            if self.extrema == self.MINIMUM:
                # minimize = find max of negative cost
                target_point *= -1

            params = row.drop(target_column).tolist()
            self.optimization_instance.register(params=params, target=target_point)

    def get_best_solution(self) -> Tuple[List[float], float, int]:
        """Get the best solution. Should be implemented in every derived class.

        Returns
        -------
        solution, solution_fitness, solution_idx
            Solution its fitness and its index in the list of data points.
        """
        # todo: this does not output solution, solution_fitness, solution_idx yet but params, cost, 0
        # todo: question is what solution_idx would even mean here
        solution = self.optimization_instance.max["target"]
        if self.extrema == self.MINIMUM:
            # maximized neg cost, converting to pos cost again
            solution *= -1
        return (
            list(self.optimization_instance.max["params"].values()),
            solution,
            0,
        )

    def get_new_points(self) -> np.ndarray:
        """Get new points from the BayesianOptimization instance. This is done via the
        `suggest` function.

        Returns
        -------
        new_points : np.ndarray
            New points for the next iteration of the optimization.
        """
        next_point = self.optimization_instance.suggest(self.utility_function)
        if self.naive_parallelization:
            new_points = self.create_points_around_suggestion(next_point)
        else:
            new_points = np.array([list(next_point.values())])
        return new_points

    def create_points_around_suggestion(
        self, suggested_point: Dict[str, float]
    ) -> np.ndarray:
        """BayesOpt only suggest a single point per iteration. In order to parallelize
        the optimization we use this function to create multiple data points to evaluate
        per iteration.

        Note: This current implementation is by now way optimal! This is the most naive and simple way to create
        multiple points to evaluate and should be improved in the future.
        """
        param_names = list(suggested_point.keys())
        param_values = list(suggested_point.values())
        param_bounds = list(self.pbounds.values())
        grid_points = []
        for p, param in enumerate(param_names):
            # calculate new ranges for each param
            if self.refinement_factors is None:
                raise ValueError(
                    "refinement_factors can not be None when creating points based on them."
                )
            delta_param = (
                self.refinement_factors[p]
                * (param_bounds[p][1] - param_bounds[p][0])
                * 0.5
            )
            grid_range = [
                param_values[p] - delta_param,
                param_values[p] + delta_param,
            ]
            new_points_in_grid = np.linspace(
                grid_range[0], grid_range[1], self.grid_size
            )

            grid_points.append(new_points_in_grid)

        # Create a meshgrid from the grid points
        grid_mesh = np.array(np.meshgrid(*grid_points)).T.reshape(-1, len(param_names))

        return grid_mesh

    def overwrite_internal_data_points(self, data_points: np.ndarray) -> None:
        """After we generated a new point with get_new_points this function can be used
        to write that data point (or another point to investigate next) to the class."""
        param_keys = self.optimization_instance._space._keys

        new_point_from_array = {}

        for i, param_name in enumerate(param_keys):
            new_point_from_array[param_name] = data_points[:, i].tolist()

        self.next_point_to_probe = new_point_from_array
