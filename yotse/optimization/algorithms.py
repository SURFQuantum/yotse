import math

import numpy as np

from yotse.optimization.ga import ModGA
from yotse.optimization.generic_optimization import GenericOptimization


class GAOpt(GenericOptimization):
    """
    Genetic algorithm.

    Parameters:
    ----------
    fitness_func : function
        Fitness/objective/cost function.
    initial_population: list
        Initial population of data points to start the optimization with.
    num_generations : int
        Number of generations in the genetic algorithm.
    num_parents_mating : int
        Number of solutions to be selected as parents in the genetic algorithm.
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
    allow_duplicate_genes : Bool (optional)
        If True, then a solution/chromosome may have duplicate gene values.
        If False, then each gene will have a unique value in its solution.
        Defaults to False.
    pygad_kwargs : (optional)
        Optional pygad arguments to be passed to `pygad.GA`.
        See https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class for documentation.
    """
    def __init__(self, initial_population: list, num_generations: int, num_parents_mating: int,
                 gene_space: dict = None, refinement_factors=None, logging_level=1, allow_duplicate_genes=False,
                 fitness_func=None,
                 **pygad_kwargs):
        if fitness_func is None:
            fitness_func = self.input_params_to_cost_value

        # Note: number of new points is determined by initial population
        # gene_space to limit space in which new genes are formed = constraints
        ga_instance = ModGA(fitness_func=self._objective_func,
                            initial_population=initial_population,
                            num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            # todo : gene_type/_space are exactly data_type/constraints of the params, see core.py
                            # gene_type=gene_type,
                            gene_space=gene_space,
                            save_best_solutions=True,
                            allow_duplicate_genes=allow_duplicate_genes,
                            mutation_by_replacement=True,
                            **pygad_kwargs
                            )
        self.constraints = gene_space
        # todo : why if save_solutions=True the optimization doesn't converge anymore?
        super().__init__(function=fitness_func, opt_instance=ga_instance, refinement_factors=refinement_factors,
                         logging_level=logging_level, extrema=self.MINIMUM, evolutionary=True)

    def _objective_func(self, ga_instance, solution, solution_idx) -> float:
        """
        Fitness function to be called from PyGAD

        Parameters:
        ----------
        ga_instance :
            # todo : add docstring
        solution : list
            List of solutions.
        solution_idx : int
            Index of solution.

        Returns:
        -------
            Fitness value.
        """
        # Invert function to find the minimum, if needed
        factor = 1.
        if self.extrema == self.MINIMUM:
            factor = -1.

        fitness = factor * self.function(ga_instance, solution, solution_idx)

        if self.logging_level >= 3:
            print(solution, solution_idx, fitness)

        return fitness

    def execute(self) -> None:
        """
        Execute single step in the genetic algorithm.
        """
        self.optimization_instance.run_single_generation()

        # Report convergence
        if self.logging_level >= 2:
            self.optimization_instance.plot_fitness()

    def get_best_solution(self) -> (list, None, None):
        """
        Get the best solution. We don't yet know the fitness for the solution (because we have not run the simulation
        for those values yet), so just return the point.

        Returns:
        -------
        solution, solution_fitness, solution_idx
            Solution its fitness and its index in the list of cost function solutions.
        """
        if self.optimization_instance is None:
            raise ValueError("GA instance not initialized.")
        best_solution = self.optimization_instance.best_solutions.tolist()[-1]
        # solution_idx = self.ga_instance.population.tolist().index(best_solution)
        return best_solution, None, None
        # todo: this could also instead return solution and fitness of the best solution one generation back?

    def get_new_points(self) -> list:
        """
        Get new points from the GA (aka return the next population).

        Returns:
        -------
        new_points : list of tuples
            New points for the next iteration of the optimization.
        """
        new_points = self.optimization_instance.population.tolist()
        # todo : this check should be redundant and could be removed
        if self.constraints is not None:
            # double check constraints are kept
            for point in new_points:
                for index, value in enumerate(point):
                    if isinstance(self.constraints, list):
                        if self.constraints[index] is not None:
                            assert self.constraints[index]['low'] <= value <= self.constraints[index]['high']
                    elif isinstance(self.constraints, dict):
                        assert self.constraints['low'] <= value <= self.constraints['high']
                    else:
                        raise TypeError(f"Unacceptable type {type} for constraints.")
        return [tuple(point) for point in new_points]

    def overwrite_internal_data_points(self, data_points: list) -> None:
        # convert list of tuples into np.array
        data_array = np.array(data_points)
        self.optimization_instance.population = data_array

    def input_params_to_cost_value(self, ga_instance, solution: list, solution_idx: int) -> float:
        """Return value of cost function for given set of input parameter values and their index in the set of points.

        Parameters:
        ----------
        solution : list
            Set of input parameter values of shape [param_1, param_2, .., param_n].
        solution_idx : int
            Index of the solution within the set of points.
        """
        # todo: input parameters of this are highly GA specific and should be made general
        row = self.input_param_cost_df.iloc[solution_idx]
        if all(math.isclose(row[i + 1], solution[i]) for i in range(len(solution))):
            return row[0]
        else:
            raise ValueError(f"Solution {solution} was not found in internal dataframe row {solution_idx}.")


# class CGOpt(GenericOptimization):
#     """
#     CG algorithm
#     #todo: fill this with more info
#     """
#     def __init__(self, function, num_iterations=100, logging_level=1):
#         """
#         Default constructor
#
#         Parameters:
#         ----------
#         function : function
#             Fitness/objective/cost function.
#         num_iterations : int (optional)
#             Number of iterations.
#             Defaults to 100.
#         logging_level : int (optional)
#             Level of logging: 1 - only essential data; 2 - include plots; 3 - dump everything.
#             Defaults to 1.
#         """
#         super().__init__(function, logging_level, self.MINIMUM)
#         self.num_iterations = num_iterations
#
#     def _objective_func(self, solution):
#         """
#         Fitness function to be called from PyGAD
#         Parameters:
#         ----------
#         solution : list
#             List of solutions.
#
#         Returns:
#         -------
#             Fitness value.
#         """
#         x, y = solution
#         # x_fixed, y_fixed = args
#
#         # Invert function to find the minimum, if needed
#         factor = 1.
#         if self.extrema == self.MAXIMUM:
#             factor = -1.
#
#         obj = factor * self.function([x, y])
#
#         # err = []
#         # for n in range(0, len(x_fixed)):
#         #     err.append(np.abs(obj - self.function([x_fixed[n], y_fixed[n]])))
#         #
#         # error = np.sum(err)
#         # print(x, y, error)
#
#         return obj
#
#         # # Invert function to find the minimum, if needed
#         # factor = 1.
#         # if self.extrema == self.MINIMUM:
#         #     factor = -1.
#         #
#         # fitness = factor * self.function([x, y])
#         #
#         # if self.logging_level >= 3:
#         #     print(solution, solution_idx, fitness)
#         #
#         # return obj
#
#     def execute(self):
#         """
#         Execute optimization.
#
#         Returns:
#         -------
#         solution, solution_fitness, solution_idx
#             Solution its fitness and its index in the list of cost function solutions.
#         """
#         x = self.data[0]
#         y = self.data[1]
#
#         # function_inputs = np.array([x, y]).T
#
#         # gene_space_min_x = np.min(x)
#         # gene_space_max_x = np.max(x)
#         # gene_space_min_y = np.min(y)
#         # gene_space_max_y = np.max(y)
#         #
#         # ga_instance = pygad.GA(num_generations=self.num_generations,
#         #                        num_parents_mating=5,
#         #                        initial_population=function_inputs,
#         #                        sol_per_pop=10,
#         #                        num_genes=len(function_inputs),
#         #                        gene_type=float,
#         #                        parent_selection_type='sss',
#         #                        gene_space=[
#         #                            {"low": gene_space_min_x, "high": gene_space_max_x},
#         #                            {"low": gene_space_min_y, "high": gene_space_max_y}
#         #                        ],
#         #                        keep_parents=-1,
#         #                        mutation_by_replacement=True,
#         #                        mutation_num_genes=1,
#         #                        # mutation_type=None,
#         #                        fitness_func=self._objective_func)
#         #
#         # ga_instance.run()
#
#         # print(function_inputs)
#
#         min_x = np.min(x)
#         max_x = np.max(x)
#         min_y = np.min(y)
#         max_y = np.max(y)
#         x0 = [min_x, min_y]
#
#         res = scipy.optimize.minimize(self._objective_func, x0,
#                                       # args=[x, y],
#                                       bounds=[(min_x, max_x), (min_y, max_y)],
#                                       method='trust-constr',
#                                       options={'maxiter': self.num_iterations})
#         # res = scipy.optimize.minimize(self.function, x0, method='L-BFGS-B',
#         #                               # bounds=bnds,
#         #                               options={'maxiter': self.num_iterations})
#         # res = scipy.optimize.fminbound(self.function, x, y)
#
#         # # Report convergence
#         # if self.logging_level >= 2:
#         #     ga_instance.plot_fitness()
#         #
#         if self.logging_level >= 1:
#             print('\n')
#             print('Solution:     ', res.x)
#             print('Fitness value: {fun}'.format(fun=res.fun))
#
#         return None, None, None
#
#         # return solution, solution_fitness
