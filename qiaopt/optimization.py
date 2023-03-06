import matplotlib.pyplot as plt
import numpy as np
from abc import ABCMeta, abstractmethod
import inspect

import pygad


class GenericOptimization:
    """
    Base class for optimization algorithms
    """
    __metaclass__ = ABCMeta

    MAXIMUM = 0
    MINIMUM = 1

    def __init__(self, logging_level=1, extrema=MINIMUM):
        """
        Default constructor
        :param logging_level: Level of logging: 1 - only essential data; 2 - include plots; 3 - dump everything
        :param extrema: Define what type of problem to solve. 'extrema' can be equal to either MINIMUM or MAXIMUM. The
                        optimization algorithm will look for minimum and maximum values respectively.
        """
        self.logging_level = 1
        self.extrema = extrema

    @abstractmethod
    def execute(self):
        """
        Execute method should be implemented in every derived class
        :return: Solution and the corresponding function value
        """
        raise NotImplementedError('The \'{}\' method is not implemented'.format(inspect.currentframe().f_code.co_name))
        # return None, None


class GAOpt(GenericOptimization):
    """
    Genetic algorithm
    """
    def __init__(self, function, data, num_generations=100):
        """
        Default constructor
        :param function: Fitness/objective function
        :param data: 2D data in a form of list [[], []]
        """
        super().__init__(1, self.MINIMUM)
        self.function = function
        self.data = data
        self.num_generations = num_generations

    def _fitness_func(self, solution, solution_idx):
        """
        Fitness function to be called from PyGAD
        :param solution: List of solutions
        :param solution_idx: Index of solution
        :return: Fitness value
        """
        x = solution[0]
        y = solution[1]

        # Invert function to find the minimum, if needed
        factor = 1.
        if self.extrema == self.MINIMUM:
            factor = -1.

        fitness = factor * self.function([x, y])

        if self.logging_level >= 3:
            print(solution, solution_idx, fitness)

        return fitness

    def execute(self):
        """
        Execute optimization
        :num_generations: Number of generations
        :return: Solution and the corresponding function value
        """
        x = self.data[0]
        y = self.data[1]

        function_inputs = np.array([x, y]).T

        gene_space_min_x = np.min(x)
        gene_space_max_x = np.max(x)
        gene_space_min_y = np.min(y)
        gene_space_max_y = np.max(y)

        ga_instance = pygad.GA(num_generations=self.num_generations,
                               num_parents_mating=5,
                               initial_population=function_inputs,
                               sol_per_pop=10,
                               num_genes=len(function_inputs[0]),
                               gene_type=float,
                               parent_selection_type='sss',
                               gene_space=[
                                   {"low": gene_space_min_x, "high": gene_space_max_x},
                                   {"low": gene_space_min_y, "high": gene_space_max_y}
                               ],
                               keep_parents=-1,
                               mutation_by_replacement=True,
                               mutation_num_genes=1,
                               # mutation_type=None,
                               fitness_func=self._fitness_func)

        ga_instance.run()

        # Report convergence
        if self.logging_level >= 2:
            ga_instance.plot_fitness()

        if self.logging_level >= 1:
            solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
            print('\n')
            print('Solution:      ', solution)
            print('Fitness value: {solution_fitness}'.format(solution_fitness=solution_fitness))

        return solution, solution_fitness


class Optimizer:
    """
    Optimizer class
    """
    def __init__(self, optimizer):
        """
        Default constructor
        :param optimizer: Object of optimizer
        """
        self.optimizer = optimizer

    def optimize(self):
        """
        Optimization step
        :return: Solution and the corresponding function value
        """
        solution, func_value = self.optimizer.execute()

        return solution, func_value


def plot(x, y, z):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(x, y, z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)

    ax.contour(x, y, z, zdir='z', offset=np.min(z), cmap='coolwarm')
    ax.contour(x, y, z, zdir='x', offset=np.min(x), cmap='coolwarm')
    ax.contour(x, y, z, zdir='y', offset=np.max(y), cmap='coolwarm')

    plt.show()


def test():
    size = 50
    x = list(range(-size, size+1))
    y = list(range(-size, size+1))

    def func(var):
        x_loc = var[0]
        y_loc = var[1]
        return x_loc ** 2 + y_loc ** 2

    ga_opt = GAOpt(func, [x, y], 100)
    opt = Optimizer(ga_opt)

    opt.optimize()

    x, y = np.meshgrid(x, y)
    c = func([x, y])
    plot(x, y, c)


if __name__ == "__main__":
    # Execute
    test()
