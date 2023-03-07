import matplotlib.pyplot as plt
import numpy as np
from abc import ABCMeta, abstractmethod
import inspect

import pygad
import scipy


class GenericOptimization:
    """
    Base class for optimization algorithms
    """
    __metaclass__ = ABCMeta

    MAXIMUM = 0
    MINIMUM = 1

    def __init__(self, function, data, logging_level=1, extrema=MINIMUM):
        """
        Default constructor
        :param logging_level: Level of logging: 1 - only essential data; 2 - include plots; 3 - dump everything
        :param extrema: Define what type of problem to solve. 'extrema' can be equal to either MINIMUM or MAXIMUM. The
                        optimization algorithm will look for minimum and maximum values respectively.
        """
        self.logging_level = 1
        self.extrema = extrema
        self.function = function
        self.data = data

    def get_function(self):
        return self.function

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
        super().__init__(function, data, 1, self.MINIMUM)
        self.num_generations = num_generations

    def _objective_func(self, solution, solution_idx):
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
        :return: Solution and the corresponding function value
        """
        x = self.data[0]
        y = self.data[1]

        function_inputs = np.array([x, y]).T

        # gene_space_min_x = np.min(x)
        # gene_space_max_x = np.max(x)
        # gene_space_min_y = np.min(y)
        # gene_space_max_y = np.max(y)

        ga_instance = pygad.GA(num_generations=self.num_generations,
                               num_parents_mating=5,
                               initial_population=function_inputs,
                               sol_per_pop=10,
                               num_genes=len(function_inputs),
                               gene_type=float,
                               parent_selection_type='sss',
                               # gene_space=[
                               #     {"low": gene_space_min_x, "high": gene_space_max_x},
                               #     {"low": gene_space_min_y, "high": gene_space_max_y}
                               # ],
                               gene_space=[x, y],
                               keep_parents=-1,
                               mutation_by_replacement=True,
                               mutation_num_genes=1,
                               # mutation_type=None,
                               fitness_func=self._objective_func)

        ga_instance.run()

        # Report convergence
        if self.logging_level >= 2:
            ga_instance.plot_fitness()

        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

        if self.logging_level >= 1:
            print('\n')
            print('Solution:     ', solution)
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

    def construct_points(self, solution, num_points, delta_x, delta_y):
        """
        Constructs new set of values around the solution
        :param solution: List of solutions ([x, y])
        :param num_points: Number of points to construct
        :param delta_x: Offset for the 'x' variable in the solution
        :param delta_y: Offset for the 'y' variable in the solution
        :return: List of 'x' values, list of 'y' values and corresponding function values
        """
        step_x = 2 * delta_x / (num_points - 1)
        step_y = 2 * delta_y / (num_points - 1)

        x = solution[0]
        y = solution[1]

        range_x = np.arange(x - delta_x, x + delta_x + step_x, step_x)
        range_y = np.arange(y - delta_y, y + delta_y + step_y, step_y)

        c = [self.optimizer.get_function()([x_loc, y_loc]) for x_loc, y_loc in zip(range_x, range_y)]

        return range_x, range_y, c


def plot(x, y, z):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(x, y, z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)

    ax.contour(x, y, z, zdir='z', offset=np.min(z), cmap='coolwarm')
    ax.contour(x, y, z, zdir='x', offset=np.min(x), cmap='coolwarm')
    ax.contour(x, y, z, zdir='y', offset=np.max(y), cmap='coolwarm')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


class Test:
    def __init__(self, func_type='paraboloid', var_range=[1., 1.]):
        self.x = list(np.arange(-var_range[0], var_range[1], 0.25))
        self.y = list(np.arange(-var_range[0], var_range[1], 0.25))

        if func_type == 'paraboloid':
            self.function = self.paraboloid
        elif func_type == 'sixhump':
            self.function = self.sixhump
        elif func_type == 'rosenbrock':
            self.function = self.rosenbrock
        else:
            raise NotImplementedError('Unknown function type: {}'.format(func_type))

    def paraboloid(self, var):
        """
        A simple paraboloid function. Has one minimum:
        f(x1,x2)=0.0; (x1,x2)=(0.0, 0.0)
        :param var: List of x and y variables
        :return: Value of the function
        """
        x_loc = var[0]
        y_loc = var[1]
        return x_loc ** 2 + y_loc ** 2

    def sixhump(self, var):
        """
        The six-hump camel back function. Has two minimums:
        f(x1,x2)=-1.0316; (x1,x2)=(-0.0898,0.7126), (0.0898,-0.7126)
        :param var: List of x and y variables
        :return: Value of the function
        """
        x_loc = var[0]
        y_loc = var[1]
        return ((4 - 2.1 * x_loc**2 + (x_loc**4) / 3.) * x_loc**2 + x_loc * y_loc
                + (-4 + 4 * y_loc**2) * y_loc**2)

    def rosenbrock(self, var):
        """
        The Rosenbrock function. Has one minimum:
        f(x1,x2)=0.0; (x1,x2)=(1.0, 1.0)
        :param var: List of x and y variables
        :return: Value of the function
        """
        x_loc = var[0]
        y_loc = var[1]
        return (1 - x_loc)**2 + 100 * (y_loc - x_loc**2)**2

    def run(self):
        ga_opt = GAOpt(self.function, [self.x, self.y], 100)
        opt = Optimizer(ga_opt)

        # cg_opt = CGOpt(self.function, [self.x, self.y], 100)
        # opt = Optimizer(cg_opt)

        solution, func_values = opt.optimize()

        x_new, y_new, func_new = opt.construct_points(solution,
                                                      num_points=5,
                                                      delta_x=0.5,
                                                      delta_y=0.5)
        print('New values x:   ', x_new)
        print('New values y:   ', y_new)
        print('New values func:', func_new)

        x, y = np.meshgrid(self.x, self.y)
        c = self.function([x, y])
        plot(x, y, c)


if __name__ == "__main__":
    # Execute test
    test = Test('paraboloid')
    test.run()
