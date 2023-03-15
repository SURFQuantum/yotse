import matplotlib.pyplot as plt
import numpy as np
from abc import ABCMeta, abstractmethod
import inspect

import pygad
import scipy

from qiaopt.pre import Parameter


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
        self.logging_level = logging_level
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
    def __init__(self, function, data, num_generations=100, logging_level=1):
        """
        Default constructor
        :param function: Fitness/objective function
        :param data: 2D data in a form of list [[], []]
        """
        super().__init__(function, data, logging_level, self.MINIMUM)
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

        ga_instance = pygad.GA(num_generations=self.num_generations,
                               num_parents_mating=5,
                               initial_population=function_inputs,
                               sol_per_pop=10,
                               num_genes=len(function_inputs),
                               gene_type=float,
                               parent_selection_type='sss',
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


class CGOpt(GenericOptimization):
    """
    Genetic algorithm
    """
    def __init__(self, function, data, num_iterations=100):
        """
        Default constructor
        :param function: Fitness/objective function
        :param data: 2D data in a form of list [[], []]
        """
        super().__init__(function, data, 1, self.MINIMUM)
        self.num_iterations = num_iterations

    def _objective_func(self, solution):
        """
        Fitness function to be called from PyGAD
        :param solution: List of solutions
        :param solution_idx: Index of solution
        :return: Fitness value
        """
        x, y = solution
        # x_fixed, y_fixed = args

        # Invert function to find the minimum, if needed
        factor = 1.
        if self.extrema == self.MAXIMUM:
            factor = -1.

        obj = factor * self.function([x, y])

        # err = []
        # for n in range(0, len(x_fixed)):
        #     err.append(np.abs(obj - self.function([x_fixed[n], y_fixed[n]])))
        #
        # error = np.sum(err)
        # print(x, y, error)

        return obj

        # # Invert function to find the minimum, if needed
        # factor = 1.
        # if self.extrema == self.MINIMUM:
        #     factor = -1.
        #
        # fitness = factor * self.function([x, y])
        #
        # if self.logging_level >= 3:
        #     print(solution, solution_idx, fitness)
        #
        # return obj

    def execute(self):
        """
        Execute optimization
        :return: Solution and the corresponding function value
        """
        x = self.data[0]
        y = self.data[1]

        # function_inputs = np.array([x, y]).T

        # gene_space_min_x = np.min(x)
        # gene_space_max_x = np.max(x)
        # gene_space_min_y = np.min(y)
        # gene_space_max_y = np.max(y)
        #
        # ga_instance = pygad.GA(num_generations=self.num_generations,
        #                        num_parents_mating=5,
        #                        initial_population=function_inputs,
        #                        sol_per_pop=10,
        #                        num_genes=len(function_inputs),
        #                        gene_type=float,
        #                        parent_selection_type='sss',
        #                        gene_space=[
        #                            {"low": gene_space_min_x, "high": gene_space_max_x},
        #                            {"low": gene_space_min_y, "high": gene_space_max_y}
        #                        ],
        #                        keep_parents=-1,
        #                        mutation_by_replacement=True,
        #                        mutation_num_genes=1,
        #                        # mutation_type=None,
        #                        fitness_func=self._objective_func)
        #
        # ga_instance.run()

        # print(function_inputs)

        min_x = np.min(x)
        max_x = np.max(x)
        min_y = np.min(y)
        max_y = np.max(y)
        x0 = [min_x, min_y]

        res = scipy.optimize.minimize(self._objective_func, x0,
                                      # args=[x, y],
                                      bounds=[(min_x, max_x), (min_y, max_y)],
                                      method='trust-constr',
                                      options={'maxiter': self.num_iterations})
        # res = scipy.optimize.minimize(self.function, x0, method='L-BFGS-B',
        #                               # bounds=bnds,
        #                               options={'maxiter': self.num_iterations})
        # res = scipy.optimize.fminbound(self.function, x, y)

        # # Report convergence
        # if self.logging_level >= 2:
        #     ga_instance.plot_fitness()
        #
        if self.logging_level >= 1:
            print('\n')
            print('Solution:     ', res.x)
            print('Fitness value: {fun}'.format(fun=res.fun))

        return None, None

        # return solution, solution_fitness


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

    def construct_points(self, solution, num_points, refinement_x, refinement_y):
        """
        Constructs new set of values around the solution
        :param solution: List of solutions ([x, y])
        :param num_points: Number of points to construct
        :param refinement_x: Refinement window for the 'x' variable in % from the (max-min)/2 range
        :param refinement_y: Refinement window for the 'y' variable in % from the (max-min)/2 range
        :return: List of ['x', 'y'] values and the corresponding list of function values ['f']
        """
        x = solution[0]
        y = solution[1]

        min_x = np.min(x)
        max_x = np.max(x)
        min_y = np.min(y)
        max_y = np.max(y)

        delta_x = refinement_x * (max_x - min_x) * 0.5
        delta_y = refinement_y * (max_y - min_y) * 0.5

        # step_x = 2 * delta_x / (num_points - 1)
        # step_y = 2 * delta_y / (num_points - 1)

        distribution_x = 'linear'
        distribution_y = 'linear'
        param_x = Parameter('x', [x - delta_x, x + delta_x], num_points, distribution_x)
        param_y = Parameter('y', [y - delta_y, y + delta_y], num_points, distribution_y)

        f = [self.optimizer.get_function()([x_loc, y_loc])
             for x_loc, y_loc in zip(param_x.data_points, param_y.data_points)]

        combined_ranges = [[param_x.data_points[i], param_y.data_points[i]]
                           for i in range(0, len(param_x.data_points))]

        return combined_ranges, f


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
    def __init__(self, func_type='paraboloid', var_range=[1.2, 1.2]):
        var_step = 0.2
        self.x = list(np.arange(-var_range[0], var_range[1], var_step))
        self.y = list(np.arange(-var_range[0], var_range[1], var_step))

        if func_type == 'paraboloid':
            self.function = self.paraboloid
        elif func_type == 'sixhump':
            self.function = self.sixhump
        elif func_type == 'rosenbrock':
            self.function = self.rosenbrock
        elif func_type == 'rastrigin':
            self.function = self.rastrigin
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
        return (4 - 2.1 * x_loc ** 2 + (x_loc ** 4) / 3.) * x_loc**2 + x_loc * y_loc + (-4 + 4 * y_loc ** 2) * y_loc**2

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

    def rastrigin(self, var):
        x_loc = var[0]
        y_loc = var[1]
        return (x_loc ** 2 - 10 * np.cos(2 * np.pi * x_loc)) + \
            (y_loc ** 2 - 10 * np.cos(2 * np.pi * y_loc)) + 20

    def run(self):
        ga_opt = GAOpt(self.function, [self.x, self.y], 100)
        opt = Optimizer(ga_opt)

        # cg_opt = CGOpt(self.function, [self.x, self.y], 100)
        # opt = Optimizer(cg_opt)

        solution, func_values = opt.optimize()

        xy_new, func_new = opt.construct_points(solution,
                                                num_points=5,
                                                delta_x=0.5,
                                                delta_y=0.5)
        print('New values [x, y]:   ', xy_new)
        print('New values func:', func_new)

        x, y = np.meshgrid(self.x, self.y)
        c = self.function([x, y])
        plot(x, y, c)


if __name__ == "__main__":
    # Execute test
    test = Test('rastrigin')
    test.run()
