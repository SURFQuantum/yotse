import unittest
import numpy as np

from qiaopt.optimization import GAOpt, Optimizer


class TestGAOpt(unittest.TestCase):

    @staticmethod
    def _paraboloid(ga_instance, solution, sol_index):
        """
        A simple paraboloid function. Has one global minimum:
        f(x1,x2)=0.0; (x1,x2)=(0.0, 0.0)
        :param solution: List of x and y variables
        :param sol_index: index related to this solution
        :return: Value of the function
        """
        x_loc = solution[0]
        y_loc = solution[1]
        return x_loc ** 2 + y_loc ** 2

    @staticmethod
    def _sixhump(ga_instance, solution, sol_index):
        """
        The six-hump camel back function. Has two global minimums:
        f(x1,x2)=-1.0316; (x1,x2)=(-0.0898,0.7126), (0.0898,-0.7126)
        :param solution: List of x and y variables
        :param sol_index: index related to this solution
        :return: Value of the function
        """
        x_loc = solution[0]
        y_loc = solution[1]
        return (4 - 2.1 * x_loc ** 2 + (x_loc ** 4) / 3.) * x_loc**2 + x_loc * y_loc + (-4 + 4 * y_loc ** 2) * y_loc**2

    @staticmethod
    def _rosenbrock(ga_instance, solution, sol_index):
        """
        The Rosenbrock function. Has one global minimum:
        f(x1,x2)=0.0; (x1,x2)=(1.0, 1.0)
        :param solution: List of x and y variables
        :param sol_index: index related to this solution
        :return: Value of the function
        """
        x_loc = solution[0]
        y_loc = solution[1]
        return (1 - x_loc)**2 + 100 * (y_loc - x_loc**2)**2

    @staticmethod
    def _rastrigin(ga_instance, solution, sol_index):
        """
        The Rastrigin function. Has one global minimum:
        f(x1,x2)=0.0; (x1,x2)=(0.0, 0.0)
        :param solution: List of x and y variables
        :param sol_index: index related to this solution
        :return: Value of the function
        """
        x_loc = solution[0]
        y_loc = solution[1]
        return (x_loc ** 2 - 10 * np.cos(2 * np.pi * x_loc)) + \
            (y_loc ** 2 - 10 * np.cos(2 * np.pi * y_loc)) + 20

    def _setup_and_execute(self, function, var_range=(1.2, 1.2), var_step=0.01):
        self.x = list(np.arange(-var_range[0], var_range[1], var_step))
        self.y = list(np.arange(-var_range[0], var_range[1], var_step))
        initial_pop = []
        for i in range(len(self.x)):
            initial_pop.append([self.x[i], self.y[i]])

        ga_opt = GAOpt(initial_population=initial_pop,
                       num_generations=100,
                       num_parents_mating=10,
                       fitness_func=function,
                       gene_type=float,
                       mutation_probability=.1,)
        opt = Optimizer(ga_opt)

        for _ in range(ga_opt.ga_instance.num_generations):
            opt.optimize()
            ga_opt.ga_instance.initial_population = ga_opt.ga_instance.population

        # matplotlib.use('Qt5Agg')
        # ga_opt.ga_instance.plot_fitness()
        return ga_opt.get_best_solution()

    def test_optimize_paraboloid(self):
        solution, solution_fitness, solution_index = self._setup_and_execute(self._paraboloid)
        x_true = 0.0
        y_true = 0.0

        self.assertTrue(np.abs(solution[0] - x_true) <= 1e-12)
        self.assertTrue(np.abs(solution[1] - y_true) <= 1e-12)

    def test_optimize_sixhump(self):
        solution, solution_fitness, solution_index = self._setup_and_execute(self._sixhump,
                                                                             var_range=[0.8, 0.8], var_step=0.01)
        x_true = -0.0898
        y_true = 0.7126

        # Check one of the possible solutions
        if solution[0] > 0:
            x_true *= -1
            y_true *= -1
        self.assertTrue(np.abs(solution[0] - x_true) <= 1e-2)
        self.assertTrue(np.abs(solution[1] - y_true) <= 1e-2)

    def test_optimize_rosenbrock(self):
        solution, solution_fitness, solution_index = self._setup_and_execute(self._rosenbrock)
        x_true = 1.0
        y_true = 1.0

        self.assertTrue(np.abs(solution[0] - x_true) <= 1e-12)
        self.assertTrue(np.abs(solution[1] - y_true) <= 1e-12)

    def test_optimize_rastrigin(self):
        solution, solution_fitness, solution_index = self._setup_and_execute(self._rastrigin)
        x_true = 0.0
        y_true = 0.0

        self.assertTrue(np.abs(solution[0] - x_true) <= 1e-12)
        self.assertTrue(np.abs(solution[1] - y_true) <= 1e-12)


if __name__ == '__main__':
    unittest.main()
