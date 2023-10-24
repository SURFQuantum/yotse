import os
import unittest

import pandas as pd
import pytest

from yotse.execution import Executor
from yotse.optimization.ga import ModGA
from yotse.pre import Experiment
from yotse.pre import OptimizationInfo
from yotse.pre import Parameter
from yotse.pre import SystemSetup


if os.getcwd().endswith("tests"):
    DUMMY_FILE = "myfunction.py"
else:
    DUMMY_FILE = "tests/myfunction.py"


class TestNewOpt(unittest.TestCase):
    @staticmethod
    def create_default_param(
        name="bright_state_parameter",
        parameter_range=[0.1, 0.9],
        number_points=9,
        distribution="linear",
        constraints=None,
        custom_distribution=None,
    ):
        return Parameter(
            name=name,
            param_range=parameter_range,
            number_points=number_points,
            distribution=distribution,
            constraints=constraints,
            custom_distribution=custom_distribution,
        )

    @staticmethod
    def create_default_experiment(parameters=None, opt_info_list=[]):
        return Experiment(
            experiment_name="default_exp",
            system_setup=SystemSetup(
                source_directory=os.getcwd(),
                program_name=DUMMY_FILE,
                command_line_arguments={"arg1": 1.0},
            ),
            parameters=parameters,
            opt_info_list=opt_info_list,
        )

    @staticmethod
    def create_default_executor(experiment):
        return Executor(experiment=experiment)

    def setUp(self) -> None:
        self.lookup_dict = {
            (1, 2, 3): 1,
            (2, 3, 4): 1.2,
            (5, 6, 7): 2,
            (8, 9, 10): 0.1,
            (11, 11, 11): 3,
        }
        self.initial_pop = list(self.lookup_dict.keys())
        df = pd.DataFrame.from_dict(
            self.lookup_dict, orient="index", columns=["Values"]
        )

        # Create separate columns for each part of the key tuples
        df[["Column1", "Column2", "Column3"]] = pd.DataFrame(
            df.index.tolist(), index=df.index
        )
        self.data = df

    @staticmethod
    def setup_ga_instance(fitness_func, initial_pop, num_gen):
        return ModGA(
            num_generations=num_gen,
            num_parents_mating=2,
            fitness_func=fitness_func,
            gene_type=int,
            mutation_probability=0.1,
            initial_population=initial_pop,
        )

    def cost_func(self, ga_instance, solution, sol_idx):
        return -1 * (solution[0] ** 2 + solution[1] ** 2 + solution[2] ** 2)

    def test_population_lookup(self):
        # todo : this test seems to still test something that no other test picks up (aka the input_to_cost_value func)
        print("Initial pop has size", len(self.initial_pop))
        test_param = [TestNewOpt.create_default_param() for _ in range(3)]
        test_exp = self.create_default_experiment(
            parameters=test_param,
            opt_info_list=[
                OptimizationInfo(
                    name="GA",
                    opt_parameters={
                        "num_generations": 10,
                        "num_parents_mating": 2,
                        "gene_type": int,
                        "mutation_probability": 0.1,
                        "refinement_factors": [0.5, 0.5],
                        "logging_level": 1,
                    },
                    is_active=True,
                )
            ],
        )
        test_exp.data_points = self.initial_pop
        test_exec = self.create_default_executor(experiment=test_exp)
        test_exec.optimizer.optimization_algorithm.update_internal_cost_data(
            experiment=test_exp, data=self.data
        )

        self.assertEqual(
            test_exec.optimizer.optimization_algorithm.optimization_instance.pop_size,
            (len(self.initial_pop), len(self.initial_pop[0])),
        )
        test_exec.optimizer.optimization_algorithm.optimization_instance.run_single_generation()

        self.assertEqual(
            test_exec.optimizer.optimization_algorithm.optimization_instance.generations_completed,
            1,
        )

    def test_new_population_func(self):
        num_generations = 100

        ga_instance = self.setup_ga_instance(
            fitness_func=self.cost_func,
            initial_pop=self.initial_pop,
            num_gen=num_generations,
        )
        print("initial population", self.initial_pop)
        for _ in range(ga_instance.num_generations):
            ga_instance.run_single_generation()
            ga_instance.population
            # print("new population", ga_instance.population)
            # ga_instance.initial_population = ga_instance.population

        assert ga_instance.generations_completed == num_generations
        # print(ga_instance.population)
        # print(ga_instance.best_solution())
        # matplotlib.use('Qt5Agg')
        # ga_instance.plot_fitness()
        assert ga_instance.best_solution()[0][0] == 0
        assert ga_instance.best_solution()[0][1] == 0
        assert ga_instance.best_solution()[0][2] == 0


class TestGA(unittest.TestCase):
    @pytest.mark.xfail(
        reason="pygad can not guarantee uniqueness of genes even with allow_duplicate_genes=False."
    )
    def test_non_uniqueness(self):
        """Minimal working example to demonstrate how mutation ruins the gene_space and allow_duplicate_genes params."""
        import pygad
        import numpy
        import itertools

        def mock_function(ga_instance, solution, sol_idx):
            return -1 * (solution[0] ** 2 + solution[1] ** 2)

        range = [0.1, 1.0]
        param_values = numpy.linspace(range[0], range[1], 10).tolist()
        initial_population = list(
            {element for element in itertools.product(param_values, param_values)}
        )

        # Remove duplicates from the initial population.
        for idx, sol in enumerate(initial_population):
            sol = list(sol)
            if sol[0] == sol[1]:
                if sol[1] < 0.101:
                    sol[1] = sol[1] + 0.001
                else:
                    sol[1] = sol[1] - 0.001
            initial_population[idx] = sol.copy()

        ga = pygad.GA(
            num_generations=100,
            num_parents_mating=10,
            sol_per_pop=100,
            num_genes=2,
            gene_type=float,
            gene_space={"low": range[0], "high": range[1]},
            fitness_func=mock_function,
            initial_population=initial_population,
            mutation_probability=0.2,  # this is the line that makes or brakes it
            allow_duplicate_genes=False,
            mutation_by_replacement=True,
        )
        ga.run()

        new_points = ga.population
        unique_points = set([tuple(x) for x in new_points])
        print(len(initial_population), len(new_points), len(unique_points))
        print(f"Best solution is {ga.best_solution()[0]}.")
        for point in new_points:
            for x in point:
                if range[0] > x:
                    print(f"Point {x} is outside gene_space ({range}).")
                elif x > range[1]:
                    print(f"Point {x} is outside gene_space ({range}).")
            # self.assertTrue(all(range[0] <= x <= range[1] for x in point))
        self.assertEqual(len(initial_population), len(unique_points))


if __name__ == "__main__":
    unittest.main()
