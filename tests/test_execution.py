import os
import json
import math
import pytest
import pandas
import shutil
import unittest
from unittest.mock import Mock

from yotse.pre import Experiment, SystemSetup, Parameter
from yotse.optimization.algorithms import GAOpt
from yotse.optimization.optimizer import Optimizer
from yotse.optimization.generic_optimization import GenericOptimization
from yotse.execution import Executor


if os.getcwd().endswith("tests"):
    DUMMY_FILE = "myfunction.py"
else:
    DUMMY_FILE = "tests/myfunction.py"


def create_default_param(name="bright_state_parameter", parameter_range=[.1, .9], number_points=9,
                         distribution="linear", constraints=None, custom_distribution=None):
    return Parameter(name=name, param_range=parameter_range, number_points=number_points,
                     distribution=distribution, constraints=constraints, custom_distribution=custom_distribution)


def create_default_experiment(parameters=None, opt_info_list=[]):
    return Experiment(experiment_name='default_exp',
                      system_setup=SystemSetup(
                          source_directory=os.getcwd(),
                          program_name=DUMMY_FILE,
                          command_line_arguments={'arg1': 1.0}),
                      parameters=parameters,
                      opt_info_list=opt_info_list)


def create_default_executor(experiment):
    return Executor(experiment=experiment)


class TestExecutor(unittest.TestCase):
    """Test the executor class."""

    def setUp(self) -> None:
        self.path = None  # path for tearDown
        # self.tearDown()

    def tearDown(self) -> None:
        [os.remove(f'stdout{i}.txt') for i in range(4) if os.path.exists(f'stdout{i}.txt')]
        if self.path is not None:
            [os.remove(f) for f in os.listdir(self.path) if f.endswith('.csv')]
            shutil.rmtree(os.path.join(self.path, 'output'))
            dirs = [f for f in os.listdir(self.path) if (f.startswith(".qcg"))]
            for d in dirs:
                shutil.rmtree(os.path.join(self.path, d))
            self.path = None

    def test_executor_experiment_input(self):
        test_exp = create_default_experiment()
        test_exec = create_default_executor(experiment=test_exp)
        self.assertTrue(isinstance(test_exec.experiment, Experiment))
        self.assertEqual(test_exec.experiment, test_exp)

    def test_executor_submit(self):
        test_exp = create_default_experiment()
        test_exec = create_default_executor(experiment=test_exp)
        test_points = [1, 2, 3, 4]
        test_exec.experiment.data_points = test_points
        job_ids = test_exec.submit()

        self.assertEqual(len(test_points), len(job_ids))

        # count no of jobs
        self.path = test_exec.experiment.system_setup.source_directory
        output_path = os.path.join(test_exec.experiment.system_setup.working_directory, '..')
        job_dirs = [d for d in os.listdir(output_path)]
        self.assertEqual(len(job_ids), len(job_dirs))
        # check if jobs were finishes successfully
        service_dirs = [f for f in os.listdir(self.path) if (f.startswith(".qcgpjm-service"))]
        with open(self.path + "/" + service_dirs[0] + "/" + "final_status.json", "r") as f:
            data = json.load(f)
        jobs_finished = data['JobStats']['FinishedJobs']
        jobs_failed = data['JobStats']['FailedJobs']
        self.assertEqual(jobs_finished, len(test_points))
        self.assertEqual(jobs_failed, 0)

    def test_executor_submit_with_analysis(self):
        """Check that when using an analysis script the right number of jobs are created as well."""
        analysis_exp = Experiment(experiment_name='default_exp',
                                  system_setup=SystemSetup(
                                      source_directory=os.getcwd(),
                                      program_name=DUMMY_FILE,
                                      command_line_arguments={'arg1': 1.0},
                                      analysis_script=DUMMY_FILE  # now with analysis script
                                  ),
                                  parameters=[create_default_param()],
                                  opt_info_list=[])

        test_exec = create_default_executor(analysis_exp)
        test_points = [1, 2, 3, 4]
        test_exec.experiment.data_points = test_points
        job_ids = test_exec.submit()

        self.assertEqual(len(test_points) + 1, len(job_ids))                # now one extra analysis job

        # count no of jobs
        self.path = test_exec.experiment.system_setup.source_directory
        output_path = os.path.join(test_exec.experiment.system_setup.working_directory, '..')
        job_dirs = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]
        self.assertEqual(len(job_ids) - 1, len(job_dirs))                    # for the analysis job no dir is created
        self.assertEqual(len(job_dirs) + 2, len([d for d in os.listdir(output_path)]))      # but 2 files are created
        # check if jobs were finishes successfully
        service_dirs = [f for f in os.listdir(self.path) if (f.startswith(".qcgpjm-service"))]
        with open(self.path + "/" + service_dirs[0] + "/" + "final_status.json", "r") as f:
            data = json.load(f)
        jobs_finished = data['JobStats']['FinishedJobs']
        jobs_failed = data['JobStats']['FailedJobs']
        self.assertEqual(jobs_finished, len(test_points) + 1)               # again one extra analysis job
        self.assertEqual(jobs_failed, 0)

    def test_executor_run(self):
        test_exec = create_default_executor(experiment=create_default_experiment())
        test_points = [1, 2, 3, 4]
        test_exec.experiment.data_points = test_points
        test_exec.run()
        # todo: this tests nothing! add test
        self.path = test_exec.experiment.system_setup.source_directory  # path for tearDown

    def test_executor_collect_data(self):
        test_exec = create_default_executor(create_default_experiment())
        test_path = os.path.join(os.getcwd(), 'temp_test_dir')
        os.makedirs(test_path)

        # test with analysis script
        test_exec.experiment.system_setup.analysis_script = DUMMY_FILE
        test_exec.experiment.system_setup.working_directory = test_path

        test_df = pandas.DataFrame({'f': [1, 2, 3], 'x': [4, 5, 6], 'y': [7, 8, 9]})
        # save dataframe as output.csv with whitespace as delimiter
        test_df.to_csv('output.csv', index=False, sep=' ')

        data = test_exec.collect_data()

        self.assertEqual(type(data), pandas.DataFrame)
        self.assertTrue(data.equals(test_df))

        os.remove('output.csv')

        # test without analysis script
        test_exec.experiment.system_setup.analysis_script = None
        os.makedirs(os.path.join(test_path, 'step0'))
        os.makedirs(os.path.join(test_path, 'step1'))
        os.makedirs(os.path.join(test_path, 'step2'))
        test_exec.experiment.system_setup.working_directory = os.path.join(test_path, 'step2')

        test_df_1 = pandas.DataFrame({'f': [1], 'x': [4], 'y': [7]})
        test_df_2 = pandas.DataFrame({'f': [2], 'x': [5], 'y': [8]})
        test_df_3 = pandas.DataFrame({'f': [3], 'x': [6], 'y': [9]})
        # save dataframe as output.csv with whitespace as delimiter
        test_df_1.to_csv(os.path.join(test_path, 'step0', 'output.csv'), index=False, sep=' ')
        test_df_2.to_csv(os.path.join(test_path, 'step1', 'output.csv'), index=False, sep=' ')
        test_df_3.to_csv(os.path.join(test_path, 'step2', 'output.csv'), index=False, sep=' ')

        data = test_exec.collect_data()

        # the columns here are not sorted so rearrange to match test_df
        data = data.sort_values(by='f')
        data = data.reset_index(drop=True)

        self.assertEqual(type(data), pandas.DataFrame)
        self.assertTrue(data.equals(test_df))

        os.remove(os.path.join(test_path, 'step0', 'output.csv'))
        os.remove(os.path.join(test_path, 'step1', 'output.csv'))
        os.remove(os.path.join(test_path, 'step2', 'output.csv'))
        os.removedirs(os.path.join(test_path, 'step0'))
        os.removedirs(os.path.join(test_path, 'step1'))
        os.removedirs(os.path.join(test_path, 'step2'))

    def test_executor_create_points(self):
        """Test create_points_based_on_optimization."""

        def mock_function(ga_instance, solution, solution_idx):
            return solution[0]**2 + solution[1]**2

        for evolutionary in [None, True, False]:
            test_executor = create_default_executor(create_default_experiment(num_params=2))
            initial_num_points = len(test_executor.experiment.data_points)
            ga_opt = GAOpt(initial_population=test_executor.experiment.data_points,
                           num_generations=100,
                           num_parents_mating=9,
                           fitness_func=mock_function,
                           gene_type=float,
                           gene_space={'low': .1, 'high': .9},
                           mutation_probability=.02,                            # todo: this line breaks it
                           crossover_probability=.7,
                           refinement_factors=[.1, .5],
                           allow_duplicate_genes=False,
                           )
            # self.assertTrue(ga_opt.ga_instance.allow_duplicate_genes is False)
            opt = Optimizer(ga_opt)
            test_executor.optimizer = opt
            test_executor.optimization_alg = ga_opt
            test_executor._opt_is_evolutionary = True

            x_list = [x for x, y in ga_opt.optimization_instance.population]
            y_list = [y for x, y in ga_opt.optimization_instance.population]
            c_list = [mock_function(ga_instance=test_executor.optimization_alg.optimization_instance,
                                    solution=sol, solution_idx=0) for sol in ga_opt.optimization_instance.population]
            data = pandas.DataFrame({'f': c_list, 'x': x_list, 'y': y_list})

            test_executor.create_points_based_on_optimization(data=data, evolutionary=evolutionary)
            self.assertEqual(test_executor.optimization_alg.optimization_instance.generations_completed, 1)
            new_points = test_executor.experiment.data_points
            self.assertIsInstance(new_points, list, list)                   # correct type
            self.assertEqual(len(new_points), initial_num_points)           # correct num points
            [self.assertEqual(len(point), 2) for point in new_points]       # each point has two param values
            # s = set([tuple(x) for x in new_points])
            # self.assertEqual(len(s), initial_num_points)                    # all points are unique
            if evolutionary is not False:
                for point in new_points:
                    self.assertTrue(all(.1 <= x <= .9 for x in point))          # each point is within constraint
            best_solution, _, _ = test_executor.optimization_alg.get_best_solution()
            assert all(math.isclose(.1, x) for x in best_solution)


class TestCore(unittest.TestCase):
    """Test the Core class."""
    def setUp(self) -> None:
        self.path = None  # path for tearDown

    def tearDown(self) -> None:
        [os.remove(f'stdout{i}.txt') for i in range(4) if os.path.exists(f'stdout{i}.txt')]
        if self.path is not None:
            [os.remove(f) for f in os.listdir(self.path) if f.endswith('.csv')]
            shutil.rmtree(os.path.join(self.path, 'output'))
            dirs = [f for f in os.listdir(self.path) if (f.startswith(".qcg"))]
            for d in dirs:
                shutil.rmtree(os.path.join(self.path, d))
            self.path = None

    @pytest.mark.xfail(reason="pygad can not guarantee uniqueness of genes even with allow_duplicate_genes=False.")
    def test_executor_create_points_uniqueness(self):
        """Test create_points_based_on_optimization."""
        # todo: merge this test with the above once uniqueness is fixed

        def mock_function(ga_instance, solution, solution_idx):
            return solution[0]**2 + solution[1]**2

        for evolutionary in [None, True, False]:
            test_core = TestCore.create_default_core(num_params=2)
            initial_num_points = len(test_core.experiment.data_points)
            ga_opt = GAOpt(initial_population=test_core.experiment.data_points,
                           num_generations=100,
                           num_parents_mating=9,
                           fitness_func=mock_function,
                           gene_type=float,
                           gene_space={'low': .1, 'high': .9},
                           mutation_probability=.02,                            # todo: this line break it
                           crossover_probability=.7,
                           refinement_factors=[.1, .5],
                           allow_duplicate_genes=False,
                           )
            # self.assertTrue(ga_opt.ga_instance.allow_duplicate_genes is False)
            opt = Optimizer(ga_opt)
            test_core.optimizer = opt
            test_core.optimization_alg = ga_opt
            test_core._opt_is_evolutionary = True

            x_list = [x for x, y in ga_opt.optimization_instance.population]
            y_list = [y for x, y in ga_opt.optimization_instance.population]
            c_list = [mock_function(ga_instance=test_core.optimization_alg.optimization_instance,
                                    solution=sol, solution_idx=0) for sol in ga_opt.optimization_instance.population]
            data = pandas.DataFrame({'f': c_list, 'x': x_list, 'y': y_list})

            test_core.create_points_based_on_optimization(data=data, evolutionary=evolutionary)
            self.assertEqual(test_core.optimization_alg.optimization_instance.generations_completed, 1)
            new_points = test_core.experiment.data_points
            s = set([tuple(x) for x in new_points])
            self.assertEqual(len(s), initial_num_points)                    # all points are unique


if __name__ == '__main__':
    unittest.main()
