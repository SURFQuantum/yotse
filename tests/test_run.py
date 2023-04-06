import os
import json
import math
import pandas
import shutil
import unittest

from qiaopt.run import Core
from qiaopt.pre import Experiment, SystemSetup, Parameter
from qiaopt.optimization import GAOpt, Optimizer


if os.getcwd()[-5:] == "tests":
    DUMMY_FILE = "myfunction.py"
else:
    DUMMY_FILE = "tests/myfunction.py"


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

    @staticmethod
    def create_default_param(name="bright_state_parameter", parameter_range=[.1, .9], number_points=9,
                             distribution="linear", constraints={}, custom_distribution=None):
        return Parameter(name=name, param_range=parameter_range, number_points=number_points,
                         distribution=distribution, constraints=constraints, custom_distribution=custom_distribution)

    @staticmethod
    def create_default_experiment(parameters=None, opt_info_list=[]):
        return Experiment(experiment_name='default_exp',
                          system_setup=SystemSetup(
                              source_directory=os.getcwd(),
                              program_name=DUMMY_FILE,
                              command_line_arguments={'arg1': 1.0}),
                          parameters=parameters,
                          opt_info_list=opt_info_list)

    @staticmethod
    def create_default_core(num_params=1):
        test_param = [TestCore.create_default_param() for _ in range(num_params)]
        test_experiment = TestCore.create_default_experiment(parameters=test_param)
        return Core(test_experiment)

    def test_core_experiment(self):
        test_core = TestCore.create_default_core()
        self.assertTrue(isinstance(test_core.experiment, Experiment))

    def test_core_submit(self):
        test_core = TestCore.create_default_core()
        test_points = [1, 2, 3, 4]
        test_core.experiment.data_points = test_points
        job_ids = test_core.submit()

        self.assertEqual(len(test_points), len(job_ids))

        # count no of jobs
        self.path = test_core.experiment.system_setup.source_directory
        output_path = os.path.join(test_core.experiment.system_setup.working_directory, '..')
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

    def test_core_submit_with_analysis(self):
        """Check that when using an analysis script the right number of jobs are created as well."""
        analysis_exp = Experiment(experiment_name='default_exp',
                                  system_setup=SystemSetup(
                                      source_directory=os.getcwd(),
                                      program_name=DUMMY_FILE,
                                      command_line_arguments={'arg1': 1.0},
                                      analysis_script=DUMMY_FILE  # now with analysis script
                                  ),
                                  parameters=[self.create_default_param()],
                                  opt_info_list=[])

        test_core = Core(analysis_exp)
        test_points = [1, 2, 3, 4]
        test_core.experiment.data_points = test_points
        job_ids = test_core.submit()

        self.assertEqual(len(test_points) + 1, len(job_ids))                # now one extra analysis job

        # count no of jobs
        self.path = test_core.experiment.system_setup.source_directory
        output_path = os.path.join(test_core.experiment.system_setup.working_directory, '..')
        job_dirs = [d for d in os.listdir(output_path)]
        self.assertEqual(len(job_ids), len(job_dirs))
        # check if jobs were finishes successfully
        service_dirs = [f for f in os.listdir(self.path) if (f.startswith(".qcgpjm-service"))]
        with open(self.path + "/" + service_dirs[0] + "/" + "final_status.json", "r") as f:
            data = json.load(f)
        jobs_finished = data['JobStats']['FinishedJobs']
        jobs_failed = data['JobStats']['FailedJobs']
        self.assertEqual(jobs_finished, len(test_points) + 1)               # again one extra analysis job
        self.assertEqual(jobs_failed, 0)

    def test_core_collect_data(self):
        test_core = TestCore.create_default_core()
        test_path = os.path.join(os.getcwd(), 'temp_test_dir')
        os.makedirs(test_path)

        # test with analysis script
        test_core.experiment.system_setup.analysis_script = DUMMY_FILE
        test_core.experiment.system_setup.working_directory = test_path

        test_df = pandas.DataFrame({'f': [1, 2, 3], 'x': [4, 5, 6], 'y': [7, 8, 9]})
        # save dataframe as output.csv with whitespace as delimiter
        test_df.to_csv('output.csv', index=False, sep=' ')

        data = test_core.collect_data()

        self.assertEqual(type(data), pandas.DataFrame)
        self.assertTrue(data.equals(test_df))

        os.remove('output.csv')

        # test without analysis script
        test_core.experiment.system_setup.analysis_script = None
        os.makedirs(os.path.join(test_path, 'step0'))
        os.makedirs(os.path.join(test_path, 'step1'))
        os.makedirs(os.path.join(test_path, 'step2'))
        test_core.experiment.system_setup.working_directory = os.path.join(test_path, 'step2')

        test_df_1 = pandas.DataFrame({'f': [1], 'x': [4], 'y': [7]})
        test_df_2 = pandas.DataFrame({'f': [2], 'x': [5], 'y': [8]})
        test_df_3 = pandas.DataFrame({'f': [3], 'x': [6], 'y': [9]})
        # save dataframe as output.csv with whitespace as delimiter
        test_df_1.to_csv(os.path.join(test_path, 'step0', 'output.csv'), index=False, sep=' ')
        test_df_2.to_csv(os.path.join(test_path, 'step1', 'output.csv'), index=False, sep=' ')
        test_df_3.to_csv(os.path.join(test_path, 'step2', 'output.csv'), index=False, sep=' ')

        data = test_core.collect_data()

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

    def test_core_create_points_based_on_optimization(self):
        test_core = TestCore.create_default_core(num_params=2)

        def mock_function(solution, sol_idx):
            return solution[0]**2 + solution[1]**2

        initial_num_points = len(test_core.experiment.data_points)

        for evolutionary in [None, True, False]:
            ga_opt = GAOpt(initial_population=test_core.experiment.data_points,
                           num_generations=100,
                           num_parents_mating=20,
                           fitness_func=mock_function,
                           gene_type=float,
                           # num_genes=len(test_core.experiment.parameters),
                           mutation_probability=.4,
                           refinement_factors=[.1, .5])
            # self.assertTrue(ga_opt.ga_instance.allow_duplicate_genes is False)
            opt = Optimizer(ga_opt)
            test_core.optimizer = opt
            test_core.optimization_alg = ga_opt
            test_core._opt_is_evolutionary = True

            print("evol", evolutionary)
            x_list = [x for x, y in ga_opt.ga_instance.population]
            y_list = [y for x, y in ga_opt.ga_instance.population]
            c_list = [mock_function(sol, 0) for sol in ga_opt.ga_instance.population]
            data = pandas.DataFrame({'f': c_list, 'x': x_list, 'y': y_list})

            test_core.create_points_based_on_optimization(data=data, evolutionary=evolutionary)
            new_points = test_core.experiment.data_points
            self.assertIsInstance(new_points, list, list)                   # correct type
            self.assertEqual(len(new_points), initial_num_points)           # correct num points
            [self.assertEqual(len(point), 2) for point in new_points]       # each point has two param values
            s = set([tuple(x) for x in new_points])
            self.assertEqual(len(s), initial_num_points)                    # all points are unique
            best_solution, _, _ = test_core.optimization_alg.get_best_solution()
            assert all(math.isclose(.1, x) for x in best_solution)
            # todo: pick remaining points and iterate around them to create unique points of same pop_size

    def test_internal_dict(self):
        """Combined test for input_params_to_cost_value and update_internal_dict."""
        test_df = pandas.DataFrame({'f': [1, 2, 3], 'x': [4, 5, 6], 'y': [7, 8, 9]})
        test_df2 = pandas.DataFrame({'f': [.01, .02, .03], 'x': [.04, .05, .06], 'y': [.07, .08, .09]})
        test_dict = {(4, 7): 1, (5, 8): 2, (6, 9): 3}
        test_dict2 = {(.04, .07): .01, (.05, .08): .02, (.06, .09): .03}
        non_unique_df = pandas.DataFrame({'f': [1, 2, 1], 'x': [1, 1, 2], 'y': [1, 1, 2]})
        test_core = self.create_default_core()
        # test update_internal_dict
        self.assertEqual(test_core.input_cost_dict, {})
        test_core.update_internal_dict(data=test_df)
        self.assertEqual(test_core.input_cost_dict, test_dict)
        test_core.update_internal_dict(test_df2)
        self.assertEqual(test_core.input_cost_dict, test_dict2)
        # test input_params_to_cost_value
        self.assertEqual(test_core.input_params_to_cost_value([0.05, 0.08], 0), 0.02)
        # test float representation error
        self.assertEqual(test_core.input_params_to_cost_value([0.04999999999, 0.07999999999], 0), 0.02)
        with self.assertRaises(KeyError):
            test_core.input_params_to_cost_value([0.049, 0.079], 0)

        # non-unique data
        with self.assertRaises(Warning):
            # non-unique inputs with different costs: one solution gets lost -> raise Warning
            test_core.update_internal_dict(non_unique_df)

    def test_core_run(self):
        test_core = TestCore.create_default_core()
        test_points = [1, 2, 3, 4]
        test_core.experiment.data_points = test_points
        test_core.run()
        self.path = test_core.experiment.system_setup.source_directory  # path for tearDown


if __name__ == '__main__':
    unittest.main()
