import os
import shutil
import unittest
import json
import numpy

import pandas

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
                             distribution="linear", constraints=[None], custom_distribution=None):
        return Parameter(name=name, parameter_range=parameter_range, number_points=number_points,
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
    def create_default_core():
        test_param = TestCore.create_default_param()
        test_experiment = TestCore.create_default_experiment(parameters=[test_param])
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

    def test_core_collect_data(self):
        test_core = TestCore.create_default_core()
        test_points = [1, 2, 3, 4]
        test_core.experiment.data_points = test_points
        test_core.submit()

        self.path = test_core.experiment.system_setup.source_directory
        data = test_core.collect_data()
        self.assertEqual(len(data), len(test_points)*100)
        self.assertTrue(isinstance(data, pandas.DataFrame))
        # todo: fix parsing args and check that data has at least n_args columns

    # def test_core_multiple_collect_data(self):
    #     test_core = TestCore.create_default_core()
    #     test_points = [1, 2, 3, 4]
    #     test_core.experiment.data_points = test_points
    #     path = test_core.experiment.system_setup.source_directory
    #     test_core.experiment.system_setup.files_needed = 'myfunction.py'
    #     data = []
    #     for step in {'one', 'two', 'three'}:
    #         os.chdir(path)
    #         tmppath = path + "/" + step
    #         if not os.path.exists(tmppath):
    #             os.mkdir(step)
    #         shutil.copyfile(path + "/" + test_core.experiment.system_setup.files_needed,
    #                         tmppath + "/" + test_core.experiment.system_setup.files_needed)
    #         test_core.experiment.system_setup.source_directory = tmppath
    #         test_core.experiment.system_setup.stdout_basename = "output" + step
    #         os.chdir(tmppath)
    #         test_core.submit()
    #         data.append(test_core.collect_data(tmppath))
    #         [f for f in os.listdir(tmppath) if (f.startswith(".qcg"))]
    #         shutil.rmtree(tmppath)
    #     self.assertEqual(len(data), 3)
    #     # TODO check elements are lists, check length
    #     path = os.getcwd()

    def test_core_create_points_based_on_method(self):
        test_core = TestCore.create_default_core()

        def mock_function(x, y):
            return x**2 + y**2

        mock_data = numpy.random.rand(2, 100)
        mock_df = pandas.DataFrame(data=mock_data)

        ga_opt = GAOpt(function=mock_function, num_generations=10)
        opt = Optimizer(ga_opt)
        test_core.optimizer = opt
        test_core.optimization_alg = ga_opt

        new_points = test_core.create_points_based_on_method(data=mock_df)
        self.assertIsInstance(new_points, list, list)
        # TODO check type of new_points.. what type do we want?


    def test_core_run(self):
        test_core = TestCore.create_default_core()
        test_points = [1, 2, 3, 4]
        test_core.experiment.data_points = test_points
        test_core.run()
        self.path = test_core.experiment.system_setup.source_directory  # path for tearDown


if __name__ == '__main__':
    unittest.main()
