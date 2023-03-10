import os
import shutil
import unittest
import json
from qiaopt.run import Core
from qiaopt.pre import Experiment, SystemSetup, Parameter


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
            shutil.rmtree(self.path + '/output')
            dirs = [f for f in os.listdir(self.path) if (f.startswith(".qcg"))]
            for d in dirs:
                shutil.rmtree(self.path + "/" + d)
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
        test_experiment = TestCore.create_default_experiment(parameters=test_param)
        return Core(test_experiment)

    def test_core_experiment(self):
        test_param = TestCore.create_default_param()
        test_experiment = TestCore.create_default_experiment(parameters=test_param)
        test_core = TestCore.create_default_core()
        self.assertEqual(type(test_core.experiment), type(test_experiment))

    def test_core_submit(self):
        test_core = TestCore.create_default_core()
        test_points = [1, 2, 3, 4]
        test_core.experiment.data_points = test_points
        job_ids = test_core.submit()

        self.assertEqual(len(test_points), len(job_ids))

        self.path = test_core.experiment.system_setup.source_directory
        files = [f for f in os.listdir(self.path) if (f.endswith('.txt') and os.path.basename(f).startswith('stdout'))]
        self.assertEqual(len(job_ids), len(files))

        service_dirs = [f for f in os.listdir(self.path) if (f.startswith(".qcgpjm-service"))]
        print("directories are", service_dirs)
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
        data = test_core.collect_data(self.path)
        self.assertEqual(len(data), len(test_points))
        self.assertEqual(len(data[0]), 100)

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
        test_points = [1, 2, 3, 4]
        test_core.experiment.data_points = test_points
        test_core.submit()
        self.path = test_core.experiment.system_setup.source_directory  # path for tearDown
        data = test_core.collect_data(self.path)
        new_points = test_core.create_points_based_on_method(data)
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
