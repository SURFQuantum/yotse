import os
from datetime import datetime
import unittest
from qiaopt.pre import Experiment, SystemSetup
from qiaopt.blueprint_tools import setup_optimization_dir, change_to_optimization_dir, exit_from_working_dir

if os.getcwd()[-5:] == "tests":
    DUMMY_FILE = "myfunction.py"
else:
    DUMMY_FILE = "tests/myfunction.py"


class TestOptimizationDirs(unittest.TestCase):
    def setUp(self):
        if not os.path.exists('src'):
            os.mkdir('src')
        self.experiment_name = 'test_experiment'
        self.system_setup = SystemSetup(source_directory=os.getcwd() + '/src', program_name=DUMMY_FILE)
        self.parameters = []
        self.opt_info_list = []
        self.experiment = Experiment(self.experiment_name, self.system_setup, self.parameters, self.opt_info_list)
        self.cwd = os.getcwd()
        self.date_time = datetime.now()

    def tearDown(self):
        os.chdir(self.cwd)
        os.rmdir('src')

    def return_output_dir(self, step):
        return os.path.join(os.getcwd(), 'output',
                            f'{self.experiment_name}_{self.date_time.strftime("%Y-%m-%d_%H:%M:%S")}',
                            f'step{step}', 'output')

    def test_setup_optimization_dir(self):
        # first step
        output_dir0 = self.return_output_dir(step=0)
        self.assertFalse(os.path.exists(output_dir0))
        setup_optimization_dir(self.experiment, step=0)
        self.assertTrue(os.path.exists(output_dir0))
        print("Output directory was:", output_dir0)
        # subsequent step
        output_dir1 = self.return_output_dir(step=1)
        self.assertFalse(os.path.exists(output_dir1))
        setup_optimization_dir(self.experiment, step=1)
        self.assertTrue(os.path.exists(output_dir1))
        print("Output directory was:", output_dir1)
        os.removedirs(output_dir0)
        os.removedirs(output_dir1)

    def test_change_to_optimization_dir(self):
        output_dir5 = self.return_output_dir(step=5)
        os.makedirs(output_dir5, exist_ok=True)
        self.experiment.system_setup.output_directory = output_dir5
        change_to_optimization_dir(self.experiment)
        self.assertEqual(os.getcwd(), self.experiment.system_setup.output_directory)
        print('Output directory was:', output_dir5)
        os.removedirs(output_dir5)

    def test_exit_from_working_dir(self):
        output_dir42 = self.return_output_dir(step=42)
        os.makedirs(output_dir42, exist_ok=True)
        self.experiment.system_setup.output_directory = output_dir42
        change_to_optimization_dir(self.experiment)
        exit_from_working_dir(self.experiment)
        self.assertEqual(os.getcwd(), self.experiment.system_setup.source_directory)
        os.removedirs(output_dir42)
        print("ending in directory:", os.getcwd())


if __name__ == '__main__':
    unittest.main()
