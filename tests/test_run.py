import os
import unittest
import numpy as np
from qiaopt.run import Core, Executor
from qiaopt.pre import Experiment, SystemSetup, Parameter


DUMMY_FILE = "myfunction.py"


class TestCore(unittest.TestCase):
    """Test the Core class."""
    @staticmethod
    def create_default_param(name="bright_state_parameter", parameter_range=[.1, .9], number_points=9,
                             distribution="linear", constraints=[None], custom_distribution=None):
        return Parameter(name=name, parameter_range=parameter_range, number_points=number_points,
                         distribution=distribution, constraints=constraints, custom_distribution=custom_distribution)

    @staticmethod
    def create_default_experiment(parameters=None, optimization_steps=None):
        return Experiment(experiment_name='default_exp',
                          system_setup=SystemSetup(
        
                              working_directory=os.getcwd(),
                              program_name=DUMMY_FILE,
                              command_line_arguments={'arg1': 1.0}),
                          parameters=parameters,
                          optimization_steps=optimization_steps)
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
        test_param = TestCore.create_default_param()
        test_experiment = TestCore.create_default_experiment(parameters=test_param)
        test_core = TestCore.create_default_core()
        test_points = [1,2,3,4]
        test_core.experiment.data_points = test_points
        job_ids = test_core.submit()
        self.assertEqual(len(test_points), len(job_ids))
    #    TODO: checko output files are created?
    #    TODO: check .qcg dir is been created
    #    TODO: check status success


    def test_core_collectdata(self):
        test_param = TestCore.create_default_param()
        test_experiment = TestCore.create_default_experiment(parameters=test_param)
        test_core = TestCore.create_default_core()
        test_points = [1,2,3,4]
        #test_core.experiment.data_points = test_points
        data = test_core.collectdata()
        #print(data)
        self.assertEqual(len(data), len(test_points))
        self.assertEqual(len(data[0]), 100)

    def test_core_create_points_based_on_method(self):
        test_param = TestCore.create_default_param()
        test_experiment = TestCore.create_default_experiment(parameters=test_param)
        test_core = TestCore.create_default_core()

    
    def test_core_run(self):
        test_param = TestCore.create_default_param()
        test_experiment = TestCore.create_default_experiment(parameters=test_param)
        test_core = TestCore.create_default_core()
        test_points = [1,2,3,4]
        test_core.experiment.data_points = test_points
        test_core.run()
        
if __name__ == '__main__':
        unittest.main()