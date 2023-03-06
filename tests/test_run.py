import os
import glob
import shutil
import unittest
import json
import pickle
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
    def create_default_experiment(parameters=None, opt_info_list=None):
        return Experiment(experiment_name='default_exp',
                          system_setup=SystemSetup(
        
                              working_directory=os.getcwd(),
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
        test_param = TestCore.create_default_param()
        test_experiment = TestCore.create_default_experiment(parameters=test_param)
        test_core = TestCore.create_default_core()
        test_points = [1,2,3,4]
        test_core.experiment.data_points = test_points
        job_ids = test_core.submit()

        self.assertEqual(len(test_points), len(job_ids))  
        
        path = test_core.experiment.system_setup.working_directory
        extension = test_core.experiment.system_setup.output_extension
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        self.assertEqual(len(job_ids), len(files)) 
        for f in os.listdir(path):
            if f.endswith('.csv'):
                os.remove(path + "/" + f)
        
        dirs = [f for f in os.listdir(path) if (f.startswith(".qcg"))]

        #self.assertEqual(len(dirs), 2)
        with open( path + "/" + dirs[1] + "/" + "final_status.json", "r") as f:
            data = json.load(f)
        jobs_finished = data['JobStats']['FinishedJobs']
        jobs_failed = data['JobStats']['FailedJobs']
        self.assertEqual(jobs_finished, len(test_points))
        self.assertEqual(jobs_failed, 0)
       
        for d in dirs:
            shutil.rmtree(path + "/" + d)
   
   
    def test_core_collectdata(self):
        test_param = TestCore.create_default_param()
        test_experiment = TestCore.create_default_experiment(parameters=test_param)
        test_core = TestCore.create_default_core()
        test_points = [1,2,3,4]
        test_core.experiment.data_points = test_points
        job_ids = test_core.submit()

        path = test_core.experiment.system_setup.directory
        data = test_core.collectdata(path)
        self.assertEqual(len(data), len(test_points))
        self.assertEqual(len(data[0]), 100)

        dirs = [f for f in os.listdir(path) if (f.startswith(".qcg"))]
        for d in dirs:
            shutil.rmtree(path + "/" + d)

    # def test_core_create_points_based_on_method(self):
    #     test_param = TestCore.create_default_param()
    #     test_experiment = TestCore.create_default_experiment(parameters=test_param)
    #     test_core = TestCore.create_default_core()

    
    # def test_core_run(self):
    #     test_param = TestCore.create_default_param()
    #     test_experiment = TestCore.create_default_experiment(parameters=test_param)
    #     test_core = TestCore.create_default_core()
    #     test_points = [1,2,3,4]
    #     test_core.experiment.data_points = test_points
    #     test_core.run()
        
if __name__ == '__main__':
        unittest.main()
