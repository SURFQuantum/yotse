import os
import yaml
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock
from ruamel.yaml import YAML
from ruamel.yaml.nodes import ScalarNode
from io import StringIO

from yotse.utils.blueprint_tools import setup_optimization_dir, update_yaml_params, replace_include_param_file, \
    create_separate_files_for_job, represent_scalar_node


class TestSetupOptimizationDir(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def set_mock_experiment(self):
        # Create a mock experiment object
        experiment = MagicMock()
        experiment.system_setup.source_directory = os.path.join(self.tmp_dir, 'src')
        if not os.path.exists(experiment.system_setup.source_directory):
            os.makedirs(experiment.system_setup.source_directory)
        experiment.system_setup.output_dir_name = 'output'
        experiment.name = 'blueprint_experiment'
        return experiment

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_multiple_step_and_job(self):
        experiment = self.set_mock_experiment()
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        dir_structure_before_call = [
            'src',
        ]
        self.assertEqual(os.listdir(self.tmp_dir), dir_structure_before_call)

        # Call the function
        setup_optimization_dir(experiment, 0, 0)

        # Check that the directory structure was created correctly
        expected_dir_structure = [
            'src',
            f'output/{experiment.name}_{timestamp_str}/step0/job0',
        ]

        for directory in expected_dir_structure:
            self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, directory)))
        sorted_dir = os.listdir(self.tmp_dir)
        sorted_dir.sort()
        self.assertEqual(sorted_dir, ['output', 'src'])
        self.assertEqual(experiment.system_setup.working_directory,
                         os.path.join(self.tmp_dir, 'output', f'{experiment.name}_{timestamp_str}/step0/job0'))

        # Call the function again
        setup_optimization_dir(experiment, 2, 1)

        # Check that the directory structure was created correctly
        expected_dir_structure_now = [
            'src',
            f'output/{experiment.name}_{timestamp_str}/step0/job0',
            f'output/{experiment.name}_{timestamp_str}/step2/job1',
        ]

        for directory in expected_dir_structure_now:
            self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, directory)))
        sorted_dir = os.listdir(self.tmp_dir)
        sorted_dir.sort()
        self.assertEqual(sorted_dir, ['output', 'src'])
        self.assertEqual(experiment.system_setup.working_directory,
                         os.path.join(self.tmp_dir, 'output', f'{experiment.name}_{timestamp_str}/step2/job1'))

    def test_existing_non_job_directory(self):
        experiment = self.set_mock_experiment()
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        experiment.system_setup.working_directory = os.path.join(self.tmp_dir, 'output',
                                                                 f'{experiment.name}_{timestamp_str}/step0/')

        # Call the function and expect it to raise a RuntimeError
        with self.assertRaises(RuntimeError):
            setup_optimization_dir(experiment, 1, 1)


class TestUpdateYamlFiles(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.tmp_dir, 'src'))

    def create_test_paramfile(self):
        # Create a YAML file with initial parameters
        self.param_file = os.path.join(self.tmp_dir, 'src', 'params.yaml')
        with open(self.param_file, 'w') as f:
            params = {
                'num_repeaters': 1,
                'det_efficiency': 1.0,
                'visibility': 1.0,
                'dark_count_prob': 0.0,
                'emission_probability': 1.0
            }
            yaml.dump(params, f, default_flow_style=False)

    def create_test_configfile(self):
        # Create a YAML config file with an INCLUDE statement
        self.config_file = os.path.join(self.tmp_dir, 'src', 'config.yaml')
        with open(self.config_file, 'w') as f:
            config = {
                'network': 'blueprint_network',
                'some_params': {
                    '&some_params': {
                        'INCLUDE': '!include nv_baseline_params.yaml'
                    },
                    '&node_type': {
                        'type': 'some_node'
                    },
                    '&repeater': {
                        '<<': '*node_type',
                        'properties': {
                            'end_node': False,
                            'num_positions': 2,
                            'port_names': [
                                'A',  # classical communication A side
                                'B',  # classical communication B side
                                'ENT_A',  # entanglement generation A side
                                'ENT_B',  # entanglement generation B side
                            ],
                            '<<': '*some_params'
                        }
                    }
                }
            }
            yaml.dump(config, f, default_flow_style=False)

    def test_update_yaml_params(self):
        self.create_test_paramfile()
        # Define the parameter updates
        param_list = [
            ['det_efficiency', 0.9],
            ['dark_count_prob', 1e-5],
            ['emission_probability', 0.1]
        ]

        # Update the YAML file
        update_yaml_params(param_list, self.param_file)

        # Load the updated parameters
        with open(self.param_file, 'r') as f:
            params = yaml.safe_load(f)

        # Check if the parameters are updated correctly
        expected_params = {
            'num_repeaters': 1,
            'det_efficiency': 0.9,
            'visibility': 1.0,
            'dark_count_prob': 1e-5,
            'emission_probability': 0.1
        }
        self.assertEqual(params, expected_params)

    @staticmethod
    def expected_config(paramfile_name):
        # Note: modifying the dict changes the order, so the expected dict is slightly reordered
        return {
            'network': 'blueprint_network',
            'some_params': {
                '&node_type': {
                    'type': 'some_node'
                },
                '&repeater': {
                    '<<': '*node_type',
                    'properties': {
                        '<<': '*some_params',
                        'end_node': False,
                        'num_positions': 2,
                        'port_names': [
                            'A',  # classical communication A side
                            'B',  # classical communication B side
                            'ENT_A',  # entanglement generation A side
                            'ENT_B',  # entanglement generation B side
                        ],
                    }
                },
                '&some_params': {
                    'INCLUDE': ScalarNode(tag='!include', value=paramfile_name, style=None)
                },
            }
        }

    def test_replace_include_param_file(self):
        yaml = YAML(typ='rt')
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.representer.add_representer(ScalarNode, represent_scalar_node)
        self.create_test_configfile()
        new_param_file_name = 'new_custom_params.yaml'

        # Call the function to replace the INCLUDE statement
        replace_include_param_file(self.config_file, new_param_file_name)

        # Load the updated YAML config file
        with open(self.config_file, 'r') as f:
            config = yaml.load(f)

        # Check if the INCLUDE statement is replaced correctly
        expected_config = self.expected_config(new_param_file_name)

        # Serialize both dictionaries to strings
        config_str = StringIO()
        yaml.dump(config, config_str)

        expected_config_str = StringIO()
        yaml.dump(expected_config, expected_config_str)

        # Compare the serialized strings
        self.assertEqual(config_str.getvalue(), expected_config_str.getvalue())

    def test_include_not_found(self):
        # Create a YAML config file WITHOUT an INCLUDE statement
        self.config_file = os.path.join(self.tmp_dir, 'config.yaml')
        with open(self.config_file, 'w') as f:
            config = {
                'network': 'blueprint_network',
                'some_params': {
                    '&node_type': {
                        'type': 'some_node'
                    }
                }
            }
            yaml.dump(config, f, default_flow_style=False)

        new_param_file_name = 'new_custom_params.yaml'

        # Call the function to replace the INCLUDE statement
        with self.assertRaises(ValueError):
            replace_include_param_file(self.config_file, new_param_file_name)

    def tearDown(self):
        # Remove the temporary directory and its contents
        shutil.rmtree(self.tmp_dir)

    def test_create_separate_files_for_job(self):
        yaml = YAML(typ='rt')
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.representer.add_representer(ScalarNode, represent_scalar_node)
        self.create_test_configfile()
        self.create_test_paramfile()

        # Create a mock experiment object
        experiment = MagicMock()
        experiment.system_setup.source_directory = os.path.join(self.tmp_dir, 'src')
        experiment.system_setup.output_dir_name = 'output'
        experiment.system_setup.cmdline_arguments = {'paramfile': 'params.yaml',
                                                     'configfile': 'config.yaml',
                                                     '--n_runs': 100}
        experiment.system_setup.program_name = 'my_program'
        experiment.name = 'blueprint_experiment'
        param1 = MagicMock()
        param1.name = 'num_repeaters'
        param1.is_active = True
        param2 = MagicMock()
        param2.name = 'visibility'
        param2.is_active = True
        experiment.parameters = [param1, param2]
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        test_datapoint_item = [3, 0.9]

        setup_optimization_dir(experiment=experiment, step_number=0, job_number=0)  # this is tested above
        new_cmdline = create_separate_files_for_job(experiment=experiment, datapoint_item=test_datapoint_item,
                                                    step_number=0, job_number=0)

        expected_program_name = os.path.join(experiment.system_setup.source_directory,
                                             experiment.system_setup.program_name)
        expected_config_path = os.path.join(self.tmp_dir, experiment.system_setup.output_dir_name,
                                            f'{experiment.name}_{timestamp_str}/step0/job0/config_step0_job0.yaml')
        expected_param_path = os.path.join(self.tmp_dir, experiment.system_setup.output_dir_name,
                                           f'{experiment.name}_{timestamp_str}/step0/job0/params_step0_job0.yaml')
        expected_cmdline = [expected_program_name, expected_config_path, "--paramfile", expected_param_path,
                            '--n_runs', str(100)]

        self.assertEqual(new_cmdline, expected_cmdline)
        # Note: modifying the dict changes the order, so the expected dict is slightly reordered
        expected_config = self.expected_config(expected_param_path)

        with open(expected_config_path, 'r') as f:
            config = yaml.load(f)

        # Serialize both dictionaries to strings
        config_str = StringIO()
        yaml.dump(config, config_str)

        expected_config_str = StringIO()
        yaml.dump(expected_config, expected_config_str)

        # Compare the serialized strings
        self.assertEqual(config_str.getvalue(), expected_config_str.getvalue())

        expected_params = {
            'num_repeaters': 3,
            'det_efficiency': 1.0,
            'visibility': 0.9,
            'dark_count_prob': 0.0,
            'emission_probability': 1.0
        }

        with open(expected_param_path, 'r') as f:
            param = yaml.load(f)
        self.assertEqual(param, expected_params)


if __name__ == '__main__':
    unittest.main()
