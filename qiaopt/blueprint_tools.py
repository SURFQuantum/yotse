import os
import yaml
import shutil
from datetime import datetime

from qiaopt.pre import Experiment


def setup_optimization_dir(experiment, step_number, job_number):
    """Create the directory for this optimization step.
    The structure will be as follows (for m optimization steps and n jobs)
    > src
        - unified_script.py
        - processing_function.py
        - config.yaml
        - baseline_params.yaml
        - qiapt_runscript.py
    > output
        > experiment_name_timestamp_str
            > step0
                > job0
                    - stdout0.txt
                    - dataframe_holder.pickle (?)
                    - baseline_params_job0.yaml
                    - config_job0.yaml
                ...
                > jobn
            ...
            > stepm
    """
    output_directory = os.path.join(experiment.system_setup.source_directory, '..',
                                    experiment.system_setup.output_directory)
    output_directory = os.path.realpath(output_directory)                                       # clean path of '..'
    if step_number == 0 and job_number == 0:
        # for first step create timestamped project directory
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        name_project_dir = os.path.join(output_directory, experiment.name + "_" + timestamp_str)
        new_working_dir = os.path.join(name_project_dir, f'step{step_number}', f'job{job_number}')
    else:
        if not os.path.basename(os.path.normpath(experiment.system_setup.working_directory)).startswith("job"):
            raise RuntimeError("The current working directory does not start with 'job'. "
                               "New working directory can't be set up properly.")
        new_working_dir = os.path.join(experiment.system_setup.working_directory, '..', '..',
                                       f'step{step_number}', f'job{job_number}')
        new_working_dir = os.path.realpath(new_working_dir)                                     # clean path of '..'

    if not os.path.exists(new_working_dir):
        os.makedirs(new_working_dir)
    experiment.system_setup.working_directory = new_working_dir


def update_yaml_params(param_list, paramfile_name):
    # Load the YAML file
    with open(paramfile_name, 'r') as f:
        params = yaml.safe_load(f)

    # Update each parameter value
    for param_name, param_value in param_list:
        if param_name in params:
            params[param_name] = param_value
        else:
            raise ValueError(f"Parameter name '{param_name}' not found in YAML file")

    # Save the updated YAML file
    with open(paramfile_name, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)


def replace_include_param_file(configfile_name, paramfile_name):
    # Load the YAML config file
    with open(configfile_name, 'r') as f:
        old_config = yaml.safe_load(f)

    # Find the line with the INCLUDE keyword recursively
    def replace_include(config, replace_str):
        found = False
        if isinstance(config, dict):
            for key, value in config.items():
                if key == 'INCLUDE':
                    config[key] = replace_str
                    found = True
                elif isinstance(value, dict):
                    found_in_nested, config[key] = replace_include(value, replace_str)
                    found = found or found_in_nested
                elif isinstance(value, list):
                    for i in range(len(value)):
                        found_in_nested, config[key][i] = replace_include(value[i], replace_str)
                        found = found or found_in_nested
        return found, config

    found_include, new_config = replace_include(config=old_config, replace_str=f"!include {paramfile_name}")

    # Check if the INCLUDE keyword was found
    if not found_include:
        raise ValueError(f"INCLUDE statement not found in '{configfile_name}'")

    # Save the updated YAML config file
    with open(configfile_name, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)


def create_separate_files_for_job(experiment: Experiment, datapoint_item: list, step_number: int, job_number: int):
    """Create a separate parameter and matching config file for this job and get ready for execution."""
    # this should execute after the directory for the specific job is set up by setup_optimization_dir
    # 1 - copy the original param and config file to the created dir
    source_directory = experiment.system_setup.source_directory
    working_directory = experiment.system_setup.working_directory
    old_cmdline_args = experiment.system_setup.cmdline_arguments
    paramfile_name = old_cmdline_args['paramfile']
    del old_cmdline_args['paramfile']
    configfile_name = old_cmdline_args['configfile']
    del old_cmdline_args['configfile']
    job_name = f'job{job_number}'
    step_name = f'step{step_number}'

    # Copy paramfile_name to working_directory
    paramfile_base_name, paramfile_ext = os.path.splitext(paramfile_name)
    paramfile_new_name = paramfile_base_name + '_' + step_name + '_' + job_name + paramfile_ext
    paramfile_source_path = os.path.join(source_directory, paramfile_name)
    paramfile_dest_path = os.path.join(working_directory, paramfile_new_name)
    shutil.copy(paramfile_source_path, paramfile_dest_path)

    # Copy configfile_name to working_directory
    configfile_base_name, configfile_ext = os.path.splitext(configfile_name)
    configfile_new_name = configfile_base_name + '_' + step_name + '_' + job_name + configfile_ext
    configfile_source_path = os.path.join(source_directory, configfile_name)
    configfile_dest_path = os.path.join(working_directory, configfile_new_name)
    shutil.copy(configfile_source_path, configfile_dest_path)

    # 2 - take the data_point for the current step + the active parameters/cmdlineargs and then overwrite those
    # in the respective param file
    param_list = []
    for p, param in enumerate(experiment.parameters):
        if param.is_active:
            if len(experiment.parameters) == 1:
                # single parameter
                param_list.append((param.name, datapoint_item))
            else:
                param_list.append((param.name, datapoint_item[p]))
    if len(param_list) != len(datapoint_item):
        raise RuntimeError("Datapoint has different length then list of parameters to be changes in paramfile.")
    update_yaml_params(param_list=param_list, paramfile_name=paramfile_dest_path)

    # 3 - overwrite the name of the paramfile inside the configfile with the new paramfile name
    replace_include_param_file(configfile_name=configfile_dest_path, paramfile_name=paramfile_dest_path)
    # 4 - construct new cmdline such that it no longer contains the varied params, but instead the correct paths to
    # the new param and config files
    cmdline = [os.path.join(experiment.system_setup.source_directory, experiment.system_setup.program_name)]
    cmdline.append(configfile_dest_path)
    cmdline.append("--paramfile")
    cmdline.append(paramfile_dest_path)
    # add fixed cmdline arguments
    for key, value in old_cmdline_args.items():
        cmdline.append(key)
        cmdline.append(str(value))

    return cmdline
