import os
import yaml
import shutil
from datetime import datetime


def setup_optimization_dir(experiment, step):
    """Create the directory for this optimization step"""
    output_directory = os.path.join(experiment.system_setup.source_directory.split('/src')[0], 'output')
    if step == 0:
        # for first step create timestamped project directory
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        name_project_dir = output_directory + '/' + experiment.name + "_" + timestamp_str + "/"
        step_output_dir = name_project_dir + f'/step{step}/output'
        experiment.system_setup.output_directory = step_output_dir
    else:
        step_output_dir = experiment.system_setup.output_directory.split(f'/step{step-1}')[0] + f'/step{step}/output'

    os.makedirs(step_output_dir, exist_ok=True)
    # todo this should be extended like set_basic_directory_structure_for_job function


def change_to_optimization_dir(experiment):
    """Change the current working directory to the optimization directory"""
    os.chdir(experiment.system_setup.output_directory)


def exit_from_working_dir(experiment):
    """Change the current working directory back to the original working directory"""
    os.chdir(experiment.system_setup.source_directory)


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
        config = yaml.safe_load(f)

    # Find the line with the INCLUDE keyword
    found_include = False
    for key, value in config.items():
        if key == "INCLUDE":
            found_include = True
            # Replace the included file name with the new parameter file name
            config[key] = f"!include {paramfile_name}"
            break

    # Check if the INCLUDE keyword was found
    if not found_include:
        raise ValueError(f"INCLUDE statement not found in '{configfile_name}'")

    # Save the updated YAML config file
    with open(configfile_name, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_separate_files_for_job(experiment, datapoint_item):
    """Create a separate parameter and matching config file for this job and get ready for execution."""
    # this should execute after the directory for the specific job is set up
    # 1 - copy the original param and config file to the created dir
    source_directory = '/path/to/source/directory'
    working_directory = '/path/to/working/directory'
    job_name = 'job0'

    # Copy paramfile_name to working_directory
    paramfile_name = 'base_params.yaml'
    paramfile_base_name, paramfile_ext = os.path.splitext(paramfile_name)
    paramfile_new_name = paramfile_base_name + '_' + job_name + paramfile_ext
    paramfile_source_path = os.path.join(source_directory, paramfile_name)
    paramfile_dest_path = os.path.join(working_directory, job_name, paramfile_new_name)
    shutil.copy(paramfile_source_path, paramfile_dest_path)

    # Copy configfile_name to working_directory
    configfile_name = 'config_file.yaml'
    configfile_base_name, configfile_ext = os.path.splitext(configfile_name)
    configfile_new_name = configfile_base_name + '_' + job_name + configfile_ext
    configfile_source_path = os.path.join(source_directory, configfile_name)
    configfile_dest_path = os.path.join(working_directory, job_name, configfile_new_name)
    shutil.copy(configfile_source_path, configfile_dest_path)

    paramfile_name_thisjob = paramfile_dest_path
    configfile_name_thisjob = configfile_dest_path
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
    update_yaml_params(param_list=param_list, paramfile_name=paramfile_name_thisjob)

    # 3 - overwrite the name of the paramfile inside the configfile with the new paramfile name
    replace_include_param_file(configfile_name=configfile_name_thisjob, paramfile_name=paramfile_name_thisjob)
    # 4 - overwrite the old_cmdline such that it no longer contains the varied params, but instead the correct paths to
    # the new param and config files
    cmdline = [os.path.join(experiment.system_setup.source_directory, experiment.system_setup.program_name)]
    # todo : make sure we have the correct path to the file
    cmdline.append(configfile_name_thisjob)
    cmdline.append("--paramfile")
    cmdline.append(paramfile_name_thisjob)
    # add fixed cmdline arguments
    for key, value in experiment.system_setup.cmdline_arguments.items():
        cmdline.append(key)
        cmdline.append(str(value))

    return cmdline
