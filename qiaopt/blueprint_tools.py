import os
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
    # this solution avoids the if check but could produce folders with different time stamps for different
    # opt steps, which is worse!

    # output_directory = os.path.join(experiment.system_setup.source_directory.split('/src')[0], 'output')
    # timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # step_output_dir = os.path.join(output_directory, f"{experiment.name}_{timestamp_str}", f"step{step}", "output")
    # os.makedirs(step_output_dir, exist_ok=True)
    # experiment.system_setup.output_directory = step_output_dir


def change_to_optimization_dir(experiment):
    """Change the current working directory to the optimization directory"""
    os.chdir(experiment.system_setup.output_directory)


def exit_from_working_dir(experiment):
    """Change the current working directory back to the original working directory"""
    os.chdir(experiment.system_setup.source_directory)
