"""blueprint_tools.py.

This module provides helper functions for the NL blueprint experiment setup within the
QIA project.
"""
import os
import shutil
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import yaml  # type: ignore[import-untyped]
from ruamel.yaml import YAML
from ruamel.yaml.nodes import ScalarNode

from yotse.pre import Experiment


def setup_optimization_dir(
    experiment: Experiment, step_number: int, job_number: int
) -> None:
    """Create the directory structure for an optimization step.

    Parameters
    ----------
    experiment : Experiment
        The Experiment object for which the directory structure should be set up.
    step_number : int
        The number of the current optimization step.
    job_number : int
        The number of the job within the optimization step.

    Notes
    -----
    This function creates the directory structure for an optimization step within an experiment. The structure
    includes a `src` directory containing several files related to the optimization, and an `output` directory
    containing directories for each step and job. The function does not return anything but modifies the file system
    to create the necessary directories.

    The directory structure for the optimization step is as follows (for m optimization steps and n jobs)::

        src/
        ├── unified_script.py
        ├── processing_function.py
        ├── config.yaml
        ├── baseline_params.yaml
        ├── qiapt_runscript.py
        output/
        ├── experiment_name_timestamp_str/
        │   ├── step0/
        │   │   ├── job0/
        │   │   │   ├── stdout0.txt
        │   │   │   ├── dataframe_holder.pickle
        │   │   │   ├── baseline_params_job0.yaml
        │   │   │   └── config_job0.yaml
        │   │   ...
        │   │   └── jobn/
        │   ...
        │   └── stepm/
    """

    output_directory = os.path.join(
        experiment.system_setup.source_directory,
        "..",
        experiment.system_setup.output_dir_name,
    )
    output_directory = os.path.realpath(output_directory)  # clean path of '..'
    if step_number == 0 and job_number == 0:
        # for first step create timestamped project directory
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        name_project_dir = os.path.join(
            output_directory, experiment.name + "_" + timestamp_str
        )
        new_working_dir = os.path.join(
            name_project_dir, f"step{step_number}", f"job{job_number}"
        )
    else:
        if not os.path.basename(
            os.path.normpath(experiment.system_setup.working_directory)
        ).startswith("job"):
            raise RuntimeError(
                "The current working directory does not start with 'job'. "
                "New working directory can't be set up properly."
            )
        new_working_dir = os.path.join(
            experiment.system_setup.working_directory,
            "../..",
            f"step{step_number}",
            f"job{job_number}",
        )
        new_working_dir = os.path.realpath(new_working_dir)  # clean path of '..'

    if not os.path.exists(new_working_dir):
        os.makedirs(new_working_dir)
    experiment.system_setup.working_directory = new_working_dir


def update_yaml_params(param_list: List[Tuple[str, Any]], paramfile_name: str) -> None:
    """Update parameter values in a YAML file and save the updated file.

    Parameters
    ----------
    param_list : List[Tuple[str, Any]]
        A list of tuples containing parameter names and their updated values.
    paramfile_name : str
        The name of the YAML file containing the parameters to update.
    """

    # Load the YAML file
    with open(paramfile_name, "r") as f:
        params = yaml.safe_load(f)

    # Update each parameter value
    for param_name, param_value in param_list:
        if param_name in params:
            params[param_name] = param_value
        else:
            raise ValueError(f"Parameter name '{param_name}' not found in YAML file")

    # Save the updated YAML file
    with open(paramfile_name, "w") as f:
        yaml.dump(params, f, default_flow_style=False)


def represent_scalar_node(dumper: yaml.Dumper, data: yaml.ScalarNode) -> ScalarNode:
    """Represent a ScalarNode object as a scalar value in a YAML file.

    Parameters
    ----------
    dumper : yaml.Dumper
        The YAML dumper object being used to write the file.
    data : yaml.ScalarNode
        The ScalarNode object being represented.

    Returns
    -------
    scalar : str
        The scalar value of the ScalarNode object.
    """
    return dumper.represent_scalar(data.tag, data.value)


def replace_include_param_file(configfile_name: str, paramfile_name: str) -> None:
    """Replace the INCLUDE keyword in a YAML config file with a reference to a parameter
    file.

    Parameters
    ----------
    configfile_name : str
        The name of the YAML configuration file to modify.
    paramfile_name : str
        The name of the parameter file to include in the configuration file.

    Notes
    -----
    This function replaces an INCLUDE keyword in a YAML configuration file with a reference to a parameter file.
    It loads the YAML config file, searches recursively for an INCLUDE keyword, and replaces it with a reference
    to the specified parameter file. If the INCLUDE keyword is not found, an error is raised.
    """
    yaml = YAML(typ="rt")
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.representer.add_representer(ScalarNode, represent_scalar_node)

    # Load the YAML config file
    with open(configfile_name, "r") as f:
        old_config = yaml.load(f)

    # Find the line with the INCLUDE keyword recursively
    def replace_include(
        config: Union[List[Any], Dict[str, Any]], replace_str: str, found: bool = False
    ) -> bool:
        """Recursively search a dictionary or list for an INCLUDE keyword and replace it
        with a reference to a parameter file.

        Parameters
        ----------
        config : dict or list
            The dictionary or list to search for an INCLUDE keyword.
        replace_str : str
            The name of the parameter file to include in place of the INCLUDE keyword.
        found : bool, optional
            A boolean flag indicating whether an INCLUDE keyword has already been found and replaced. Defaults to False.

        Returns
        -------
        found : bool
            True if an INCLUDE keyword was found and replaced in the dictionary or list, False otherwise.
        """
        if isinstance(config, dict):
            for key, value in list(config.items()):
                if key != "INCLUDE":
                    found = replace_include(value, replace_str, found)
                elif key == "INCLUDE" and not found:
                    config[key] = ScalarNode(
                        tag="!include", value=replace_str, style=None
                    )
                    found = True
                elif key == "INCLUDE" and found:
                    del config[key]
        elif isinstance(config, list):
            for i in range(len(config)):
                found = replace_include(config[i], replace_str, found)

        return found

    found_include = replace_include(old_config, paramfile_name)

    # Check if the INCLUDE keyword was found
    if not found_include:
        raise ValueError(f"INCLUDE statement not found in '{configfile_name}'")

    # Save the updated YAML config file
    with open(configfile_name, "w") as f:
        yaml.dump(old_config, f)


def create_separate_files_for_job(
    experiment: Experiment,
    datapoint_item: List[float],
    step_number: int,
    job_number: int,
) -> List[Any]:
    """Create separate parameter and configuration files for a job and prepare for
    execution.

    Parameters
    ----------
    experiment : Experiment
        The experiment object containing information about the experiment.
    datapoint_item : List[float]
        A single item of data points for the job, represented as a list.
    step_number : int
        The number of the step in the experiment.
    job_number : int
        The number of the job within the step.

    Returns
    -------
    job_cmdline : list
        The command line arguments for running the job.

    Notes
    -----
    This function creates separate parameter and configuration files for a job based on the provided experiment,
    datapoint item, step number, and job number. It prepares the job for execution by setting up the necessary files
    and returning the command line arguments for running the job. The function returns the command line arguments
    as a list for use with QCG-Pilotjob.

    The created files will be saved in the experiment's directory, under a subdirectory for the step and job.
    The parameter file will have a name like "params_stepY_jobX.yaml" and the configuration file will have a name like
    "config_stepY_jobX.yaml", where "X" is the job number and "Y" the step number.
    """
    # this should execute after the directory for the specific job is set up by setup_optimization_dir
    # 1 - copy the original param and config file to the created dir
    source_directory = experiment.system_setup.source_directory
    working_directory = experiment.system_setup.working_directory
    old_cmdline_args = experiment.system_setup.cmdline_arguments.copy()
    paramfile_name = os.path.basename(old_cmdline_args["paramfile"])
    configfile_name = os.path.basename(old_cmdline_args["configfile"])
    # delete unnecessary args from dict copy
    del old_cmdline_args["paramfile"]
    del old_cmdline_args["configfile"]
    job_name = f"job{job_number}"
    step_name = f"step{step_number}"

    # Copy paramfile_name to working_directory
    paramfile_base_name, paramfile_ext = os.path.splitext(paramfile_name)
    paramfile_new_name = (
        paramfile_base_name + "_" + step_name + "_" + job_name + paramfile_ext
    )
    paramfile_source_path = os.path.join(source_directory, paramfile_name)
    paramfile_dest_path = os.path.join(working_directory, paramfile_new_name)
    shutil.copy(paramfile_source_path, paramfile_dest_path)

    # Copy configfile_name to working_directory
    configfile_base_name, configfile_ext = os.path.splitext(configfile_name)
    configfile_new_name = (
        configfile_base_name + "_" + step_name + "_" + job_name + configfile_ext
    )
    configfile_source_path = os.path.join(source_directory, configfile_name)
    configfile_dest_path = os.path.join(working_directory, configfile_new_name)
    shutil.copy(configfile_source_path, configfile_dest_path)

    # 2 - take the data_point for the current step + the active parameters/cmdlineargs and then overwrite those
    # in the respective param file
    param_list = []
    for p, param in enumerate(experiment.parameters):
        if param.is_active:
            param_list.append((param.name, datapoint_item[p]))
    if len(param_list) != len(datapoint_item):
        raise RuntimeError(
            "Datapoint has different length then list of parameters to be changes in paramfile."
        )
    update_yaml_params(param_list=param_list, paramfile_name=paramfile_dest_path)

    # 3 - overwrite the name of the paramfile inside the configfile with the new paramfile name
    replace_include_param_file(
        configfile_name=configfile_dest_path, paramfile_name=paramfile_dest_path
    )
    # 4 - construct new cmdline such that it no longer contains the varied params, but instead the correct paths to
    # the new param and config files
    cmdline = [
        os.path.join(
            experiment.system_setup.source_directory,
            experiment.system_setup.program_name,
        )
    ]
    cmdline.append(configfile_dest_path)
    cmdline.append("--paramfile")
    cmdline.append(paramfile_dest_path)
    # add fixed cmdline arguments
    for key, value in old_cmdline_args.items():
        cmdline.append(key)
        cmdline.append(str(value))

    return cmdline
