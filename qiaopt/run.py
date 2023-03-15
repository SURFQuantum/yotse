""" Defines the run"""
import os
import pandas as pd
from qiaopt.pre import Experiment
from qiaopt.optimization import Optimizer, GAOpt
from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager


def create_points(parameters):
    """
    Creates a list of points based on the given parameters.

    Parameters:
    -----------
    parameters: list
        List of parameters used to create new points.

    Returns:
    --------
    list
        List of new points created based on the given parameters.
    """
    return [3, 4, 5]


def qcgpilot_commandline(experiment, datapoint_item):
    """
     Creates a command line for the QCG-PilotJob executor based on the experiment configuration.

     Parameters:
     -----------
     experiment: Experiment
         The experiment to configure the command line for.

     Returns:
     --------
     list
         A list of strings representing the command line arguments for the QCG-PilotJob executor.
     """
    cmdline = [os.path.join(experiment.system_setup.source_directory, experiment.system_setup.program_name)]
    # add parameters
    for p, param in enumerate(experiment.parameters):
        if param.is_active:
            cmdline.append(f"--{param.name}")
            if len(experiment.parameters) == 1:
                # single parameter
                cmdline.append(datapoint_item)
            else:
                cmdline.append(datapoint_item[p])
    # add fixed cmdline arguments
    for key, value in experiment.system_setup.cmdline_arguments.items():
        cmdline.append(key)
        cmdline.append(str(value))
    return cmdline


def get_files_by_extension(directory, extension):
    """
    Returns a list of files in the given directory with the specified extension.

    Parameters:
    -----------
    directory: str
        The directory to search for files in.
    extension: str
        The file extension to search for.

    Returns:
    --------
    list
        A list of files (and their actual location) in the given directory with the specified extension.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]


def files_to_list(files):
    """
    Reads CSV files and returns their contents as a list of dataframes.

    Parameters:
    -----------
    files: list
        A list of CSV files to read.

    Returns:
    --------
    list
        A list of pandas dataframes, each representing the contents of a CSV file.
    """
    # todo check if extension is csv?
    dfs = [pd.read_csv(file, delimiter=' ') for file in files]
    return pd.concat(dfs, ignore_index=True)


def set_basic_directory_structure_for_job(experiment: Experiment, step_number: int, job_number: int) -> None:
    """
    Creates a new directory for the given step number and updates the experiment's working directory accordingly.

    The basic directory structure is as follows
    source_dir
        - output_dir
            - step_{i}
                analysis_script.py
                analysis_output.csv
                - job_{j}
                    output_of_your_script.extension
                    stdout{j}.txt

    Parameters
    ----------
    experiment : Experiment
        The :obj:Experiment that is being run.
    step_number : int
        The number of the current step.
    job_number : int
        The number of the current job.
    """
    source_dir = experiment.system_setup.source_directory
    output_dir = experiment.system_setup.output_directory
    new_working_dir = os.path.join(source_dir, output_dir, f'step{step_number}', f'job{job_number}')

    if not os.path.exists(new_working_dir):
        os.makedirs(new_working_dir)
    experiment.system_setup.working_directory = new_working_dir


# def copy_job_output_from_job_dir_to_step_dir(experiment, extension):
#     step_dir = experiment.system_setup.current_step_directory
#     for i, item in enumerate(experiment.data_points):
#         pass


# TODO: old and maybe no longer needed
# def move_output_to_output_directory(self):
#     output_dir = self.experiment.system_setup.output_directory
#     files_list = get_files_by_extension(directory=output_dir, extension=self.experiment.system_setup.output_extension)
#     for file in files_list:
#         shutil.move(file, output_dir)
#
#
# def move_stdout_to_log_dir(self):
#     def is_stdout_file(file, basename):
#         return os.path.basename(file).endswith('txt') and os.path.basename(file).startswith(basename)
#     file_list = [file for file in os.listdir(self.experiment.system_setup.source_directory)
#                  if is_stdout_file(file, self.experiment.system_setup.stdout_basename)]
#     for file in file_list:
#         shutil.move(file, log_dir)


class Core:
    """
    Defines the default run function for the executor.

    Parameters:
    -----------
    experiment: Experiment
        The experiment to run.
    """

    def __init__(self, experiment):
        self.experiment = experiment
        if self.experiment.optimization_information_list:
            if len([opt for opt in self.experiment.optimization_information_list if opt.is_active]) > 1:
                raise RuntimeError('Multiple active optimization steps. Please set all but one to active=False')
            opt_info = self.experiment.optimization_information_list[0]
            self.refinement_x = opt_info.parameters['refinement_x']
            self.refinement_y = opt_info.parameters['refinement_y']
            self.num_points = opt_info.parameters['num_points']
            if opt_info.name == 'GA':
                num_generation = opt_info.parameters['num_generations']
                self.optimization_alg = GAOpt(function=self.experiment.cost_function,
                                              data=None,
                                              num_generations=opt_info.parameters['num_generations'],
                                              logging_level=opt_info.parameters['logging_level'])
            else:
                raise ValueError('Unknown optimization algorithm.')

            self.optimizer = Optimizer(self.optimization_alg)
        else:
            self.optimization_alg = None
            self.optimizer = None

    def run(self, step=0):
        """ Submits jobs to the LocalManager, collects the output, creates new data points, and finishes the run."""
        print(f"Starting default run of {self.experiment.name} (step{step}): submit, collect, create")
        self.submit(step_number=step)
        data = self.collect_data()
        self.create_points_based_on_method(data)
        print("Finished run")

    def submit(self, step_number=0):
        """
        Submits jobs to the LocalManager.

        Returns:
        --------
        list
            A list of job IDs submitted to the LocalManager.
        """
        manager = LocalManager()
        stdout = self.experiment.system_setup.stdout_basename

        jobs = Jobs()
        if not self.experiment.data_points:
            raise RuntimeError(f"Can not submit jobs for Experiment {self.experiment.name}: No datapoints available.")

        for i, item in enumerate(self.experiment.data_points):
            set_basic_directory_structure_for_job(experiment=self.experiment, step_number=step_number, job_number=i)
            jobs.add(
                name=self.experiment.name + str(i),
                exec=self.experiment.system_setup.executor,
                args=qcgpilot_commandline(self.experiment, datapoint_item=item),
                stdout=stdout + str(i) + ".txt",
                wd=self.experiment.system_setup.working_directory,
            )
        if self.experiment.system_setup.analysis_script is not None:
            # add analysis job with correct dependency
            jobs.add(
                name=self.experiment.name + f"step{step_number}_analysis",
                exec=self.experiment.system_setup.executor,
                args=[os.path.join(self.experiment.system_setup.source_directory,
                                   self.experiment.system_setup.analysis_script)],
                stdout=stdout + f"step{step_number}_analysis.txt",
                wd=self.experiment.system_setup.current_step_directory,
                after=jobs.job_names()
            )
        job_ids = manager.submit(jobs)
        manager.wait4(job_ids)
        manager.finish()
        manager.cleanup()
        return job_ids

    def collect_data(self):
        """
        Collects data from output files of the current step of the experiment.


        Returns
        -------
        list
            A list of pandas dataframes, each containing the data from an output file.

        """
        if self.experiment.system_setup.analysis_script is None:
            # no analysis script: extract data from output files in job dirs and combine to single dataframe
            output_directory_current_step = os.path.join(self.experiment.system_setup.working_directory, '..')
            extension = self.experiment.system_setup.output_extension
            files = []
            for job_dir in [x[0] for x in os.walk(output_directory_current_step)
                            if x[0] != output_directory_current_step]:
                files.extend(get_files_by_extension(job_dir, extension))
            data = files_to_list(files)
        else:
            # analysis script is given and will output file 'output.csv' with format 'cost_fun param0 param1 ...'
            data = pd.read_csv(os.path.join(self.experiment.system_setup.current_step_directory, 'output.csv'),
                               delimiter=' ')
        return data

    def create_points_based_on_method(self, data):
        """
        Applies a method to process the collected data and create new data points.

        Parameters
        ----------
        data : list
            A list of pandas dataframes containing the collected data.

        Returns
        -------
        list
            A list of new data points created based on the input data.

        """

        if self.optimization_alg is not None:
            self.optimization_alg.data = data
            solution, func_values = self.optimizer.optimize()

            xy_new, func_new = self.optimizer.construct_points(solution,
                                                               num_points=self.num_points,
                                                               refinement_x=self.refinement_x,
                                                               refinement_y=self.refinement_y)
            return xy_new
        else:
            return


class Executor(Core):
    def __init__(self, experiment):
        super().__init__(experiment)

    def run(self, step=0):
        super().run(step)
