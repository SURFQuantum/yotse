""" Defines the run"""
import os
# import glob
import pandas as pd
from qcg.pilotjob.api.manager import LocalManager
from qcg.pilotjob.api.job import Jobs


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


def qcgpilot_commandline(experiment):
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
    cmdline = [experiment.system_setup.program_name]
    for key, value in experiment.system_setup.cmdline_arguments.items():
        cmdline.append(key)
        cmdline.append(str(value))
    return cmdline


# def getfiles2(directory, extension):
#     all_filenames = [i for i in glob.glob('{}.{}'.format(directory, extension))]


def getfiles(directory, extension):
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
        A list of files in the given directory with the specified extension.
    """
    dir_path = directory
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(extension):
            files.append(file)
    return files


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
    data = []
    for file in files:
        # if extension == "csv":
        filedata = pd.read_csv(file)
        data.append(filedata)
    return data


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

    def run(self):
        """ Submits jobs to the LocalManager, collects the output, creates new data points, and finishes the run."""
        print("Starting default run: submit, collect, create")
        self.submit()
        directory = self.experiment.system_setup.working_directory
        data = self.collect_data(directory)
        self.create_points_based_on_method(data)
        print("Finished run")

    def submit(self):
        """
        Submits jobs to the LocalManager.

        Returns:
        --------
        list
            A list of job IDs submitted to the LocalManager.
        """
        manager = LocalManager()
        extension = self.experiment.system_setup.output_extension
        directory = self.experiment.system_setup.working_directory
        stdout = self.experiment.system_setup.stdout

        jobs = Jobs()
        for i, item in enumerate(self.experiment.data_points):
            jobs.add(
                name=self.experiment.name + str(i),
                exec=self.experiment.system_setup.executor,
                args=qcgpilot_commandline(self.experiment),
                stdout=stdout + str(i) + "." + extension,
                wd=directory,
            )
        job_ids = manager.submit(jobs)
        manager.wait4(job_ids)
        manager.finish()
        manager.cleanup()
        return job_ids

    def collect_data(self, directory):
        """
        Collects data from output files in the given directory.

        Parameters
        ----------
        directory : str
            Path to the directory containing the output files.

        Returns
        -------
        list
            A list of pandas dataframes, each containing the data from an output file.

        """
        directory = self.experiment.system_setup.working_directory
        extension = self.experiment.system_setup.output_extension
        files = getfiles(directory, extension)
        # filesextension = getfiles2(directory,extension)
        data = files_to_list(files)
        return data

    @staticmethod
    def create_points_based_on_method(data):
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
        # do something with data
        new_points = data
        return new_points


class Executor(Core):
    def run(self):
        pass
