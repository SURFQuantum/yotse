""" Defines the run"""
import csv
import os
import pandas as pd
from pre import Experiment
from qcg.pilotjob.api.manager import LocalManager
from qcg.pilotjob.api.job import Jobs


def createpoints(parameters):
	return [3,4,5]

def qcgpilotcommand(general):
    """Create exec and args for qcgpilotjob based on info provided by user
    Parameters
    ----------
        general: dict
    Return
    ------
        exec: string
            program executioner, e.g python, qcc, 
        args: list 
            [programname, linearguments]
    """
    argsprogram = []
    execprogram = general[0]
    argsprogram.append(general[1])
    print(type(execprogram))
    print(argsprogram)
    return execprogram, argsprogram

def getfiles2(directory, extension)
    all_filenames = [i for i in glob.glob('{}.{}'.format(directory,extension))]

def getfiles(directory, extension):  
    dir_path = directory
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(extension):
        files.append(file)
    return files

def filestolist(files, extension):
    data = []
    for file in file:
        if extension is "csv":
            filedata = pd.read_csv(csvfile)
            data.append(filedata)
    return data



class Core:
	"""Defines the default run function for the executor
	Parameters
	----------
	experiment : Experiment
		Experiment to run.
	"""
	def __init__(self, experiment):
		self.experiment = experiment

	def run(self):
		sumbit()
		collectdata(extension="csv") 
		create_points_based_on_method()


	def submit(self):
		manager = LocalManager()
		extension = self.experiment.extension
		directory = self.experiment.directory
		jobs = Jobs()
		for i, item in enumerate(self.experiment.points):
			jobs.add(
				name = self.experiment.name + str(i), 
				exec = self.experiment.executor,
                args = [self.experiment.program_name, self.experiment.command_line_arguments],  
				stdout = self.experiment.stdout + str(i) + "." + extension,
				wd = directory,
			)
		job_ids = manager.submit(jobs)
		manager.wait4(job_ids)
		manager.cleanup()
		manager.finish()

	def collectdata(self):
		directory = self.experiment.directory
		extension = self.experiment.extension
    	filesextension = getfiles(directory,extension)
		#filesextension = getfiles2(directory,extension)

    	data = filestolist(filesextension, extension)
    	return data

	def create_points_based_on_method(self):
		#call optimization(experiments)
    	parameters = [5,5,5]
    	return parameters


class Executor(Core):
	def run():
		pass