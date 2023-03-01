import csv
import pandas as pd
import os
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


def csvfileslist(directory, extension):
    """ Creates a sublist of the files in directory based on a file type"""
    #all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    filesdir =  os.listdir(directory)
    newlist = []
    for names in filesdir:
        if names.endswith(extension):
            newlist.append(names)
    return newlist

def getcsvfiles():
    directory = os.getcwd()
    csvs = csvfileslist(directory, ".csv")
    return csvs

def csvstolist(csvs):
    data = []
    for csvfile in csvs:
        csvdata = pd.read_csv(csvfile)
        data.append(csvdata)
    return data

def submit(points, general):
	manager = LocalManager()
	# Loop over runs (points)
	jobs = Jobs()
	for i, item in enumerate(points):
	    #Fix: qcgpilotcommand(general)
		jobs.add(
				name = 'test_' + str(i), 
				exec = 'python',
                args = ['mytest.py', i],  
				stdout ="out" + str(i) + ".csv"
				)
    
	job_ids = manager.submit(jobs)
	manager.wait4(job_ids)
	manager.cleanup()
	manager.finish()


def collectresults():
    csvs = getcsvfiles()
    data = csvstolist(csvs)
    return data

#class?
def optimization(data):
    parameters = [5,5,5]
    return parameters

