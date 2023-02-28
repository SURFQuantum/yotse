from qcg.pilotjob.api.manager import LocalManager
from qcg.pilotjob.api.job import Jobs

def createpoints(parameters):
	return [3,4,5]

def submit(points, general):
	manager = LocalManager()
	# Loop over runs (points)
	jobs = Jobs()
	for i, item in enumerate(points):
		print(i)
		names = []
		jobs.add(name = 'test_' + str(i), exec = 'python3',args = [general[0], item], stdout="out"+str(i))
	print(jobs)
	job_ids = manager.submit(jobs)
	print(job_ids)
	manager.wait4(job_ids)
	manager.cleanup()
	manager.finish()


def main(parameters, general):
	points = createpoints(parameters)
	submit(points, general)
#	data = collectresults()
#	newpoints = optimization(data)
#	submit(newpoints, general)


if __name__ == "__main__":
	parameters = [0,1,2]
	general = ['mytest']
	main(parameters, general)
