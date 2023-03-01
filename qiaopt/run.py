#!/usr/bin/env
from tools import collectresults
from tools import optimization
from tools import createpoints
from tools import submit

def run(parameters, general):
	points = createpoints(parameters)
	submit(points, general)
	data = collectresults()

	optsteps = 3
	for step in range(0,optsteps):
		print(step)
		newpoints = optimization(data)
		submit(newpoints, general)
		data = collectresults()
	

if __name__ == "__main__":
	parameters = [(0,0),(1,1),(2,2)]
	general = ['python','mytest']
	run(parameters, general)
