#!/usr/local/bin/python

import sys
import numpy as np

with open(sys.argv[1]) as datafile:
	data = datafile.read()
	datafile.close()

data = data.split("\n")
data.pop(-1)

# for each pair of trajectories A and B in data

	# if at least the last two points of A are the first two points of B
		# change tid of B to match A and and remove the duplicate points 
		# can do this to the original data because compound stitches are allowed

# write the new data back to file