#!/usr/local/bin/python

''' analyse_separations.py
- This is a one-off utility written to aide with the analysis
- Input: separations.txt or other similarly formatted
- Format: newline separated list of lists of 3-space dist between
reconstructions and ground truth. Lists are separated by double newline.
- Output: The mean separation for each indexed point and std dev.
'''


import os.path
import sys
import numpy as np

folder = os.path.dirname(sys.argv[1])

datafile = open(sys.argv[1], 'r')
all_data = datafile.read()
datafile.close()

runs = all_data.split('\n\n')
num_runs = len(runs)

first = runs[0]
first = first.split('\n')

# gobble blank lines at SOF/EOF
if first[0] in ['\n', '\r\n', '']:
    first.pop(0)
if first[-1] in ['\n', '\r\n', '']:
    first.pop(-1)

length = len(first)

print length

outfile = open(os.path.join(folder, 'mean_separations.txt'), 'w')

# for each trajectory point
for i in xrange(0, length):
    print "trajectory point:", i

    these_seps = []

    # get the ith point of each run
    for index, r in enumerate(runs):
        seps = r.split('\n')

        # gobble blank lines at SOF/EOF
        if seps[0] in ['\n', '\r\n', '']:
            seps.pop(0)

        if seps[-1] in ['\n', '\r\n', '']:
            seps.pop(-1)

        sep = float(seps[i])
        print index, sep
        these_seps.append(sep)

    mean = np.mean(these_seps)
    std = np.std(these_seps)
    print "STATS:", mean, std
    outfile.write(str(mean) + ' ' + str(std) + '\n')
    these_seps = []

outfile.close()
