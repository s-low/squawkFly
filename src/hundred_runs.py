#!/usr/local/bin/python

''' hundred_runs.py

Does what it says on the tin.

Tiny script to run the reconstruction over and over during the analysis.
'''

import os.path

for x in xrange(0, 100):
    os.system("./reconstruct.py depthAnalysis2 3.0 3.0")
