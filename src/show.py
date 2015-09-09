#!/usr/local/bin/python

''' show.py

    command line accessible 2D plot of any space separated two column text file
'''

import sys
import matplotlib.pyplot as plt
import plotting as plot

filename = sys.argv[1]

with open(filename) as datafile:
    data = datafile.read()
    datafile.close()

data = data.split('\n')
if data[-1] in ['\n', '\r\n', '']:
    data.pop(-1)

x = [row.split()[0] for row in data]
y = [row.split()[1] for row in data]

fig = plt.figure('Show.py')
ax = plt.axes()
# ax.set_title("Simple 2D Plot", y=1.03)
ax.set_xlabel("Graphical X")
ax.set_ylabel("Graphical Y")
ax.plot(x, y, 'k.')

plt.show()
